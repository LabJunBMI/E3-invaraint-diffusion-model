import torch
from torch import nn
from transformers import BertConfig

from model import ConditionalBertForDiffusion
from dataset import LigandBindingSiteDataset, NoisedAnglesDataset
from utils import compute_alphas, modulo_with_wrapped_range

import pickle
from tqdm.auto import tqdm

MODEL_PATH = "" # The path of trained model
OUTPUT = "./data/output.pkl" # the sampled angles, can be transfom to pdb file by "create_pdb.py"
DATA_FILE = "./data/biolip.pt"
GPU_ID = 3
STEP = 1 # step size, a larger number can increase the sample speed but with lower performance

# Config this accordingly
# The ext 1 model is traind with "max_seq_len" of 64, others are trained with 128
CONFIG = {
    "pocket_ext":0,
    "timesteps": 1000,
    "max_seq_len": 64,

    "num_heads": 12,
    "dropout_p": 0.1,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "intermediate_size": 1024,
    "position_embedding_type": "relative_key",

    "lr": 5e-5,
    "l2_norm": 0.1,
    "loss": "smooth_l1",
    "gradient_clip": 1.0,
    "lr_scheduler": "LinearWarmup",

    "min_epochs": 500,
    "max_epochs": 1000,
    "batch_size": 64,
}


# Get cpu, gpu or mps device for training.
DEVICE = (
    f"cuda:{GPU_ID}"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.cuda.set_device(GPU_ID)
print(f"Using {DEVICE} device")

@torch.no_grad()
def p_sample(
    model: ConditionalBertForDiffusion,

    ligand_mask: torch.Tensor,
    ligand_angle_noise: torch.Tensor,
    receptor_seq:torch.Tensor,
    receptor_mask:torch.Tensor,
    receptor_angle:torch.Tensor,

    timestep: int,
    betas: torch.Tensor,
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)
    """
    # Calculate alphas and betas
    alpha_beta_values = compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])
    # Select based on time
    t_unique = torch.unique(timestep)
    assert len(t_unique) == 1, f"Got multiple values for t: {t_unique}"
    t_index = t_unique.item()
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_index]
    betas_t = betas[t_index]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][t_index]

    model_output = model(
        timestep, ligand_angle_noise, ligand_mask, 
        receptor_seq, receptor_angle, receptor_mask, 
    )
    model_mean = sqrt_recip_alphas_t * (
        ligand_angle_noise - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
    )
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        ligand_angle_noise = torch.randn_like(ligand_angle_noise)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * ligand_angle_noise

@torch.no_grad()
def p_sample_loop(
    model: nn.Module,

    ligand_mask: torch.Tensor,
    ligand_angle_noise: torch.Tensor,
    receptor_seq:torch.Tensor,
    receptor_mask:torch.Tensor,
    receptor_angle:torch.Tensor,

    total_timesteps: int,
    betas: torch.Tensor,
    disable_pbar: bool = False,
) -> torch.Tensor:
    """
    Returns a tensor of shape (timesteps, batch_size, seq_len, n_ft)
    """
    b = ligand_angle_noise.shape[0]
    noises = []
    for i in tqdm(
        reversed(range(0, total_timesteps, STEP)),
        desc="sampling loop time step",
        total=int(total_timesteps/STEP),
        disable=disable_pbar,
    ):
        # Shape is (batch, seq_len, 4)
        ligand_angle_noise = p_sample(
            model=model,

            ligand_mask=ligand_mask,
            ligand_angle_noise=ligand_angle_noise,
            receptor_seq=receptor_seq,
            receptor_mask=receptor_mask,
            receptor_angle=receptor_angle,

            timestep=torch.full((b,), i, device=DEVICE, dtype=torch.long),  # time vector
            betas=betas,
        )
        # Wrap if angular
        ligand_angle_noise = modulo_with_wrapped_range(
            ligand_angle_noise, range_min=-torch.pi, range_max=torch.pi
        )
        noises.append(ligand_angle_noise.cpu())
    return torch.stack(noises)

def get_dataset(file_path):
    test_angle_ds = LigandBindingSiteDataset(
        file_path,
        "test", 
        CONFIG["max_seq_len"],
        CONFIG["pocket_ext"],
    )
    test_angle_ds = NoisedAnglesDataset(
        test_angle_ds,
        timesteps=CONFIG["timesteps"],
    )
    return test_angle_ds

def load_model(dataset):
    encoder_config = BertConfig(
        max_position_embeddings=CONFIG["max_seq_len"],
        num_attention_heads=CONFIG["num_heads"],
        hidden_size=CONFIG["hidden_size"],
        intermediate_size=CONFIG["intermediate_size"],
        num_hidden_layers=CONFIG["num_hidden_layers"],
        position_embedding_type=CONFIG["position_embedding_type"],
        hidden_dropout_prob=CONFIG["dropout_p"],
        attention_probs_dropout_prob=CONFIG["dropout_p"],
        use_cache=False,
    )
    decoder_config = BertConfig(
        max_position_embeddings=CONFIG["max_seq_len"],
        num_attention_heads=CONFIG["num_heads"],
        hidden_size=CONFIG["hidden_size"],
        intermediate_size=CONFIG["intermediate_size"],
        num_hidden_layers=CONFIG["num_hidden_layers"],
        position_embedding_type=CONFIG["position_embedding_type"],
        hidden_dropout_prob=CONFIG["dropout_p"],
        attention_probs_dropout_prob=CONFIG["dropout_p"],
        use_cache=False,
        is_decoder=True,
        add_cross_attention=True
    )
    model = ConditionalBertForDiffusion(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        feature_names= dataset.feature_names,
        epochs=CONFIG["max_epochs"],
        lr_scheduler=CONFIG["lr_scheduler"],
        l2_lambda=CONFIG["l2_norm"],
        steps_per_epoch=len(dataset),
        learning_rate=CONFIG["lr"],
        loss_func=[ConditionalBertForDiffusion.diheral_loss_func
        ]*3 + [ConditionalBertForDiffusion.angle_loss_func]*3,
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.eval().to(DEVICE)
    return model

def sample(model, test_angle_ds):
    def chunkify_features(dataset, feature_name):
        feats = [dataset[i][feature_name] for i in range(len(dataset))]
        feats = [torch.stack(feats[i : i + CONFIG["batch_size"]]) for i in range(0, len(dataset), CONFIG["batch_size"])]
        return feats
    ligand_mask = chunkify_features(test_angle_dataset, "ligand_attn_mask")
    receptor_angle = chunkify_features(test_angle_dataset, "receptor_angles")
    receptor_seq = chunkify_features(test_angle_dataset, "receptor_seq")
    receptor_mask = chunkify_features(test_angle_dataset, "receptor_attn_mask")
    pad, feature_size = test_angle_ds[0]["ligand_angles"].shape
    ligand_len = [m.sum(dim=1).int() for m in ligand_mask]
    retval = []
    for idx, this_lengths in enumerate(ligand_len):
        print(f"Generating Batch {idx}/{len(ligand_len)}")
        batch = len(this_lengths)
        # Sample noise and sample the lengths
        lignad_angle_noise = test_angle_ds.sample_noise(
            torch.zeros((batch, pad, feature_size), dtype=torch.float32))
        sampled = p_sample_loop(
            model=model,

            ligand_mask=ligand_mask[idx].to(DEVICE),
            ligand_angle_noise=lignad_angle_noise.to(DEVICE),
            receptor_seq=receptor_seq[idx].to(DEVICE),
            receptor_mask=receptor_mask[idx].to(DEVICE),
            receptor_angle=receptor_angle[idx].to(DEVICE),

            total_timesteps=test_angle_ds.timesteps,
            betas=test_angle_ds.alpha_beta_terms["betas"],
            disable_pbar=False,
        )
        # Gets to size (timesteps, seq_len, n_ft)
        trimmed_sampled = [
            sampled[:, i, :l, :].numpy() for i, l in enumerate(this_lengths)
        ]
        # trimmed_sampled = [s[-1] for s in trimmed_sampled] # extract last time step
        retval.extend(trimmed_sampled)
        break
    return retval

if __name__ == "__main__":
    test_angle_dataset = get_dataset(DATA_FILE)
    model = load_model(test_angle_dataset)
    sample_result = sample(model, test_angle_dataset)
    with open(OUTPUT, "+wb") as f:
        pickle.dump(sample_result, f)