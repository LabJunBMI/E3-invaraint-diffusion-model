from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from transformers import BertConfig

from tqdm.auto import tqdm
import pandas as pd
import torch
import gc

from dataset import LigandBindingSiteDataset, AA_VOCAB
from model import PeptideDiff
from utils import (
    PredefinedNoiseScheduleDiscrete, 
    DiscreteUniformTransition,
    BlosumTransition
)

gc.enable()
GPU_ID=4
DEVICE = torch.device(f"cuda:{GPU_ID}")
THREAD_NUM=16
DATA_PATH = "./data/biolip.pt"
MODEL_PATH = "" # The file path of trained model
OUTPUT_PATH = "./data/from_generated_angles/output.pkl"

# Config this accordingly
# The ext 1 model is traind with "max_seq_len" of 64, others are trained with 128
CONFIG = {
    "pocket_ext":0,
    "timesteps": 50,
    "max_seq_len": 64,
    "noise_schedule":"cosine",
    
    "num_heads": 12,
    "dropout_p": 0.1,
    "hidden_size": 768,
    "num_hidden_layers": 6,
    "intermediate_size": 1024,
    "position_embedding_type": "relative_key",
    
    "lr": 5e-5,
    "l2_norm": 0.1,
    "loss": "smooth_l1",
    "gradient_clip": 1.0,
    "lr_scheduler": "LinearWarmup",
    
    "min_epochs": 100,
    "max_epochs": 150,
    "batch_size": 64,
}


def get_dataloader(file_path):
    test_angle_ds = LigandBindingSiteDataset(
        file_path,
        "test", 
        CONFIG["max_seq_len"],
        CONFIG["pocket_ext"]
    )
    test_dataloader = DataLoader(
        dataset=test_angle_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=THREAD_NUM,
    )
    return test_dataloader

def get_model(steps_per_epoch)->PeptideDiff:
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
    model = PeptideDiff(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        feature_names= LigandBindingSiteDataset.feature_names,
        max_epochs=CONFIG["max_epochs"],
        lr_scheduler=CONFIG["lr_scheduler"],
        l2_lambda=CONFIG["l2_norm"],
        steps_per_epoch=steps_per_epoch,
        learning_rate=CONFIG["lr"],
        loss_func=torch.nn.CrossEntropyLoss(),
        noise_schedule=CONFIG["noise_schedule"],
        timesteps=CONFIG["timesteps"],
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.eval().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters")
    return model

def generate_discrete_noise(batch_size, length, num_classes=20):
    random_indices = torch.randint(0, num_classes, (batch_size, length))
    one_hot_matrix = torch.zeros(batch_size, length, num_classes)
    one_hot_matrix[torch.arange(batch_size).unsqueeze(1), torch.arange(length), random_indices] = 1
    return one_hot_matrix.to(DEVICE)


# !!! Work on Following Functions !!!
def compute_batched_over0_posterior_distribution(X_t,Q_t,Qsb,Qtb,batch):
    """ M: X or E
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0 
    X_t: bs, n, dt          or bs, n, n, dt
    Qt: bs, d_t-1, dt
    Qsb: bs, d0, d_t-1
    Qtb: bs, d0, dt.
    """
    #X_t is a sample of q(x_t|x_t+1)
    Qt_T = Q_t.transpose(-1,-2)
    X_t_ = X_t.unsqueeze(dim = -2)
    left_term = X_t_ @ Qt_T[batch] #[N,1,d_t-1]
    # left_term = left_term.unsqueeze(dim = 1) #[N,1,dt-1]
    right_term = Qsb[batch] #[N,d0,d_t-1]
    numerator = left_term * right_term #[N,d0,d_t-1]
    prod = Qtb[batch] @ X_t.unsqueeze(dim=2) # N,d0,1
    denominator = prod
    denominator[denominator == 0] = 1e-6        
    out = numerator/denominator
    return out

def sample_p_zs_given_zt_discrete(
    t, s, noised_data, pred_noise,
    noise_schedule, transition, diverse, is_last_step):
    """
    sample zs~p(zs|zt)
    """
    if is_last_step:
        return pred_noise
    batch_size, seq_len, num_class = noised_data.shape
    repeat_idx = torch.tensor([[i]*seq_len for i in range(batch_size)]).reshape(-1)
    # reshape to fit legacy codes
    noised_data = noised_data.reshape(
            batch_size*seq_len, num_class)
    pred_noise = pred_noise.reshape(
            batch_size*seq_len, num_class)
    alpha_t_bar = noise_schedule.get_alpha_bar(t_normalized=t)
    alpha_s_bar = noise_schedule.get_alpha_bar(t_normalized=s)
    Qtb = transition.get_Qt_bar(alpha_t_bar, DEVICE)
    Qsb = transition.get_Qt_bar(alpha_s_bar, DEVICE)
    Qt = (Qsb/Qtb)/(Qsb/Qtb).sum(dim=-1).unsqueeze(dim=2) #approximate
    
    pred_X = F.softmax(pred_noise,dim = -1) 
    p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(
        X_t=noised_data, Q_t=Qt, Qsb=Qsb, Qtb=Qtb, batch=repeat_idx)#[N,d0,d_t-1] 20,20 approximate Q_t-s with Qt 
    weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X #[N,d0,d_t-1]
    unnormalized_prob_X = weighted_X.sum(dim=1)             #[N,d_t-1]
    unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
    prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  #[N,d_t-1]
    sample_s = []
    for prob in prob_X:
        if(prob.sum()!=0 and diverse):
            sample_s.append(prob.multinomial(1)[0])
        elif(prob.sum()!=0 and (not diverse)):
            sample_s.append(prob.argmax())
        else:
            sample_s.append(0)
    sample_s = torch.Tensor(sample_s).reshape(batch_size, seq_len).long()
    X_s = F.one_hot(sample_s, num_classes = num_class).float().to(DEVICE)
    return X_s

@torch.no_grad
def denoise(batch, model:PeptideDiff, noise_schedule, transition, diverse):
    batch_size, max_len, num_class = batch["ligand_seq"].shape
    ligand_seq_noise = generate_discrete_noise(batch_size, max_len, num_class)
    ligand_seq = batch["ligand_seq"].to(DEVICE)
    ligand_mask = batch["ligand_attn_mask"].to(DEVICE)
    ligand_angles = batch["ligand_angles"].to(DEVICE)
    receptor_seq = batch["receptor_seq"].to(DEVICE)
    receptor_angles = batch["receptor_angles"].to(DEVICE)
    receptor_attn_mask = batch["receptor_attn_mask"].to(DEVICE)
    # denoise loop
    for s_int in tqdm(list(reversed(range(0, CONFIG["timesteps"], 1)))):
    # for s_int in list(reversed(range(0, CONFIG["timesteps"]))):
        s_array = s_int * torch.ones((batch_size, 1))
        s_norm = s_array / CONFIG["timesteps"]
        t_array = s_array + 1
        t_norm = t_array / CONFIG["timesteps"]
        # predict noise
        pred_noise = model.forward(
            s_array.to(DEVICE),
            ligand_seq_noise, ligand_angles, ligand_mask,
            receptor_seq, receptor_angles, receptor_attn_mask,
        )
        ligand_seq_noise  = sample_p_zs_given_zt_discrete(
            t_norm, s_norm, ligand_seq_noise, pred_noise, 
            noise_schedule, transition, diverse, is_last_step=s_int==0
        )
    recovery_rates = []
    pred_sequences = []
    true_sequences = []
    structure_ids = []
    for i in range(batch_size):
        pred_seq = ligand_seq_noise[i].argmax(dim=1)
        true_seq = ligand_seq[i].argmax(dim=1)
        mask = ligand_mask[i].bool()
        # calculate recovery rate
        recovery_rate = pred_seq[mask] == true_seq[mask]
        recovery_rate = recovery_rate.sum()/mask.sum()
        recovery_rates.append(recovery_rate.item())
        # Recover tokens
        pred_sequences.append(
            "".join([AA_VOCAB[i] for i in pred_seq[mask]]))
        true_sequences.append(
            "".join([AA_VOCAB[i] for i in true_seq[mask]]))
        structure_ids.append(
            f'{batch["structure_ids"]["pdb_id"][i]}_{batch["structure_ids"]["ligand_chain"][i]}'
        )
    print(sum(recovery_rates)/len(recovery_rates))
    return structure_ids, true_sequences, pred_sequences, recovery_rates

if __name__ == "__main__":
    torch.set_num_threads(THREAD_NUM)
    torch.set_float32_matmul_precision("medium")
    testing_dataloader = get_dataloader(DATA_PATH)
    model = get_model(len(testing_dataloader))
    noise_schedule = PredefinedNoiseScheduleDiscrete(CONFIG["noise_schedule"], CONFIG["timesteps"]).to(DEVICE)
    transition = BlosumTransition(x_classes=20)
    structure_ids = []
    true_sequences = []
    pred_sequences = []
    recovery_rates = []
    for idx, batch in enumerate(testing_dataloader):
        print(f"Generating Batch {idx}")
        ids, true_seq, pred_seq, rec_rates = denoise(
            batch, model, noise_schedule, transition, True)
        structure_ids.extend(ids)
        recovery_rates.extend(rec_rates)
        pred_sequences.extend(pred_seq)
        true_sequences.extend(true_seq)
    res = pd.DataFrame(
        zip(structure_ids,
            true_sequences, 
            pred_sequences, 
            recovery_rates), 
        columns=["structure_ids", "true_sequence", "predict_sequence", "recovery_rate"]
    )
    res.to_pickle(OUTPUT_PATH)
    print(res)
        

"""
from sample import *
torch.set_num_threads(THREAD_NUM)
torch.set_float32_matmul_precision("medium")
testing_dataloader = get_dataloader(DATA_PATH)
model = get_model(len(testing_dataloader))
noise_schedule = PredefinedNoiseScheduleDiscrete(CONFIG["noise_schedule"], CONFIG["timesteps"]).to(DEVICE)
transition = DiscreteUniformTransition(20)
for batch in testing_dataloader:
    break

denoise(batch, model, noise_schedule, transition, False)

for data in train_dataloader:
    data = data.to(device)
    break

denoise(data, model, False)

for i in train_dataloader:
    i = i.to(device)
    print(denoise(i, model, False))

for data in testing_dataloader:
    data = data.to(device)
    break

"""