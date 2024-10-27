import pytorch_lightning as pl
import functools
from typing import (
    Any,
    Dict,
    List,
    Union,
)

import torch
from torch import nn
from torch.nn import functional as F

from transformers.activations import get_activation
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertConfig,
    BertAttention
)


from utils import radian_l1_loss, radian_smooth_l1_loss


# self-defined modules
class SELayer(nn.Module):
    # according to the paper: https://arxiv.org/pdf/2401.13858
    def __init__(self, bert_config, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(bert_config.hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(bert_config.hidden_size, elementwise_affine=False)
        
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(bert_config.hidden_size, 6 * bert_config.hidden_size, bias=True)
        )
        
        self.attn = BertAttention(bert_config, **block_kwargs)

        self.mlp = nn.Sequential(
            nn.Linear(bert_config.hidden_size, int(bert_config.hidden_size*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(bert_config.hidden_dropout_prob),
            nn.Linear(int(bert_config.hidden_size*mlp_ratio), bert_config.hidden_size),
            nn.Dropout(bert_config.hidden_dropout_prob),
        )

        nn.init.zeros_(self.adaLN_modulation[0].weight)
        nn.init.zeros_(self.adaLN_modulation[0].bias)

    def forward(self, x, c, mask):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self._modulate(self.norm1(self.attn(x, mask)[0]), shift_msa, scale_msa)
        x = x + gate_mlp * self._modulate(self.norm2(self.mlp(x)), shift_mlp, scale_mlp)
        return x

    def _modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    Built primarily for score-based models.

    Source:
    https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, embed_dim: int = 384, scale: float = 2 * torch.pi):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        w = torch.randn(embed_dim // 2) * scale
        assert w.requires_grad == False
        self.register_buffer("W", w)

    def forward(self, x: torch.Tensor):
        """
        takes as input the time vector and returns the time encoding
        time (x): (batch_size, )
        output  : (batch_size, embed_dim)
        """
        if x.ndim > 1:
            x = x.squeeze()
        elif x.ndim < 1:
            x = x.unsqueeze(0)
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        embed = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return embed

class BertEmbeddings(nn.Module):
    """
    Adds in positional embeddings if using absolute embeddings, adds layer norm and dropout
    """

    def __init__(self, in_features, bert_config:BertConfig):
        super().__init__()
        self.linear = nn.Linear(in_features, bert_config.hidden_size)
        self.LayerNorm = nn.LayerNorm(bert_config.hidden_size, eps=bert_config.layer_norm_eps)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

    def forward(
        self,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.linear(input_embeds)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class AnglesPredictor(nn.Module):
    """
    Predict angles from the embeddings. For BERT, the MLM task is done using an
    architecture like
    d_model -> dense -> d_model -> activation -> layernorm -> dense -> d_output
    https://github.com/huggingface/transformers/blob/v4.21.1/src/transformers/models/bert/modeling_bert.py#L681

    activation should be given as nn.ReLU for example -- NOT nn.ReLU()
    """

    def __init__(
        self,
        d_model: int,
        d_out: int = 4,
        activation: Union[str, nn.Module] = "gelu",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.dense1 = nn.Linear(d_model, d_model)

        if isinstance(activation, str):
            self.dense1_act = get_activation(activation)
        else:
            self.dense1_act = activation()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)
        self.dense2 = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.dense1_act(x)
        x = self.layer_norm(x)
        x = self.dense2(x)
        return x

# main model
class ConditionalBertForDiffusionBase(nn.Module):
    def __init__(
        self, 
        encoder_config:BertConfig, 
        decoder_config:BertConfig, 
        feature_size:int
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        # Encoder: Encode receptor features
        self.receptor_seq_emb = BertEmbeddings(20, encoder_config)
        self.receptor_angle_emb = BertEmbeddings(feature_size, encoder_config)
        self.receptor_emb = SELayer(encoder_config)# Match with encoder shape
        self.encoder = BertEncoder(encoder_config)
        # Decoder: De-noise peptide angles
        self.ligand_angle_emb = BertEmbeddings(feature_size, decoder_config)
        self.timestep_projector = GaussianFourierProjection(
            decoder_config.hidden_size)
        self.timestep_emb = SELayer(decoder_config)
        self.decoder = BertEncoder(decoder_config)
        # Timestep projector
        self.angles_predictor = AnglesPredictor(decoder_config.hidden_size, feature_size)
    def forward(self, timestep,
                noised_ligand_angles, ligand_attention_masks, 
                receptor_seq, receptor_angles, receptor_attention_masks, 
                ligand_pos_ids=None, receptor_pos_ids=None):
        # Create position ids (0~len) if not given
        ligand_pos_ids = self._create_pos_ids(noised_ligand_angles) if ligand_pos_ids is None else ligand_pos_ids
        receptor_pos_ids = self._create_pos_ids(receptor_angles) if receptor_pos_ids is None else receptor_pos_ids
        # Extend attention masks
        ligand_attention_masks = self._exetend_attention_mask(ligand_attention_masks)
        receptor_attention_masks = self._exetend_attention_mask(receptor_attention_masks)
        # Encode receptor angles
        receptor_angles = self.receptor_angle_emb(receptor_angles)
        receptor_seq = self.receptor_seq_emb(receptor_seq)
        receptor_embedding = self.receptor_emb(
            receptor_angles, receptor_seq, 
            receptor_attention_masks
        )
        encoder_outputs = self.encoder(
            receptor_embedding,
            attention_mask=receptor_attention_masks,
        ).last_hidden_state
        # De-noise ligand angles
        ligand_embedding = self.ligand_angle_emb(noised_ligand_angles)
        ligand_timestep = self.timestep_projector(timestep.squeeze(dim=-1)).unsqueeze(1)
        ligand_embedding_with_time = self.timestep_emb(
            ligand_embedding, ligand_timestep,
            ligand_attention_masks
        )
        decoder_output = self.decoder(
            hidden_states=ligand_embedding_with_time,
            attention_mask=ligand_attention_masks,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=receptor_attention_masks,
        ).last_hidden_state
        output = self.angles_predictor(decoder_output)
        return output

    def _create_pos_ids(self, sequences:torch.Tensor):
        batch_size, seq_length, *_  = sequences.size()
        position_ids = torch.arange(
            seq_length,
        ).expand(
            batch_size, -1
        )
        return position_ids

    def _exetend_attention_mask(self, mask):
        # From hugggingface modeling_utils
        extended_attention_mask = mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.type_as(mask)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

class ConditionalBertForDiffusion(ConditionalBertForDiffusionBase, pl.LightningModule):
    """
    Wraps model by pl LightningModule
    """
    diheral_loss_func = radian_l1_loss
    angle_loss_func = functools.partial(
        radian_smooth_l1_loss, beta=torch.pi / 10)
    def __init__(
        self,
        encoder_config:BertConfig,
        decoder_config:BertConfig,
        feature_names: List[str],
        loss_func:List,
        epochs: int = 1,
        lr_scheduler = None,
        l2_lambda: float = 0.0,
        steps_per_epoch: int = 250,
        learning_rate: float = 5e-5,
        **kwargs,
    ):
        """Feed args to BertForDiffusionBase and then feed the rest into"""
        ConditionalBertForDiffusionBase.__init__(self, encoder_config, decoder_config, len(feature_names))
        # Store information about leraning rates and loss
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        self.feature_names = feature_names
        self.lr_scheduler = lr_scheduler
        self.train_epoch_losses = []
        self.valid_epoch_losses = []
        self.loss_func = loss_func
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.train_epoch_counter=0
    def _get_loss_terms(
        self, batch
    ) -> List[torch.Tensor]:
        """
        Returns the loss terms for the model. Length of the returned list
        is equivalent to the number of features we are fitting to.
        """
        known_noise = batch["known_noise"]
        predicted_noise = self.forward(
            timestep=batch["timestep"],

            noised_ligand_angles=batch["noised_ligand_angle"],
            ligand_attention_masks=batch["ligand_attn_mask"],

            receptor_seq=batch["receptor_seq"],
            receptor_angles=batch["receptor_angles"],
            receptor_attention_masks=batch["receptor_attn_mask"],

            ligand_pos_ids=batch["ligand_pos_id"],
            receptor_pos_ids=batch["receptor_pos_id"]
        )
        assert ( known_noise.shape == predicted_noise.shape
        ), f"{known_noise.shape} != {predicted_noise.shape}"
        # Indexes into batch then indices along sequence length
        # attn_mask has shape (batch, seq_len) --> where gives back
        # two lists of values, one for each dimension
        # known_noise has shape (batch, seq_len, num_fts)
        unmask_idx = torch.where(batch["ligand_attn_mask"])
        assert len(unmask_idx) == 2
        loss_terms = []
        for i in range(known_noise.shape[-1]):
            loss_fn = self.loss_func[i] if isinstance(
                self.loss_func, list) else self.loss_func
            loss_terms.append(loss_fn(
                predicted_noise[unmask_idx[0], unmask_idx[1], i],
                known_noise[unmask_idx[0], unmask_idx[1], i],
            ))
        return torch.stack(loss_terms)

    def training_step(self, batch, batch_idx):
        """
        Training step, runs once per batch
        """
        loss_terms = self._get_loss_terms(batch)
        avg_loss = torch.mean(loss_terms)
        pseudo_ft_names = self.feature_names
        assert len(loss_terms) == len(pseudo_ft_names)
        loss_dict = {
            f"train_loss_{val_name}": val
            for val_name, val in zip(pseudo_ft_names, loss_terms)
        }
        loss_dict["train_loss"] = avg_loss
        self.log_dict(loss_dict)  # Don't seem to need rank zero or sync dist
        return avg_loss

    def on_train_batch_end(self, outputs, batch_idx, dataloader_idx=0) -> None:
        """Log the average training loss over the epoch"""
        # pl.utilities.rank_zero_info(outputs)
        self.train_epoch_losses.append(float(outputs["loss"]))

    def on_train_epoch_end(self) -> None:
        pl.utilities.rank_zero_info(
            f"Traning Loss:{sum(self.train_epoch_losses)/len(self.train_epoch_losses)}"
        )
        self.train_epoch_losses = []
        self.train_epoch_counter += 1

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Validation step
        """
        with torch.no_grad():
            loss_terms = self._get_loss_terms(batch)
        avg_loss = torch.mean(loss_terms)
        # Log each of the loss terms
        pseudo_ft_names = self.feature_names
        assert len(loss_terms) == len(pseudo_ft_names)
        loss_dict = {
            f"val_loss_{val_name}": self.all_gather(val).mean()
            for val_name, val in zip(pseudo_ft_names, loss_terms)
        }
        loss_dict["val_loss"] = avg_loss
        # with rank zero it seems that we don't need to use sync_dist
        self.log_dict(loss_dict, rank_zero_only=True)
        return {"val_loss": avg_loss}
    
    def on_validation_epoch_end(self) -> None:
        pl.utilities.rank_zero_info(
            f"Validation Loss:{sum(self.valid_epoch_losses)/len(self.valid_epoch_losses)}"
        )

    def on_validation_batch_end(self, outputs, batch_idx, dataloader_idx=0) -> None:
        """Log the average validation loss over the epoch"""
        self.valid_epoch_losses.append(float(outputs["val_loss"]))

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Return optimizer. Limited support for some optimizers
        """
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )
        retval = {"optimizer": optim}
        if self.lr_scheduler:
            if self.lr_scheduler == "OneCycleLR":
                retval["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optim,
                        max_lr=1e-2,
                        epochs=self.epochs,
                        steps_per_epoch=self.steps_per_epoch,
                    ),
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "step",
                }
            elif self.lr_scheduler == "LinearWarmup":
                # https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
                # Transformers typically do well with linear warmup
                warmup_steps = int(self.epochs * 0.1)
                pl.utilities.rank_zero_info(
                    f"Using linear warmup with {warmup_steps}/{self.epochs} warmup steps"
                )
                retval["lr_scheduler"] = {
                    "scheduler": get_linear_schedule_with_warmup(
                        optim,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=self.epochs,
                    ),
                    "frequency": 1,
                    "interval": "epoch",  # Call after 1 epoch
                }
            else:
                raise ValueError(f"Unknown lr scheduler {self.lr_scheduler}")
        pl.utilities.rank_zero_info(f"Using optimizer {retval}")
        return retval
