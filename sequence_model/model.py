import pytorch_lightning as pl
from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F

from transformers.activations import get_activation
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertEncoder,
    BertConfig,
)

from utils import elbo_loss

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
from utils import (
    PredefinedNoiseScheduleDiscrete, 
    DiscreteUniformTransition,
    BlosumTransition
)

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

class AminoAcidPredictor(nn.Module):
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
        # Timestep projector
        self.timestep_projector = GaussianFourierProjection(
            decoder_config.hidden_size)
        # Combine seq and angle
        self.ligand_seq_embedding = BertEmbeddings(20, encoder_config)
        self.ligand_angle_embedding = BertEmbeddings(8, encoder_config)
        self.ligand_feature_emb = SELayer(encoder_config)

        self.receptor_seq_embedding = BertEmbeddings(20, encoder_config)
        self.receptor_angle_embedding = BertEmbeddings(8, encoder_config)
        self.receptor_feature_emb = SELayer(encoder_config)
        # Decoder: De-noise peptide seq
        self.decoder = BertEncoder(decoder_config)
        self.decoder_normalize = SELayer(decoder_config)
        self.amino_acid_predictor = AminoAcidPredictor(decoder_config.hidden_size, feature_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _constant_init(module, i):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, i)
                if module.bias is not None:
                    nn.init.constant_(module.bias, i)

        self.apply(_basic_init)
        _constant_init(self.decoder_normalize.adaLN_modulation[0], 0)

    def forward(self, timestep,
                noised_ligand_seq, ligand_angle, ligand_attention_masks, 
                receptor_seq, receptor_angle, receptor_attention_masks, 
                ligand_pos_ids=None, receptor_pos_ids=None):
        # Create position ids (0~len) if not given
        ligand_pos_ids = self._create_pos_ids(noised_ligand_seq) if ligand_pos_ids is None else ligand_pos_ids
        receptor_pos_ids = self._create_pos_ids(receptor_seq) if receptor_pos_ids is None else receptor_pos_ids
        # Extend attention masks
        ligand_attention_masks = self._exetend_attention_mask(ligand_attention_masks)
        receptor_attention_masks = self._exetend_attention_mask(receptor_attention_masks)
        # combine ligand seq and angle
        denoise_timestep = self.timestep_projector(timestep.squeeze(dim=-1)).unsqueeze(1)
        noised_ligand_seq = self.ligand_seq_embedding(noised_ligand_seq)
        ligand_angle = self.ligand_angle_embedding(ligand_angle)+denoise_timestep
        ligand_feature = self.ligand_feature_emb(
            noised_ligand_seq, ligand_angle, 
            ligand_attention_masks
        )
        # combine receptor seq and angle
        receptor_seq = self.receptor_seq_embedding(receptor_seq)
        receptor_angle = self.receptor_angle_embedding(receptor_angle)+denoise_timestep
        receptor_feature = self.ligand_feature_emb(
            receptor_seq, receptor_angle,
            receptor_attention_masks
        )
        # De-noise ligand seq
        decoder_output = self.decoder(
            hidden_states=ligand_feature,
            attention_mask=ligand_attention_masks,
            encoder_hidden_states=receptor_feature,
            encoder_attention_mask=receptor_attention_masks,
        ).last_hidden_state
        decoder_output = self.decoder_normalize(
            decoder_output, denoise_timestep, 
            ligand_attention_masks
        )
        output = self.amino_acid_predictor(decoder_output)
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


class PeptideDiff(ConditionalBertForDiffusionBase, pl.LightningModule):
    def __init__(
        self,
        encoder_config:BertConfig,
        decoder_config:BertConfig,
        feature_names: List[str],
        loss_func:List,
        noise_schedule,
        timesteps,
        max_epochs: int = 1,
        lr_scheduler = None,
        l2_lambda: float = 0.0,
        steps_per_epoch: int = 250,
        learning_rate: float = 5e-5,
        **kwargs,
    ):
        ConditionalBertForDiffusionBase.__init__(self, encoder_config, decoder_config, len(feature_names))
        self.noise_schedule = noise_schedule
        self.timesteps = timesteps

        self.aa_transition_model = BlosumTransition(x_classes=20)
        self.discrete_noise_schedule = PredefinedNoiseScheduleDiscrete(
            noise_schedule=self.noise_schedule, 
            timesteps=self.timesteps
        )

        self.loss_function = loss_func
        self.lr = learning_rate
        self.l2_lambda = l2_lambda
        self.lr_scheduler = lr_scheduler
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.valid_epoch_losses = []
        self.train_epoch_losses = []

    def apply_aa_noise(self, ligand_seq, t_int):
        batch_size, seq_len, _ = ligand_seq.shape
        t_float = t_int / self.timesteps
        # calculating prob.
        ligand_seq = ligand_seq.reshape(
            batch_size*seq_len, ligand_seq.shape[-1])
        repeat_idx = torch.tensor([[i]*seq_len for i in range(batch_size)]).reshape(-1)
        alpha_t_bar = self.discrete_noise_schedule.get_alpha_bar(t_normalized=t_float)
        Qtb = self.aa_transition_model.get_Qt_bar(alpha_t_bar, device=ligand_seq.device)
        Qtb = Qtb[repeat_idx]# repeat Qtb for same seq
        prob_X = (Qtb @ ligand_seq.unsqueeze(2)).squeeze()
        # sampling from prob.
        X_t = []
        for prob in prob_X:
            if(prob.sum()!=0):
                X_t.append(prob.multinomial(1)[0])
            else:
                X_t.append(0)
        X_t = torch.Tensor(X_t).long().reshape(batch_size, seq_len)
        noised_data = F.one_hot(torch.Tensor(X_t), num_classes = 20)
        return noised_data.float().to(ligand_seq.device)

    def get_loss(
        self, batch, t_norm, noised_ligand_seq,
    ):
        ligand_mask = batch["ligand_attn_mask"].bool() # only calculate the loss on ligand
        noised_mask =  noised_ligand_seq.argmax(dim=-1) != batch["ligand_seq"].argmax(dim=-1)

        pred_aa = self.forward(
            t_norm,
            noised_ligand_seq, batch["ligand_angles"], batch["ligand_attn_mask"], 
            batch["receptor_seq"], batch["receptor_angles"], batch["receptor_attn_mask"]
        )

        aa_noise_rate = noised_ligand_seq.argmax(dim=-1)[ligand_mask] == batch["ligand_seq"][ligand_mask].argmax(dim=-1)
        aa_noise_rate = aa_noise_rate.sum()/ligand_mask.sum()
        aa_recovery_rate = pred_aa.argmax(dim=-1)[ligand_mask] == batch["ligand_seq"][ligand_mask].argmax(dim=-1)
        aa_recovery_rate = aa_recovery_rate.sum()/ligand_mask.sum()

        aa_noised_loss = self.loss_function(
            pred_aa[noised_mask].view(-1,20),
            batch["ligand_seq"][noised_mask].argmax(dim=-1).view(-1)
        )
        aa_all_loss = self.loss_function(
            pred_aa[ligand_mask&(~noised_mask)].view(-1,20),
            batch["ligand_seq"][ligand_mask&(~noised_mask)].argmax(dim=-1).view(-1)
        )
        elbo = elbo_loss(pred_aa[noised_mask], batch["ligand_seq"][noised_mask])
        # total_loss = (aa_noised_loss*t_norm + aa_all_loss*(1-t_norm)).mean()
        # total_loss = self.loss_function(
        #     pred_aa[ligand_mask].view(-1,20),
        #     batch["ligand_seq"][ligand_mask].argmax(dim=-1).view(-1)
        # )
        total_loss = aa_noised_loss+elbo
        return total_loss, elbo, aa_noised_loss, aa_all_loss, aa_recovery_rate, aa_noise_rate

    def training_step(self, batch, batch_idx):
        t_int = torch.randint(
            0, self.timesteps + 1, 
            size=(batch["ligand_seq"].shape[0], 1),
            device=batch["ligand_seq"].device
        ).float()
        t_norm = t_int/self.timesteps
        aa = self.apply_aa_noise(batch["ligand_seq"], t_int)
        loss, elbo, aa_noised_loss, aa_all_loss, aa_recovery_rate, aa_noise_rate = self.get_loss(
            batch, t_norm, aa
        )
        self.log_dict({
            "aa_noise_rate":aa_noise_rate,
            "aa_recovery_rate":aa_recovery_rate,
            "avg_timestep": t_int.mean().int(),
            "train_loss": loss,
            "train_aa_noised_loss":aa_noised_loss,
            "train_aa_all_loss":aa_all_loss,
            "train_elbo_loss":elbo
        })
        return loss

    def on_train_batch_end(self, outputs, batch_idx, dataloader_idx=0) -> None:
        """Log the average training loss over the epoch"""
        # pl.utilities.rank_zero_info(outputs)
        self.train_epoch_losses.append(float(outputs["loss"]))

    def on_train_epoch_end(self) -> None:
        pl.utilities.rank_zero_info(
            f"Traning Loss:{sum(self.train_epoch_losses)/len(self.train_epoch_losses)}"
        )
        self.train_epoch_losses = []
    
    ### Validation Step OPTs ###
    def validation_step(self, batch, batch_idx):
        # apply noise
        t_int = torch.randint(
            0, self.timesteps + 1, 
            size=(batch["ligand_seq"].shape[0], 1),
            device=batch["ligand_seq"].device
        ).float()
        t_norm = t_int/self.timesteps
        
        aa = self.apply_aa_noise(batch["ligand_seq"], t_int)
        with torch.no_grad():
            loss, elbo, aa_noised_loss, aa_all_loss, aa_recovery_rate, aa_noise_rate = self.get_loss(
                batch, t_norm, aa
            )
        self.log_dict({
            "aa_noise_rate":aa_noise_rate,
            "aa_recovery_rate":aa_recovery_rate,
            "avg_timestep": t_int.mean().int(),
            "val_loss": loss,
            "val_aa_noised_loss":aa_noised_loss,
            "val_aa_all_loss":aa_all_loss,
            "val_elbo":elbo,
        })
        return torch.mean(loss)
    
    def on_validation_epoch_end(self) -> None:
        pl.utilities.rank_zero_info(
            f"Validation Loss:{sum(self.valid_epoch_losses)/len(self.valid_epoch_losses)}"
        )
        self.valid_epoch_losses = []

    def on_validation_batch_end(self, outputs, batch_idx, dataloader_idx=0) -> None:
        """Log the average validation loss over the epoch"""
        self.valid_epoch_losses.append(float(outputs))

    def configure_optimizers(self):
        """
        Return optimizer. Limited support for some optimizers
        """
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr, weight_decay=self.l2_lambda,)
        retval = {"optimizer": optim}
        if self.lr_scheduler:
            if self.lr_scheduler == "OneCycleLR":
                retval["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optim, max_lr=1e-2,epochs=self.max_epochs,
                        steps_per_epoch=self.steps_per_epoch),
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "step",
                }
            elif self.lr_scheduler == "LinearWarmup":
                warmup_steps = int(self.max_epochs * 0.1)
                pl.utilities.rank_zero_info(
                    f"Using linear warmup with {warmup_steps}/{self.max_epochs} warmup steps")
                retval["lr_scheduler"] = {
                    "scheduler": get_linear_schedule_with_warmup(
                        optim,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=self.max_epochs,
                    ),
                    "frequency": 1,
                    "interval": "epoch",  # Call after 1 epoch
                }
            else:
                raise ValueError(f"Unknown lr scheduler {self.lr_scheduler}")
        pl.utilities.rank_zero_info(f"Using optimizer {retval}")
        return retval