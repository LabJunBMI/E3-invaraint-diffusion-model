import torch
import pytorch_lightning as pl
from transformers import BertConfig
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from model import PeptideDiff
from dataset import LigandBindingSiteDataset

MODEL_PATH = "" # The file path of trained model
DATA_FILE = "./data/biolip.pt"
GPU_ID=[4]
NUM_THREAD = 16

# Config this accordingly
# The ext 1 model is traind with "max_seq_len" of 64, others are trained with 128
CONFIG = {
    "pocket_ext":4,
    "timesteps": 50,
    "max_seq_len": 128,
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
    train_angle_ds = LigandBindingSiteDataset(
        file_path,
        "train", 
        CONFIG["max_seq_len"],
        CONFIG["pocket_ext"]
    )
    train_dataloader = DataLoader(
        dataset=train_angle_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,  # Shuffle only train loader
        num_workers=NUM_THREAD,
    )
    val_angle_ds = LigandBindingSiteDataset(
        file_path,
        "validation", 
        CONFIG["max_seq_len"],
        CONFIG["pocket_ext"]
    )
    val_dataloader = DataLoader(
        dataset=val_angle_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,  # Shuffle only train loader
        num_workers=NUM_THREAD,
    )
    return train_dataloader, val_dataloader

def train_model(encoder_config:BertConfig, decoder_config:BertConfig, train_dataloader: DataLoader, val_dataloader: DataLoader):
    model = PeptideDiff(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        feature_names= train_dataloader.dataset.feature_names,
        max_epochs=CONFIG["max_epochs"],
        lr_scheduler=CONFIG["lr_scheduler"],
        l2_lambda=CONFIG["l2_norm"],
        steps_per_epoch=len(train_dataloader),
        learning_rate=CONFIG["lr"],
        loss_func=torch.nn.CrossEntropyLoss(),
        noise_schedule=CONFIG["noise_schedule"],
        timesteps=CONFIG["timesteps"],
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./",
        monitor='val_loss',    # Monitor the validation loss
        filename='best_val_model', # Filename template
        save_top_k=1,          # Only save the best model
        mode='max'             # Save the model with the highest validation loss
    )
    
    print(f"Model has {num_params} trainable parameters")
    trainer = pl.Trainer(
        default_root_dir="./",
        gradient_clip_val=CONFIG["gradient_clip"],
        callbacks=[checkpoint_callback],
        min_epochs=CONFIG["min_epochs"],
        max_epochs=CONFIG["max_epochs"],
        check_val_every_n_epoch=1,
        log_every_n_steps=30,
        accelerator="gpu",
        devices=GPU_ID
        # move_metrics_to_cpu=False,  # Saves memory
    )
    print("Start training")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return trainer, model

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium") # increase performance, the loss of precision has a negligible impact. 
    torch.set_num_threads(NUM_THREAD)
    print("Loading Data")
    train_dataloader, val_dataloader = get_dataloader(DATA_FILE)
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
    
    trainer, model = train_model(encoder_config, decoder_config, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), MODEL_PATH)

