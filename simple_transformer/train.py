import os
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformer import Transformer
from data_process import load_dataset


class TranslationDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, num_workers: int = 0):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = load_dataset(self.data_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )


class TransformerLightning(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = Transformer(
            src_vocab_size=self.hparams.src_vocab_size,
            tgt_vocab_size=self.hparams.tgt_vocab_size,
            d_model=self.hparams.d_model,
            num_heads=self.hparams.num_heads,
            num_layers=self.hparams.num_layers,
            d_ff=self.hparams.d_ff,
            dropout=self.hparams.dropout,
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.hparams.pad_idx)

    def forward(self, src, tgt):
        # return logits: (batch, seq_len, vocab)
        return self.model(src, tgt, self.hparams.pad_idx, self.hparams.pad_idx)

    def _step(self, batch, stage: str):
        src, tgt = batch['input_ids'], batch['labels']
        logits = self(src, tgt[:, :-1])
        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1)
        )
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._step(batch, 'val')

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = StepLR(
            optimizer,
            step_size=self.hparams.step_size,
            gamma=self.hparams.gamma
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }


def main(config: dict):
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)

    # Logger and callbacks
    logger = TensorBoardLogger(
        save_dir=config['save_dir'],
        name='lightning_logs'
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(config['save_dir'], 'checkpoints'),
        filename='best_model_{epoch:02d}-{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # DataModule and model
    data_module = TranslationDataModule(
        data_path=config['data_path'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4)
    )
    model = TransformerLightning(config)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator='auto',
        devices=config.get('gpus', 1),
        precision=32,  # use float16 to enable mixed precision
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        logger=logger,
        gradient_clip_val=config.get('grad_clip', 1.0),
        log_every_n_steps=50,
    )

    # Train
    trainer.fit(model, data_module)

    # Save final model
    ckpt_path = checkpoint_cb.best_model_path
    print(f"Best model saved at {ckpt_path}")


if __name__ == '__main__':
    config = {
        'src_vocab_size': 32000,
        'tgt_vocab_size': 32000,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'pad_idx': 0,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'step_size': 1,
        'gamma': 0.95,
        'batch_size': 8, # 32
        'epochs': 10,
        'save_dir': 'checkpoints',
        'data_path': 'data/translation2019zh_train.json',
        'num_workers': 8,
        'grad_clip': 1.0,
    }
    main(config)
