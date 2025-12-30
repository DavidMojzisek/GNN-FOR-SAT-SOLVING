import os
import torch
from data.cnf_data import SATDataModule, SATInMemoryDataset, SATOnDiskDataset
from models.wrapper import LitModel
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import logging
import numpy as np
import random
import torch_ema
from pytorch_lightning.callbacks import Callback
import uuid
import os.path as osp


class EMACallback(Callback):
    """Exponential Moving Average callback for stable training."""
    def __init__(self, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.ema = None

    def on_fit_start(self, trainer, pl_module):
        if self.ema is None:
            self.ema = torch_ema.ExponentialMovingAverage(
                pl_module.parameters(),
                decay=self.decay
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.ema.update(pl_module.parameters())

    def on_validation_start(self, trainer, pl_module):
        self.ema.store(pl_module.parameters())
        self.ema.copy_to(pl_module.parameters())

    def on_validation_end(self, trainer, pl_module):
        self.ema.restore(pl_module.parameters())

    def on_test_start(self, trainer, pl_module):
        self.ema.store(pl_module.parameters())
        self.ema.copy_to(pl_module.parameters())

    def on_test_end(self, trainer, pl_module):
        self.ema.restore(pl_module.parameters())

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            return {"ema_state_dict": self.ema.state_dict()}
        return {}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_state_dict" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])


class EMAModelCheckpoint(ModelCheckpoint):
    """Model checkpoint that includes EMA state."""
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        for callback in trainer.callbacks:
            if isinstance(callback, EMACallback) and hasattr(callback, 'ema'):
                checkpoint["ema_state_dict"] = callback.ema.state_dict()
                break


def generate_model_signature(cfg):
    """Generate a descriptive model signature for logging."""
    short_uuid = str(uuid.uuid4())[:6]
    dataset_name = os.path.basename(cfg.data.data_path)

    signature = (
        f"d{cfg.model.d_model}"
        f"_{cfg.model.update_type}"
        f"_i{cfg.model.num_iters}"
        f"_lr{cfg.train.lr}"
        f"_{cfg.train.supervision_mode}"
    )

    if cfg.train.use_ema:
        signature += "_ema"

    if cfg.train.supervision_mode == 'assignment' and cfg.train.train_with_closest_assignment:
        signature += "_closest"

    if cfg.train.supervision_mode != 'sat' and cfg.data.sat_only:
        signature += "_satonly"

    if cfg.train.sat_weight != 1.0:
        signature += f"_w{cfg.train.sat_weight}"

    signature += f"_b{cfg.data.batch_size}_{dataset_name}_{short_uuid}"

    return signature


logging.basicConfig(level=logging.INFO)


def validate_config(cfg: DictConfig):
    """Validate configuration for compatibility and correctness."""
    if cfg.model.update_type == 'primal_dual' and cfg.model.graph_type != 'var':
        raise ValueError(
            f"Primal-dual GNN requires graph_type='var', got '{cfg.model.graph_type}'"
        )

    # Only validate closest_assignment if it's actually enabled AND we're in assignment mode
    if cfg.train.get('train_with_closest_assignment', False):
        valid_modes = ['assignment', 'closest_assignment']
        if cfg.train.supervision_mode not in valid_modes:
            # Don't fail - just warn and disable it
            logging.warning(
                f"train_with_closest_assignment=True only works with supervision_mode in {valid_modes}. "
                f"Got '{cfg.train.supervision_mode}'. Ignoring this flag."
            )

    assert cfg.model.d_model > 0, "d_model must be positive"
    assert cfg.model.num_iters > 0, "num_iters must be positive"
    assert 0 < cfg.train.ema_decay < 1, "ema_decay must be in (0, 1)"
    assert cfg.train.num_local_search_steps > 0, "num_local_search_steps must be positive"

    logging.info("✓ Configuration validated successfully")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if 'gpu_id' in cfg.system and cfg.system.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.system.gpu_id)
        logging.info(f"Using GPU: {cfg.system.gpu_id}")

    validate_config(cfg)

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Generate model signature for checkpoints and logging
    model_signature = generate_model_signature(cfg)
    logging.info(f"Model signature: {model_signature}")

    # Initialize model
    wrapped_model = LitModel(
        model_cfg=cfg.model,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        supervision_mode=cfg.train.supervision_mode,
        train_with_closest_assignment=cfg.train.get('train_with_closest_assignment', False),
        sat_weight=cfg.train.get('sat_weight', 1.0)
    )

    logger = None
    if cfg.logging.get('use_wandb', True):
        logger = WandbLogger(
            project=cfg.logging.get('wandb_project', 'SAT_GNN'),
            name=f"{model_signature}",
            log_model=False
        )
        logging.info(f"WandB logging enabled: project={cfg.logging.wandb_project}")

    # Load datasets
    # Use hydra.utils.to_absolute_path to handle Hydra's working directory changes
    from hydra.utils import to_absolute_path
    data_path = to_absolute_path(cfg.data.data_path)
    train_dir = osp.join(data_path, "train")
    val_dir = osp.join(data_path, "val")
    test_dir = osp.join(data_path, "test")

    dataset_class = SATInMemoryDataset if cfg.data.in_memory else SATOnDiskDataset

    train_dataset = dataset_class(train_dir, cfg.model.d_model, cfg.model.graph_type)[:cfg.data.trainset_size]
    val_dataset = dataset_class(val_dir, cfg.model.d_model, cfg.model.graph_type)[:cfg.data.valset_size]
    test_dataset = dataset_class(test_dir, cfg.model.d_model, cfg.model.graph_type)[:cfg.data.testset_size]

    # Apply SAT-only filtering if configured
    if cfg.data.get('sat_only', False) and cfg.train.supervision_mode != 'sat':
        logging.info("Applying SAT-only filtering to training data")
        original_size = len(train_dataset)
        train_dataset = SATDataModule.filter_sat_only(train_dataset)
        logging.info(f"SAT-only filtering: {len(train_dataset)}/{original_size} instances kept")

    # Create DataModule
    data = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    datamodule = SATDataModule(
        data=data,
        batch_size=cfg.data.batch_size,
        d_model=cfg.model.d_model,
        graph_type=cfg.model.graph_type,
        in_memory=cfg.data.in_memory,
        num_workers=cfg.data.num_workers,
        supervision_mode=cfg.train.supervision_mode
    )

    # Setup checkpoint callback
    checkpoint_dir = os.path.join(cfg.system.base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = EMAModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{model_signature}_{{epoch}}_{{val_accuracy:.4f}}",
        monitor=cfg.checkpointing.get('monitor', 'val_accuracy'),
        mode=cfg.checkpointing.get('mode', 'max'),
        save_top_k=cfg.checkpointing.get('save_top_k', 1),
        save_on_train_epoch_end=False,
        save_last=False,
        verbose=True,
        auto_insert_metric_name=False
    )

    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            check_on_train_epoch_end=False,
            mode='max'
        ),
        checkpoint_callback
    ]

    # Add EMA callback if enabled
    if cfg.train.get('use_ema', False):
        ema_decay = cfg.train.get('ema_decay', 0.9999)
        logging.info(f"Using EMA with decay rate: {ema_decay}")
        callbacks.append(EMACallback(decay=ema_decay))

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        logger=logger,
        accelerator='gpu',
        devices=1,
        gradient_clip_val=cfg.train.gradient_clip_val,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        log_every_n_steps=cfg.logging.get('log_every_n_steps', 50)
    )

    # Train model
    trainer.fit(wrapped_model, datamodule)

    # Validate
    val_results = trainer.validate(wrapped_model, datamodule)
    logging.info(f"Validation results: {val_results}")

    # Test
    test_results = trainer.test(wrapped_model, datamodule)
    logging.info(f"Test results: {test_results}")

    # Save final model checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, f"{model_signature}_final.ckpt")
    trainer.save_checkpoint(final_checkpoint_path)
    logging.info(f"Final model saved to {final_checkpoint_path}")

if __name__ == '__main__':
    main()
