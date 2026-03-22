import math
import os
import os.path as osp
import logging
import numpy as np
import random
import uuid

import torch
import torch_ema
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
import hydra
from omegaconf import DictConfig

from data.cnf_data import SATDataModule, SATInMemoryDataset, SATOnDiskDataset
from models.wrapper import LitModel


# ---------------------------------------------------------------------------
# EMA callbacks
# ---------------------------------------------------------------------------

class EMACallback(Callback):
    """Exponential Moving Average of model parameters for stable training."""

    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema = None
        self._pending_ema_state = None  # holds loaded state until on_fit_start initialises ema

    def on_fit_start(self, trainer, pl_module):
        if self.ema is None:
            self.ema = torch_ema.ExponentialMovingAverage(
                pl_module.parameters(), decay=self.decay
            )
        # Apply any EMA state that was loaded from a checkpoint before on_fit_start ran
        if self._pending_ema_state is not None:
            self.ema.load_state_dict(self._pending_ema_state)
            self._pending_ema_state = None

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
            checkpoint['ema_state_dict'] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if 'ema_state_dict' in checkpoint:
            if self.ema is not None:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
            else:
                # on_fit_start hasn't run yet; defer until it does
                self._pending_ema_state = checkpoint['ema_state_dict']


class BestCheckpoint(ModelCheckpoint):
    """Checkpoint that preserves EMA state alongside model weights.

    EMA state is written to the checkpoint dict by EMACallback.on_save_checkpoint,
    which is called for every checkpoint save. No duplication needed here.
    """
    pass


# ---------------------------------------------------------------------------
# Curriculum learning
# ---------------------------------------------------------------------------

class CurriculumCallback(Callback):
    """
    Curriculum scheduler for SAT/UNSAT supervision.

    - Filters training AND validation data to a sliding window:
        [max(min_vars, current_max - window_stages*step), current_max]
      (prevents catastrophic forgetting of recent stages while dropping very small formulas)
    - At max_vars stage the window is expanded to the full [min_vars, max_vars] range to
      maximise training data and prevent overfitting on the small window subset.
    - Advancement threshold is linearly scheduled from acc_threshold_min (at min_vars)
      to acc_threshold_max (at max_vars).
    - Advancement decision uses val_curr_dec_acc (filtered val set).
    - Full val metrics (val_dec_acc) are always also logged.
    - post_curriculum_cosine_epochs: if set, starts a CosineAnnealingLR once max_vars is
      reached, decaying from current LR to 5% over that many epochs.

    Requires: datamodule with set_curriculum_max_vars(max_vars, min_vars_window) method.
    """

    def __init__(self, min_vars, max_vars, step,
                 acc_threshold_min, acc_threshold_max,
                 patience, window_stages=4,
                 post_curriculum_cosine_epochs=None):
        super().__init__()
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.step = step
        self.acc_threshold_min = acc_threshold_min
        self.acc_threshold_max = acc_threshold_max
        self.patience = patience
        self.window_stages = window_stages
        self.post_curriculum_cosine_epochs = post_curriculum_cosine_epochs

        self.current_max = min_vars
        self._epochs_at_stage = 0
        self._post_curriculum_scheduler = None

    def _scheduled_threshold(self):
        """Linear interpolation of advancement threshold based on current stage."""
        if self.max_vars <= self.min_vars:
            return self.acc_threshold_min
        t = (self.current_max - self.min_vars) / (self.max_vars - self.min_vars)
        return self.acc_threshold_min + t * (self.acc_threshold_max - self.acc_threshold_min)

    def _current_min_window(self):
        """Lower bound of the sliding window.
        At max stage expands to min_vars so the full dataset is used (prevents overfitting
        on the small window subset that would otherwise be ~24% of data).
        """
        if self.current_max >= self.max_vars:
            return self.min_vars
        return max(self.min_vars, self.current_max - self.window_stages * self.step)

    def _update_datamodule(self, trainer):
        trainer.datamodule.set_curriculum_max_vars(
            self.current_max, self._current_min_window()
        )

    def on_fit_start(self, trainer, pl_module):
        self._update_datamodule(trainer)
        logging.info(
            f"Curriculum: starting — max_vars={self.current_max}, "
            f"window=[{self._current_min_window()}, {self.current_max}], "
            f"threshold={self._scheduled_threshold():.3f}"
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        dec_acc = float(metrics.get('val_curr_dec_acc', metrics.get('val_dec_acc', 0.0)))
        threshold = self._scheduled_threshold()

        # Always increment so epochs_at_stage is a useful counter at every stage
        self._epochs_at_stage += 1

        # Log training set size (after curriculum filter)
        train_size = len(trainer.datamodule._get_curriculum_dataset())
        pl_module.log('curriculum/train_size', float(train_size))

        pl_module.log('curriculum/max_vars', float(self.current_max))
        pl_module.log('curriculum/window_min', float(self._current_min_window()))
        pl_module.log('curriculum/threshold', threshold)
        pl_module.log('curriculum/dec_acc_curr', dec_acc)
        pl_module.log('curriculum/epochs_at_stage', float(self._epochs_at_stage))

        if self.current_max >= self.max_vars:
            # Start cosine LR decay on first epoch at max stage
            if (self._post_curriculum_scheduler is None
                    and self.post_curriculum_cosine_epochs is not None):
                optimizer = pl_module.optimizers()
                if isinstance(optimizer, list):
                    optimizer = optimizer[0]
                current_lr = optimizer.param_groups[0]['lr']
                self._post_curriculum_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.post_curriculum_cosine_epochs,
                    eta_min=current_lr * 0.05,
                )
                logging.info(
                    f"Curriculum complete — starting cosine LR decay from {current_lr:.2e} "
                    f"over {self.post_curriculum_cosine_epochs} epochs"
                )
            elif self._post_curriculum_scheduler is not None:
                self._post_curriculum_scheduler.step()
            return

        advance = (dec_acc >= threshold) or (self._epochs_at_stage >= self.patience)
        if advance:
            prev_max = self.current_max
            self.current_max = min(self.current_max + self.step, self.max_vars)
            self._epochs_at_stage = 0
            self._update_datamodule(trainer)
            logging.info(
                f"Curriculum: {prev_max} → {self.current_max} "
                f"(dec_acc={dec_acc:.3f} vs threshold={threshold:.3f}), "
                f"new window=[{self._current_min_window()}, {self.current_max}], "
                f"new_threshold={self._scheduled_threshold():.3f}"
            )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def generate_model_signature(cfg):
    """Human-readable experiment identifier for checkpoints and WandB."""
    short_uuid = str(uuid.uuid4())[:6]
    dataset_name = os.path.basename(cfg.data.data_path)
    sig = (
        f"{cfg.model.graph_type}"
        f"_{cfg.model.update_type}"
        f"_d{cfg.model.d_model}"
        f"_i{cfg.model.num_iters}"
        f"_{cfg.train.supervision_mode}"
    )
    if cfg.train.supervision_mode == 'closest_assignment':
        sig += f"_{cfg.train.closest_assignment_method}"
    if cfg.train.get('use_ema', True):
        sig += "_ema"
    if cfg.data.get('sat_only', False):
        sig += "_satonly"
    if cfg.train.get('sat_weight', 1.0) != 1.0:
        sig += f"_w{cfg.train.sat_weight}"
    sig += f"_b{cfg.data.batch_size}_{dataset_name}_{short_uuid}"
    return sig


def validate_config(cfg: DictConfig):
    """Validate configuration for compatibility. Raises on hard conflicts, warns on soft ones."""
    if cfg.model.update_type == 'primal_dual' and cfg.model.graph_type != 'var':
        raise ValueError(
            f"update_type='primal_dual' requires graph_type='var', got '{cfg.model.graph_type}'"
        )

    if cfg.model.get('use_edge_features', False) is False and cfg.model.get('separate_direction_mlps', False):
        logging.warning("separate_direction_mlps=True has no effect when use_edge_features=False")

    curriculum_cfg = cfg.train.get('curriculum', None)
    if curriculum_cfg and curriculum_cfg.get('enabled', False):
        if cfg.train.supervision_mode != 'sat':
            logging.warning(
                "curriculum is only meaningful with supervision_mode='sat'; "
                "curriculum callbacks will not be added for other modes."
            )

    assert cfg.model.d_model > 0, "d_model must be positive"
    assert cfg.model.num_iters > 0, "num_iters must be positive"
    assert 0 < cfg.train.get('ema_decay', 0.999) < 1, "ema_decay must be in (0, 1)"
    assert cfg.train.get('num_local_search_steps', 10) > 0, "num_local_search_steps must be positive"

    curriculum_cfg = cfg.train.get('curriculum', None)
    if curriculum_cfg and curriculum_cfg.get('enabled', False):
        assert curriculum_cfg.get('acc_threshold_min', 0.65) < curriculum_cfg.get('acc_threshold_max', 0.85), \
            "acc_threshold_min must be less than acc_threshold_max"

    logging.info("Configuration validated.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    from hydra.utils import to_absolute_path
    logging.basicConfig(level=logging.INFO)

    if cfg.system.get('gpu_id') is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.system.gpu_id)

    validate_config(cfg)

    seed = cfg.system.get('seed', 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_signature = generate_model_signature(cfg)
    logging.info(f"Run: {model_signature}")

    # Build model
    wrapped_model = LitModel(
        model_cfg=cfg.model,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        supervision_mode=cfg.train.supervision_mode,
        closest_assignment_method=cfg.train.get('closest_assignment_method', 'rc2'),
        num_local_search_steps=cfg.train.get('num_local_search_steps', 10),
        sat_weight=cfg.train.get('sat_weight', 1.0),
        train_with_closest_assignment=cfg.train.get('train_with_closest_assignment', False),
        num_test_samples=cfg.train.get('num_test_samples', 1),
        lr_schedule=cfg.train.get('lr_schedule', 'cosine'),
        fixed_assignment_warmup_epochs=cfg.train.get('fixed_assignment_warmup_epochs', 0),
    )

    # Logger
    logger = None
    if cfg.logging.get('use_wandb', False):
        logger = WandbLogger(
            project=cfg.logging.get('wandb_project', 'SAT_GNN'),
            name=model_signature,
            log_model=False,
        )

    # Datasets
    data_path = to_absolute_path(cfg.data.data_path)
    dataset_class = SATInMemoryDataset if cfg.data.get('in_memory', True) else SATOnDiskDataset

    train_dataset = dataset_class(osp.join(data_path, 'train'), cfg.model.graph_type)
    val_dataset   = dataset_class(osp.join(data_path, 'val'),   cfg.model.graph_type)
    test_dataset  = dataset_class(osp.join(data_path, 'test'),  cfg.model.graph_type)

    train_dataset = train_dataset[:cfg.data.trainset_size]
    val_dataset   = val_dataset[:cfg.data.valset_size]
    test_dataset  = test_dataset[:cfg.data.testset_size]

    if cfg.data.get('sat_only', False) and cfg.train.supervision_mode != 'sat':
        logging.info("Applying SAT-only filter to training data")
        orig = len(train_dataset)
        train_dataset = SATDataModule.filter_sat_only(train_dataset)
        logging.info(f"SAT-only filter: {len(train_dataset)}/{orig} kept")

    # Load mixed-size curriculum val dataset when curriculum is active.
    # This is separate from val_dataset (which is max-size only, fixed benchmark).
    _curriculum_cfg_early = cfg.train.get('curriculum', None)
    curr_val_dataset = None
    if _curriculum_cfg_early and _curriculum_cfg_early.get('enabled', False) and cfg.train.supervision_mode == 'sat':
        curr_val_path = cfg.data.get('curriculum_val_path', None)
        if curr_val_path:
            curr_val_path = to_absolute_path(curr_val_path)
            logging.info(f"Loading curriculum val dataset from {curr_val_path}")
            curr_val_raw = dataset_class(curr_val_path, cfg.model.graph_type)
            curr_val_size = cfg.data.get('curriculum_valset_size', len(curr_val_raw))
            curr_val_dataset = curr_val_raw[:curr_val_size]
            logging.info(f"Curriculum val dataset: {len(curr_val_dataset)} instances")
        else:
            logging.warning(
                "Curriculum is enabled but data.curriculum_val_path is not set. "
                "val_curr_* metrics will reflect max-size val, not current curriculum stage. "
                "Generate a mixed-size val set and set data.curriculum_val_path."
            )

    datamodule = SATDataModule(
        data={'train': train_dataset, 'val': val_dataset, 'test': test_dataset},
        batch_size=cfg.data.batch_size,
        graph_type=cfg.model.graph_type,
        in_memory=cfg.data.get('in_memory', True),
        num_workers=cfg.data.num_workers,
        supervision_mode=cfg.train.supervision_mode,
        curr_val_dataset=curr_val_dataset,
    )

    # Checkpoints — save best and last only
    checkpoint_dir = osp.join(cfg.system.base_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    monitor_metric = cfg.checkpointing.get('monitor', 'val_dec_acc')

    best_ckpt = BestCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{model_signature}_best",
        monitor=monitor_metric,
        mode=cfg.checkpointing.get('mode', 'max'),
        save_top_k=1,
        save_last=False,
        verbose=True,
        auto_insert_metric_name=False,
    )
    last_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{model_signature}_last",
        save_top_k=0,
        save_last=True,
        auto_insert_metric_name=False,
    )

    # Callbacks
    early_stopping_patience = cfg.train.get('early_stopping_patience', 20)
    callbacks = [
        EarlyStopping(monitor=monitor_metric, patience=early_stopping_patience, mode='max',
                      check_on_train_epoch_end=False),
        best_ckpt,
        last_ckpt,
    ]

    if cfg.train.get('use_ema', True):
        callbacks.append(EMACallback(decay=cfg.train.get('ema_decay', 0.999)))

    curriculum_cfg = cfg.train.get('curriculum', None)
    if curriculum_cfg and curriculum_cfg.get('enabled', False) and cfg.train.supervision_mode == 'sat':
        callbacks.append(CurriculumCallback(
            min_vars=curriculum_cfg.min_vars,
            max_vars=curriculum_cfg.max_vars,
            step=curriculum_cfg.step,
            acc_threshold_min=curriculum_cfg.get('acc_threshold_min', 0.65),
            acc_threshold_max=curriculum_cfg.get('acc_threshold_max', 0.85),
            patience=curriculum_cfg.patience,
            window_stages=curriculum_cfg.get('window_stages', 4),
            post_curriculum_cosine_epochs=curriculum_cfg.get('post_curriculum_cosine_epochs', None),
        ))

    curriculum_active = (
        curriculum_cfg is not None
        and curriculum_cfg.get('enabled', False)
        and cfg.train.supervision_mode == 'sat'
    )

    if curriculum_active:
        num_stages = math.ceil((curriculum_cfg.max_vars - curriculum_cfg.min_vars) / curriculum_cfg.step) + 1
        max_epochs = curriculum_cfg.patience * num_stages * 2  # 2x safety buffer
        logging.info(
            f"Curriculum active: {num_stages} stages × {curriculum_cfg.patience} patience × 2 buffer "
            f"= {max_epochs} max_epochs (config num_epochs={cfg.train.num_epochs} ignored)"
        )
    else:
        max_epochs = cfg.train.num_epochs

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        accelerator='gpu',
        devices=1,
        gradient_clip_val=cfg.train.gradient_clip_val,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        log_every_n_steps=cfg.logging.get('log_every_n_steps', 50),
        # Reload dataloaders each epoch when curriculum is active so the filtered
        # val loader reflects the current curriculum window.
        reload_dataloaders_every_n_epochs=1 if curriculum_active else 0,
    )

    trainer.fit(wrapped_model, datamodule)
    trainer.validate(wrapped_model, datamodule)
    trainer.test(wrapped_model, datamodule)


if __name__ == '__main__':
    main()
