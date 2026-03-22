import math
import logging

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models.model import GNN_SAT
from models.losses import (
    _get_var_votes,
    compute_sat_loss,
    compute_assignment_CE_loss,
    compute_closest_assignment_CE_loss,
    compute_walksat_assignment_CE_loss,
    compute_unsupervised_loss_linear,
    compute_unsupervised_loss_log,
    compute_unsupervised_loss_quad,
)
from models.metrics import compute_metrics, compute_metrics_resampled


class LitModel(pl.LightningModule):
    """
    PyTorch Lightning wrapper for GNN_SAT.

    Supervision modes:
      assignment          — CE loss against fixed solver assignments
      closest_assignment  — CE loss against best nearby assignment (see closest_assignment_method)
      sat                 — BCE loss for SAT/UNSAT classification only
      unsupervised_linear — L_lin = -∑_c V_c
      unsupervised_log    — L_log = -∑_c log(V_c)
      unsupervised_quad   — L_quad = ∑_c (1-V_c)²

    Closest assignment methods (only used when supervision_mode='closest_assignment'):
      rc2     — MaxSAT exact solver (slowest, optimal)
      walksat — WalkSAT local search (fast, online, approximate)
      greedy  — Greedy max-gain heuristic (fast, online, approximate)
    """

    def __init__(self, model_cfg, lr, weight_decay,
                 supervision_mode='assignment',
                 closest_assignment_method='rc2',
                 num_local_search_steps=10,
                 sat_weight=1.0,
                 num_test_samples=1,
                 lr_schedule='cosine',
                 fixed_assignment_warmup_epochs=0,
                 # Legacy flag — deprecated, use supervision_mode='closest_assignment'
                 train_with_closest_assignment=False):
        super().__init__()
        self.save_hyperparameters()

        self.model = GNN_SAT(
            d_model=model_cfg.d_model,
            update_type=model_cfg.update_type,
            graph_type=model_cfg.graph_type,
            use_edge_features=model_cfg.get('use_edge_features', False),
            use_polarity_scalar=model_cfg.get('use_polarity_scalar', False),
            use_clause_voting=model_cfg.get('use_clause_voting', False),
            separate_direction_mlps=model_cfg.get('separate_direction_mlps', False),
            collect_all_votes=model_cfg.get('collect_all_votes', False),
            normalize_embeddings=model_cfg.get('normalize_embeddings', None),
            output_bias=model_cfg.get('output_bias', True),
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_iters = model_cfg.num_iters
        self._train_loss_accum = []

        # Resolve supervision mode — legacy flag takes lowest priority
        if supervision_mode == 'assignment' and train_with_closest_assignment:
            logging.warning(
                "train_with_closest_assignment=True is deprecated. "
                "Use supervision_mode='closest_assignment' instead. "
                "Treating as closest_assignment with method=rc2."
            )
            supervision_mode = 'closest_assignment'
            closest_assignment_method = 'rc2'

        self.supervision_mode = supervision_mode
        self.closest_assignment_method = closest_assignment_method
        self.num_local_search_steps = num_local_search_steps
        self.sat_weight = sat_weight
        self.num_test_samples = num_test_samples
        self.lr_schedule = lr_schedule
        self.fixed_assignment_warmup_epochs = fixed_assignment_warmup_epochs

    def _get_loss(self, outputs, batch):
        mode = self.supervision_mode

        if mode == 'assignment':
            if self.sat_weight != 1.0:
                return self._weighted_assignment_loss(outputs, batch)
            return compute_assignment_CE_loss(outputs, batch, self.model.graph_type)

        if mode == 'closest_assignment':
            method = self.closest_assignment_method
            if method == 'rc2':
                return compute_closest_assignment_CE_loss(outputs, batch, self.model.graph_type)
            elif method in ('walksat', 'greedy'):
                return compute_walksat_assignment_CE_loss(
                    outputs, batch, self.model.graph_type,
                    num_steps=self.num_local_search_steps, method=method
                )
            else:
                raise ValueError(f"Unknown closest_assignment_method: '{method}'. Choose rc2/walksat/greedy")

        if mode == 'sat':
            return compute_sat_loss(outputs, batch)
        if mode == 'unsupervised_linear':
            return compute_unsupervised_loss_linear(outputs, batch, self.model.graph_type)
        if mode == 'unsupervised_log':
            return compute_unsupervised_loss_log(outputs, batch, self.model.graph_type)
        if mode == 'unsupervised_quad':
            return compute_unsupervised_loss_quad(outputs, batch, self.model.graph_type)

        raise ValueError(f"Unknown supervision_mode: '{mode}'")

    def _weighted_assignment_loss(self, outputs, batch):
        """Assignment CE loss with higher weight for SAT instances (vectorised)."""
        votes = _get_var_votes(outputs, batch, self.model.graph_type)
        targets = ((batch.assignment + 1) / 2).long().to(votes.device)

        # Build per-formula weight vector then expand to per-variable weights
        weight_per_formula = torch.where(
            batch.y == 1,
            torch.full_like(batch.y, self.sat_weight),
            torch.ones_like(batch.y),
        )
        weights = torch.repeat_interleave(weight_per_formula, batch.num_variables.long())

        # Normalise to preserve gradient magnitude
        weights = weights * len(weights) / weights.sum()
        loss = F.cross_entropy(votes, targets, reduction='none')
        return (loss * weights).mean()

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch, self.num_iters)
        if (self.fixed_assignment_warmup_epochs > 0
                and self.trainer.current_epoch < self.fixed_assignment_warmup_epochs):
            loss = compute_assignment_CE_loss(outputs, batch, self.model.graph_type)
        else:
            loss = self._get_loss(outputs, batch)
        self._train_loss_accum.append(loss.detach())
        self.log('lr', self.optimizers().param_groups[0]['lr'],
                 prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        # Log train_loss here so it shares the same wandb step as val metrics.
        # If logged in training_step, PL commits it at end of train epoch (different step).
        if self._train_loss_accum:
            avg = torch.stack(self._train_loss_accum).mean()
            self.log('train_loss', avg, prog_bar=True)
            self._train_loss_accum = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # dataloader_idx=0: full val set  |  dataloader_idx=1: curriculum-filtered val set
        prefix = 'val' if dataloader_idx == 0 else 'val_curr'
        outputs = self.model(batch, self.num_iters)
        loss = self._get_loss(outputs, batch)
        metrics = compute_metrics(outputs, batch, self.supervision_mode, self.model.graph_type)
        self.log(f'{prefix}_loss', loss, prog_bar=(prefix == 'val'), add_dataloader_idx=False)
        for name, value in metrics.items():
            prog = (prefix == 'val') and (name in ('dec_acc', 'sat_acc', 'avg_gap'))
            self.log(f'{prefix}_{name}', value, prog_bar=prog, add_dataloader_idx=False)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.model(batch, self.num_iters)
        loss = self._get_loss(outputs, batch)

        if self.num_test_samples > 1:
            metrics = compute_metrics_resampled(
                self.model, batch, self.num_iters,
                self.supervision_mode, self.model.graph_type,
                n_samples=self.num_test_samples
            )
        else:
            metrics = compute_metrics(outputs, batch, self.supervision_mode, self.model.graph_type)

        self.log('test_loss', loss, prog_bar=True)
        for name, value in metrics.items():
            self.log(f'test_{name}', value, prog_bar=(name in ('dec_acc', 'sat_acc')))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.lr_schedule == 'constant':
            return optimizer

        min_lr = self.lr * 0.1

        def lr_lambda(epoch):
            half = max(self.trainer.max_epochs // 2, 1)
            if epoch < half:
                cosine = 0.5 * (1 + math.cos(math.pi * epoch / half))
                return min_lr / self.lr + (1 - min_lr / self.lr) * cosine
            return min_lr / self.lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
