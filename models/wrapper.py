import pytorch_lightning as pl
import torch
import numpy as np
from models.model import GNN_SAT
from models.losses import (compute_sat_loss, compute_assignment_CE_loss,
                           compute_unsupervised_loss_linear, compute_unsupervised_loss_log, compute_unsupervised_loss_quad,
                           compute_unsupervised_loss_1, compute_unsupervised_loss_2,
                           compute_closest_assignment_CE_loss)
from models.metrics import compute_metrics


class LitModel(pl.LightningModule):
   def __init__(self, model_cfg, lr, weight_decay, supervision_mode='sat',
                train_with_closest_assignment=False, sat_weight=1.0):
       """
       Initialize the primal-dual GNN model.

       Args:
           model_cfg: Model configuration
           lr: Learning rate
           weight_decay: Weight decay
           supervision_mode: Type of supervision ('sat', 'assignment', 'unsupervised_linear', 'unsupervised_log', 'unsupervised_quad', 'combined')
                             Also supports legacy names: 'unsupervised1' (linear), 'unsupervised2' (log)
           train_with_closest_assignment: Whether to train with closest assignments
           sat_weight: Weight for SAT instances (UNSAT instances have weight 1.0)
       """
       super().__init__()
       self.save_hyperparameters(ignore=['model'])

       # Only GNN_SAT for primal-dual
       self.model = GNN_SAT(
            d_model=model_cfg.d_model,
            update_type=model_cfg.update_type,
            graph_type=model_cfg.graph_type,
            collect_embeddings=model_cfg.collect_embeddings,
            use_edge_features=model_cfg.get('use_edge_features', False),
            use_clause_voting=model_cfg.get('use_clause_voting', False),
            assignment_CE_loss=True  # Always use CE loss for assignment
        )
       self.lr = lr
       self.weight_decay = weight_decay
       self.num_iters = model_cfg.num_iters

       self.supervision_mode = supervision_mode

       # Add parameter for closest assignment training (only used with assignment supervision)
       self.train_with_closest_assignment = train_with_closest_assignment

       # Add parameter for SAT weighting
       self.sat_weight = sat_weight
       
   def _get_loss(self, outputs, batch):
       # Only use closest assignment training if supervision mode is assignment
       if self.supervision_mode == 'assignment':
           if self.train_with_closest_assignment:
               return compute_closest_assignment_CE_loss(outputs, batch, self.model.graph_type)
           elif self.sat_weight != 1.0:
               # Apply SAT weighting to standard assignment loss
               device = batch.y.device
               
               # Get model predictions
               if self.model.graph_type == 'lit':    
                   # For lit graph, extract just the positive literals
                   grouped_votes = [outputs['final_votes'][batch.x_l_batch==i] for i in range(batch.x_l_batch.max()+1)]
                   processed_votes = []
                   for votes in grouped_votes:
                       n_vars = votes.size(0) // 2
                       var_votes = votes[:n_vars]
                       processed_votes.append(var_votes)
                   votes = torch.cat(processed_votes)
               else:
                   # For var graph, use all predictions
                   votes = outputs['final_votes']
               
               # Get targets from batch assignments
               targets = ((batch.assignment + 1) / 2).long().to(device)
               
               # Apply SAT instance weighting
               var_cumsum = torch.cumsum(batch.num_variables, dim=0)
               sample_weights = torch.ones_like(targets, dtype=torch.float)
               
               # Assign weight based on whether the formula is SAT or UNSAT
               for i in range(len(batch.y)):
                   start_idx = 0 if i == 0 else var_cumsum[i-1].item()
                   end_idx = var_cumsum[i].item()
                   
                   # Higher weight for SAT instances
                   is_sat = batch.y[i].item() == 1
                   formula_weight = self.sat_weight if is_sat else 1.0
                   sample_weights[start_idx:end_idx] = formula_weight
               
               # Normalize weights to maintain gradient magnitude
               if sample_weights.sum() > 0:
                   sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()
               
               # Compute weighted cross entropy loss
               criterion = torch.nn.CrossEntropyLoss(reduction='none')
               per_sample_loss = criterion(votes, targets)
               weighted_loss = (per_sample_loss * sample_weights).mean()
               
               return weighted_loss
           else:
               # Standard assignment loss without weighting
               return compute_assignment_CE_loss(outputs, batch, self.model.graph_type)
           
       # For other supervision modes, use the appropriate loss function
       if self.supervision_mode == 'sat':
           return compute_sat_loss(outputs, batch)
       elif self.supervision_mode in ["unsupervised_linear", "unsupervised1"]:
           return compute_unsupervised_loss_linear(outputs, batch, self.model.graph_type)
       elif self.supervision_mode in ["unsupervised_log", "unsupervised2"]:
           return compute_unsupervised_loss_log(outputs, batch, self.model.graph_type)
       elif self.supervision_mode == "unsupervised_quad":
           return compute_unsupervised_loss_quad(outputs, batch, self.model.graph_type)
       elif self.supervision_mode == "combined":
            assignment_loss = compute_assignment_CE_loss(outputs, batch, self.model.graph_type)
            unsup_loss = compute_unsupervised_loss_linear(outputs, batch, self.model.graph_type)
            return assignment_loss + unsup_loss
       else:
           raise ValueError(f"Unknown supervision mode: {self.supervision_mode}")
        
   def training_step(self, batch, batch_idx):
       self.log('learning_rate', self.optimizers().param_groups[0]['lr'], prog_bar=True, on_step=False, on_epoch=True)

       # Only print information about the current loss at the beginning of each epoch
       if batch_idx == 0:
           if self.train_with_closest_assignment and self.supervision_mode == 'assignment':
               print(f"\nEpoch {self.trainer.current_epoch}: Using closest assignment loss")
           elif self.sat_weight != 1.0 and self.supervision_mode == 'assignment':
               print(f"\nEpoch {self.trainer.current_epoch}: Using assignment loss with SAT weight={self.sat_weight}")
           else:
               print(f"\nEpoch {self.trainer.current_epoch}: Using {self.supervision_mode} loss")

       # Run the model forward pass
       outputs = self.model(batch, self.num_iters)

       # Get loss using the appropriate method
       loss = self._get_loss(outputs, batch)
       self.log('train_loss', loss, prog_bar=True)

       return loss

   def validation_step(self, batch, batch_idx):
       """
       Validation step.

       Args:
           batch: The input batch
           batch_idx: Index of the batch

       Returns:
           Validation loss
       """
       outputs = self.model(batch, self.num_iters)
       loss = self._get_loss(outputs, batch)

       # Standard metrics calculation based on supervision mode
       metrics = compute_metrics(outputs, batch, self.supervision_mode, self.model.graph_type)

       # Log all metrics
       self.log('val_loss', loss, prog_bar=True)
       for name, value in metrics.items():
           self.log(f'val_{name}', value, prog_bar=True)
       return loss

   def test_step(self, batch, batch_idx):
       """
       Test step - identical to validation step but logs with test_ prefix.
       """
       outputs = self.model(batch, self.num_iters)
       loss = self._get_loss(outputs, batch)

       # Standard metrics calculation based on supervision mode
       metrics = compute_metrics(outputs, batch, self.supervision_mode, self.model.graph_type)

       # Log all metrics with test_ prefix
       self.log('test_loss', loss, prog_bar=True)
       for name, value in metrics.items():
           self.log(f'test_{name}', value, prog_bar=True)
       return loss

   def configure_optimizers(self):
       optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
       min_lr = self.lr * 0.1  # Set explicit minimum LR 1e-5
    
       # Use a custom lambda function to define our schedule
       def lr_lambda(epoch):
           if epoch < self.trainer.max_epochs // 2:
               # Cosine annealing for first half of training
               cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(epoch / (self.trainer.max_epochs // 2) * torch.pi))).item()
               return min_lr/self.lr + (1 - min_lr/self.lr) * cosine_factor
           else:
               # Constant minimum LR for second half
               return min_lr/self.lr
    
       scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
       return {
           "optimizer": optimizer,
           "lr_scheduler": {
               "scheduler": scheduler,
               "interval": "epoch",
               "frequency": 1,
               "monitor": None,
               "strict": True
           }
       }
