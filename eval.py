"""
Evaluate a saved checkpoint on one or more test datasets.

Model settings (architecture, supervision mode, graph type, etc.) are restored
automatically from the checkpoint — you do NOT need to match the config to how
the model was trained. Only the data path and data loading settings come from config.

Usage:
    # Single dataset
    python eval.py checkpoint=checkpoints/my_model_best.ckpt

    # Multiple datasets (bar chart of avg gap per dataset)
    python eval.py checkpoint=checkpoints/my_model_best.ckpt \\
        "eval.datasets=[./data/cnfs/sr40,./data/cnfs/sr100,./data/cnfs/3sat100]"

    # Test-time scaling analysis (gap vs message-passing iterations)
    python eval.py checkpoint=checkpoints/my_model_best.ckpt \\
        eval.scaling_analysis=True eval.max_test_iters=100

    # All in one
    python eval.py checkpoint=checkpoints/my_model_best.ckpt \\
        "eval.datasets=[./data/cnfs/sr40,./data/cnfs/sr100]" \\
        eval.scaling_analysis=True eval.max_test_iters=100
"""

import os
import os.path as osp
import logging

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from data.cnf_data import SATInMemoryDataset, SATOnDiskDataset, SATDataModule
from models.wrapper import LitModel
from models.metrics import compute_gap_per_iteration


def _restore_ema(model, checkpoint_path):
    """Copy EMA weights into the model if the checkpoint contains them."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'ema_state_dict' not in checkpoint:
        return False
    try:
        import torch_ema
        ema = torch_ema.ExponentialMovingAverage(model.parameters(), decay=0.999)
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.copy_to(model.parameters())
        return True
    except Exception as e:
        logging.warning(f"Could not restore EMA weights: {e}")
        return False


def _load_test_dataset(data_path, graph_type, testset_size, in_memory):
    dataset_class = SATInMemoryDataset if in_memory else SATOnDiskDataset
    return dataset_class(osp.join(data_path, 'test'), graph_type)[:testset_size]


def _eval_dataset(trainer, model, test_dataset, cfg):
    """Run PL trainer test on one dataset, return result dict."""
    dm = SATDataModule(
        data={'train': test_dataset, 'val': test_dataset, 'test': test_dataset},
        batch_size=cfg.data.batch_size,
        graph_type=model.model.graph_type,
        in_memory=cfg.data.get('in_memory', True),
        num_workers=cfg.data.num_workers,
        supervision_mode=model.supervision_mode,
    )
    results = trainer.test(model, dm, verbose=False)
    return results[0] if results else {}


def _run_scaling_analysis(model, test_dataset, max_test_iters, device, graph_type):
    """
    Plot avg gap vs message-passing iteration for 1..max_test_iters.

    Overrides model.num_iters and collect_all_votes at test time so the model
    runs beyond its training iteration count. Returns array of shape [max_test_iters].
    """
    from torch_geometric.loader import DataLoader

    original_num_iters = model.num_iters
    original_collect = model.model.collect_all_votes

    model.num_iters = max_test_iters
    model.model.collect_all_votes = True
    model.eval()

    loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                        follow_batch=['x_l', 'x_v', 'x_c'])
    gap_sums = np.zeros(max_test_iters)
    n_instances = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model.model(batch, max_test_iters)
            all_votes = outputs['all_votes']  # list of [N_vars, 2], length=max_test_iters
            gaps_per_iter = compute_gap_per_iteration(all_votes, batch, graph_type)
            gap_sums += gaps_per_iter * batch.num_graphs
            n_instances += batch.num_graphs

    model.num_iters = original_num_iters
    model.model.collect_all_votes = original_collect

    return gap_sums / max(n_instances, 1)


def _save_gap_bar_chart(dataset_names, avg_gaps, plots_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5, len(dataset_names) * 1.5), 4))
    bars = ax.bar(dataset_names, avg_gaps)
    ax.set_ylabel('Average Gap (unsatisfied clauses)')
    ax.set_title('Average Gap per Dataset')
    for bar, val in zip(bars, avg_gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    path = osp.join(plots_dir, 'avg_gap_per_dataset.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info(f"Saved bar chart: {path}")


def _save_scaling_plot(gaps_per_dataset, train_iters, plots_dir):
    """
    gaps_per_dataset: dict mapping dataset_name -> np.array of shape [max_test_iters]
    Saves a single combined plot with one line per dataset.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for ds_name, avg_gaps in gaps_per_dataset.items():
        iters = np.arange(1, len(avg_gaps) + 1)
        ax.plot(iters, avg_gaps, linewidth=1.5, label=ds_name)

    # Draw training-iteration marker using the length of the first dataset's array
    first_gaps = next(iter(gaps_per_dataset.values()))
    if train_iters is not None and train_iters <= len(first_gaps):
        ax.axvline(x=train_iters, linestyle='--', color='gray', linewidth=1,
                   label=f'training iters ({train_iters})')

    ax.legend()
    ax.set_xlabel('Message-passing iterations')
    ax.set_ylabel('Average Gap (unsatisfied clauses)')
    ax.set_title('Test-time scaling: gap vs iterations')
    ax.set_xlim(1, len(first_gaps))
    plt.tight_layout()
    path = osp.join(plots_dir, 'scaling_gap_vs_iters.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info(f"Saved scaling plot: {path}")



@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    from hydra.utils import to_absolute_path
    logging.basicConfig(level=logging.INFO)

    checkpoint_path = cfg.get('checkpoint', None)
    if checkpoint_path is None:
        raise ValueError(
            "Provide checkpoint path, e.g.:\n"
            "  python eval.py checkpoint=checkpoints/my_model_best.ckpt"
        )
    checkpoint_path = to_absolute_path(checkpoint_path)
    if not osp.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if cfg.system.get('gpu_id') is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.system.gpu_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------ #
    # Load model — all training settings restored from checkpoint         #
    # ------------------------------------------------------------------ #
    model = LitModel.load_from_checkpoint(checkpoint_path, strict=False)
    ema_loaded = _restore_ema(model.model, checkpoint_path)
    train_iters = model.num_iters
    logging.info(
        f"Loaded:  supervision={model.supervision_mode} | "
        f"graph={model.model.graph_type} | "
        f"update={model.model.update_type} | "
        f"d={model.model.d_model} | iters={train_iters} | "
        f"EMA={'yes' if ema_loaded else 'no'}"
    )

    num_test_samples = cfg.train.get('num_test_samples', 1)
    model.num_test_samples = num_test_samples
    if num_test_samples > 1:
        logging.info(f"Test-time resampling: {num_test_samples} samples/instance")

    graph_type = model.model.graph_type
    in_memory = cfg.data.get('in_memory', True)
    testset_size = cfg.data.testset_size

    # ------------------------------------------------------------------ #
    # Resolve dataset paths                                               #
    # ------------------------------------------------------------------ #
    eval_cfg = cfg.get('eval', {})
    dataset_paths_raw = eval_cfg.get('datasets', None)

    if dataset_paths_raw is not None:
        if isinstance(dataset_paths_raw, str):
            dataset_paths = [dataset_paths_raw]
        else:
            dataset_paths = list(dataset_paths_raw)
        dataset_paths = [to_absolute_path(p) for p in dataset_paths]
    else:
        dataset_paths = [to_absolute_path(cfg.data.data_path)]

    plots_dir = to_absolute_path(eval_cfg.get('plots_dir', './eval_plots'))
    save_plots = eval_cfg.get('save_plots', True)
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Multi-dataset evaluation                                            #
    # ------------------------------------------------------------------ #
    import pytorch_lightning as pl

    logger = None
    if cfg.logging.get('use_wandb', False):
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            project=cfg.logging.get('wandb_project', 'SAT_GNN'),
            name=f"eval_{osp.basename(checkpoint_path)}",
            log_model=False,
        )

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        enable_progress_bar=True,
    )

    all_results = {}
    for ds_path in dataset_paths:
        ds_name = osp.basename(ds_path)
        logging.info(f"\nEvaluating on: {ds_path}")
        test_dataset = _load_test_dataset(ds_path, graph_type, testset_size, in_memory)
        logging.info(f"  {len(test_dataset)} instances")
        results = _eval_dataset(trainer, model, test_dataset, cfg)
        all_results[ds_name] = results

    # ------------------------------------------------------------------ #
    # Print results table                                                 #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for ds_name, r in all_results.items():
        print(f"\nDataset: {ds_name}")
        print(f"  Dec. Acc.       : {r.get('test_dec_acc', 0):.4f}")
        print(f"  SAT Acc.        : {r.get('test_sat_acc', 0):.4f}")
        print(f"  Avg Gap         : {r.get('test_avg_gap', 0):.4f}")
        print(f"  Avg Gap (SAT)   : {r.get('test_avg_gap_on_sat', 0):.4f}")
        print(f"  Avg Gap (UNSAT) : {r.get('test_avg_gap_on_unsat', 0):.4f}")
        if num_test_samples > 1:
            print(f"  [{num_test_samples} samples/instance]")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Bar chart: avg gap per dataset                                      #
    # ------------------------------------------------------------------ #
    if save_plots and len(all_results) > 1:
        names = list(all_results.keys())
        gaps = [all_results[n].get('test_avg_gap', 0) for n in names]
        _save_gap_bar_chart(names, gaps, plots_dir)

    # ------------------------------------------------------------------ #
    # Test-time scaling analysis                                          #
    # ------------------------------------------------------------------ #
    scaling = eval_cfg.get('scaling_analysis', False)
    if scaling:
        max_test_iters = int(eval_cfg.get('max_test_iters', 100))
        logging.info(f"\nRunning test-time scaling analysis (1..{max_test_iters} iterations)...")

        model.to(device)
        gaps_per_dataset = {}
        for ds_path in dataset_paths:
            ds_name = osp.basename(ds_path)
            logging.info(f"  Scaling analysis: {ds_name}")
            scaling_dataset = _load_test_dataset(ds_path, graph_type, testset_size, in_memory)
            avg_gaps = _run_scaling_analysis(
                model, scaling_dataset, max_test_iters, device, graph_type
            )
            gaps_per_dataset[ds_name] = avg_gaps

            print(f"\nScaling analysis ({ds_name}):")
            print(f"  Gap at iter   1 : {avg_gaps[0]:.4f}")
            print(f"  Gap at iter  {train_iters:2d} : {avg_gaps[min(train_iters, max_test_iters) - 1]:.4f}")
            print(f"  Gap at iter {max_test_iters:3d} : {avg_gaps[-1]:.4f}")

        if save_plots:
            _save_scaling_plot(gaps_per_dataset, train_iters, plots_dir)


if __name__ == '__main__':
    main()
