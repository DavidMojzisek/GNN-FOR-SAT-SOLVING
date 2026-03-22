# GNN for SAT Approximation

Graph Neural Network for Boolean Satisfiability (SAT) solving using message passing on variable-clause graphs. Supports multiple update rules (RNN, LSTM, Primal-Dual), graph representations (VCG, LCG), and supervision modes including unsupervised clause-satisfaction losses and curriculum-based SAT/UNSAT classification.

---

## General information


- Dataset title: Graph Neural Network for Boolean Satisfiability (SAT) solving using message passing on variable-clause graphs.

- Dataset DOI: 10.5281/zenodo.18740316

- Author: David Mojžíšek, ORCID: 0000-0002-3867-644X

- Affiliation: University of Ostrava
- ROR: https://ror.org/00pyqav47

- Description: Dataset for Article "Neural approaches to SAT solving: Design choices and interpretability" and PhD Thesis "Graph Neural Networks for Constraint Satisfaction: Theory, Design Space, and Connections to Continuous Relaxation Methods"

- Funding sources: This work has been produced with the financial support of the European Union under the: 'Biography of Fake News with a Touch of AI: Dangerous Phenomenon through the Prism of Modern Human Sciences' project no.: CZ.02.01.01/00/23_025/0008724 via the Operational Programme Jan Ámos Komenský.

- Language of dataset: English

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quickstart

The fastest way to try the full workflow — data generation, training, and evaluation with plots:

```bash
bash quickstart.sh
```

This generates SR40 data (30 000 instances, 3–40 variables), trains an RNN with the unsupervised quadratic loss for 300 epochs, then evaluates on SR40 and SR100 test sets and saves test-time scaling plots to `quickstart_output/eval_plots/`.

To get results faster, reduce the epoch count:

```bash
python train.py \
  model.update_type=rnn model.graph_type=var \
  model.use_edge_features=True model.normalize_embeddings=True \
  train.supervision_mode=unsupervised_quad train.lr_schedule=cosine \
  train.num_epochs=50 train.curriculum.enabled=False \
  data.dataset_name=sr40 logging.use_wandb=False \
  hydra.run.dir=quickstart_output
```

---

## Data generation

Three dataset types are provided:

- `sr40` — Selsam-Random SAT/UNSAT pairs, 3–40 variables (training)
- `sr100` — Selsam-Random SAT/UNSAT pairs, 12–100 variables (evaluation)
- `3sat100` — Random 3-SAT at ratio 4.26, 40–100 variables (evaluation)

```bash
# Quick datasets for trying things out (used by quickstart.sh)
bash data/generate_datasets.sh --quick

# Full datasets for complete experiments
bash data/generate_datasets.sh --full
```

Data is written to `data/cnfs/<name>/{train,val,test}/`. Setting `data.dataset_name` in the config resolves the path automatically.

---

## Training

Edit `conf/config.yaml` and run:

```bash
python train.py
```

All config values can also be overridden from the command line using Hydra syntax:

```bash
python train.py train.supervision_mode=unsupervised_quad train.num_epochs=100
```

The config file is extensively commented — it is the primary reference for all available options. Key things to set: `model.update_type`, `model.graph_type`, `train.supervision_mode`, `data.dataset_name`. Curriculum learning is only active when `train.supervision_mode=sat`.

---

## Evaluation

```bash
# Evaluate on a single dataset
python eval.py checkpoint=path/to/last.ckpt

# Evaluate on multiple datasets and produce a gap comparison bar chart
python eval.py checkpoint=path/to/last.ckpt \
  "eval.datasets=[./data/cnfs/sr40,./data/cnfs/sr100,./data/cnfs/3sat100]"

# Test-time scaling: plot how avg gap changes from 1 to 100 message-passing iterations
python eval.py checkpoint=path/to/last.ckpt \
  eval.scaling_analysis=True eval.max_test_iters=100

# Test-time resampling: run N forward passes per formula, keep best assignment
python eval.py checkpoint=path/to/last.ckpt train.num_test_samples=5
```

The model architecture is restored automatically from the checkpoint. Plots are saved to `eval.plots_dir` (default `./eval_plots`).

---

## Legal and ethical aspects

- License: CC BY 4.0
- Conditions of use: https://creativecommons.org/licenses/by/4.0/

---

*Development assisted by Claude (Anthropic).*
