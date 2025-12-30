# GNN for SAT Approximation

Graph Neural Network for Boolean Satisfiability (SAT) solving using message passing on variable-clause graphs.

## Features

- Multiple GNN architectures: RNN, LSTM, and Primal-Dual inspired
- Variable-Clause Graph (VCG) representation with support for Literal-Clause Graph (LCG)
- Various supervision modes: assignment prediction, SAT/UNSAT classification, and unsupervised learning
- Configurable training with Hydra and experiment tracking with WandB

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train.py
```

Configuration can be modified in `conf/config.yaml` or via command-line arguments using Hydra.
