#!/bin/bash
# End-to-end quickstart: generate data, train, evaluate.
#
# This script trains a GNN for SAT solving using the unsupervised quadratic loss
# on SR40 data (Selsam-Random, 3-40 variables). After training it evaluates the
# model on both SR40 and SR100 test sets and produces test-time scaling plots.
#
# Training configuration:
#   - Model: RNN + Variable-Clause Graph (VCG)
#   - Edge feature MLPs: separate MLPs for positive/negative literal edges
#   - Embedding normalisation: L2 after each message-passing iteration
#   - Loss: unsupervised quadratic  L_quad = sum_c (1 - V_c)^2
#   - LR schedule: cosine decay
#   - Training set: 30 000 SR40 instances (15 000 pairs, 3-40 variables)
#
# Requirements: pip install -r requirements.txt
# GPU: recommended (CUDA). Set GPU_ID below.
#
# Expected runtime: several hours on a single GPU for 300 epochs.
# To reduce training time, lower num_epochs (e.g. train.num_epochs=50).

set -e

GPU_ID=0
OUTPUT_DIR="quickstart_output"

echo "=== Step 1: Generate datasets ==="
bash data/generate_datasets.sh --quick

echo ""
echo "=== Step 2: Train ==="
python train.py \
    system.gpu_id=$GPU_ID \
    model.update_type=rnn \
    model.graph_type=var \
    model.use_edge_features=True \
    model.normalize_embeddings=True \
    train.supervision_mode=unsupervised_quad \
    train.lr_schedule=cosine \
    train.num_epochs=300 \
    train.curriculum.enabled=False \
    data.dataset_name=sr40 \
    data.trainset_size=30000 \
    data.valset_size=1000 \
    data.testset_size=1000 \
    data.curriculum_val_path=null \
    logging.use_wandb=False \
    hydra.run.dir=$OUTPUT_DIR

echo ""
echo "=== Step 3: Evaluate ==="
python eval.py \
    system.gpu_id=$GPU_ID \
    checkpoint=$OUTPUT_DIR/checkpoints/last.ckpt \
    "eval.datasets=[./data/cnfs/sr40,./data/cnfs/sr100]" \
    eval.scaling_analysis=True \
    eval.max_test_iters=100 \
    eval.plots_dir=$OUTPUT_DIR/eval_plots \
    data.testset_size=1000 \
    logging.use_wandb=False

echo ""
echo "=== Done ==="
echo "Plots saved to: $OUTPUT_DIR/eval_plots/"
