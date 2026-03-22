#!/bin/bash
# Generate SAT datasets for training and evaluation.
#
# Usage:
#   bash data/generate_datasets.sh --quick     # try-it datasets only (fast)
#   bash data/generate_datasets.sh --full      # full training + eval datasets (slow)
#
# Datasets generated:
#
#   SR40        Selsam-Random pairs, 3-40 variables    (training only)
#   SR100       Selsam-Random pairs, 12-100 variables  (training only)
#   SR200       Selsam-Random pairs, exactly 200 vars  (eval only)
#   SR400       Selsam-Random pairs, exactly 400 vars  (eval only)
#   3SAT100     Random 3-SAT, 40-100 variables, ratio 4.26  (training only)
#   3SAT200     Random 3-SAT, exactly 200 vars, ratio 4.26  (eval only)
#
# val/ and test/ splits always contain instances at the maximum (fixed) size only.
# Training ranges are used for train/ only.
#
# Curriculum training note:
#   When using supervision_mode=sat with curriculum.enabled=True, two val sets are used:
#     - sr40/val/           (n=40 exactly) — tracks generalisation to the target size;
#                           used for checkpointing and early stopping (val_dec_acc)
#     - sr40_curriculum/val/ (n=3..40, mixed) — tracks performance at the current
#                           curriculum stage; used for stage-advancement decisions
#                           (val_curr_dec_acc). Generated only in --full mode.
#   Both val sets are required for curriculum training to work as intended.
#   Set data.curriculum_val_path in config to point to sr40_curriculum/val.
#
# Quick mode (--quick):
#   SR40     : 15 000 pairs train (3-40), 500 pairs val (n=40), 500 pairs test (n=40)
#   SR100    : 500 pairs test only (n=100)
#   3SAT100  : 1 000 instances test only (n=100)
#
# Full mode (--full):
#   SR40     : 25 000 pairs train (3-40), 1 000 pairs val (n=40), 1 000 pairs test (n=40)
#   SR40 curriculum val: ~3 800 instances, mixed 3-40 vars (100 pairs per size level)
#   SR100    : 25 000 pairs train (12-100), 1 000 pairs val (n=100), 1 000 pairs test (n=100)
#   SR200    : 1 000 pairs test only (n=200)
#   SR400    : 1 000 pairs test only (n=400)
#   3SAT100  : 50 000 instances train (40-100), 1 000 val (n=100), 1 000 test (n=100)
#   3SAT200  : 1 000 instances test only (n=200)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT_DIR/data/cnfs"

MODE="${1:-}"
if [[ "$MODE" != "--quick" && "$MODE" != "--full" ]]; then
    echo "Usage: bash data/generate_datasets.sh --quick | --full"
    exit 1
fi

echo "Generating datasets in: $DATA_DIR  (mode: $MODE)"

# ---------------------------------------------------------------------------
# SR40
#   train: 3-40 vars (range used during training)
#   val:   n=40 exactly (fixed-size benchmark for checkpointing)
#   test:  n=40 exactly (fixed-size benchmark for final eval)
# ---------------------------------------------------------------------------
echo ""
echo "=== SR40 ==="

if [[ "$MODE" == "--quick" ]]; then
    SR40_TRAIN=15000
    SR40_VAL=500
    SR40_TEST=500
else
    SR40_TRAIN=25000
    SR40_VAL=1000
    SR40_TEST=1000
fi

echo "  train: $SR40_TRAIN pairs (3-40 vars)"
python "$SCRIPT_DIR/generate_random_sat.py" \
    --out_dir "$DATA_DIR/sr40/train" \
    --n_pairs $SR40_TRAIN --min_n 3 --max_n 40 --py_seed 0 --np_seed 0

echo "  val: $SR40_VAL pairs (n=40 exactly)"
python "$SCRIPT_DIR/generate_random_sat.py" \
    --out_dir "$DATA_DIR/sr40/val" \
    --n_pairs $SR40_VAL --min_n 40 --max_n 40 --py_seed 1 --np_seed 1

echo "  test: $SR40_TEST pairs (n=40 exactly)"
python "$SCRIPT_DIR/generate_random_sat.py" \
    --out_dir "$DATA_DIR/sr40/test" \
    --n_pairs $SR40_TEST --min_n 40 --max_n 40 --py_seed 2 --np_seed 2

# ---------------------------------------------------------------------------
# SR40 curriculum val (--full only)
#   Mixed-size set covering the full 3-40 training range (~100 pairs per size).
#   Used exclusively for curriculum stage-advancement decisions (val_curr_dec_acc).
#   Does NOT replace sr40/val — both are needed when curriculum.enabled=True.
#   Point data.curriculum_val_path at this directory in the config.
# ---------------------------------------------------------------------------
if [[ "$MODE" == "--full" ]]; then
    echo ""
    echo "=== SR40 curriculum val (mixed 3-40 vars, for curriculum stage tracking) ==="
    mkdir -p "$DATA_DIR/sr40_curriculum/val"

    for N in $(seq 3 2 39); do
        python "$SCRIPT_DIR/generate_random_sat.py" \
            --out_dir "$DATA_DIR/sr40_curriculum/val" \
            --n_pairs 100 --min_n $N --max_n $N --py_seed $((N + 100)) --np_seed $((N + 100))
    done
    echo "  done: $(ls "$DATA_DIR/sr40_curriculum/val" | wc -l) files"
fi

# ---------------------------------------------------------------------------
# SR100
#   train: 12-100 vars (range used during training)
#   val:   n=100 exactly
#   test:  n=100 exactly
# ---------------------------------------------------------------------------
echo ""
echo "=== SR100 ==="

if [[ "$MODE" == "--quick" ]]; then
    echo "  test only: 500 pairs (n=100 exactly)"
    python "$SCRIPT_DIR/generate_random_sat.py" \
        --out_dir "$DATA_DIR/sr100/test" \
        --n_pairs 500 --min_n 100 --max_n 100 --py_seed 3 --np_seed 3
else
    echo "  train: 25000 pairs (12-100 vars)"
    python "$SCRIPT_DIR/generate_random_sat.py" \
        --out_dir "$DATA_DIR/sr100/train" \
        --n_pairs 25000 --min_n 12 --max_n 100 --py_seed 3 --np_seed 3

    echo "  val: 1000 pairs (n=100 exactly)"
    python "$SCRIPT_DIR/generate_random_sat.py" \
        --out_dir "$DATA_DIR/sr100/val" \
        --n_pairs 1000 --min_n 100 --max_n 100 --py_seed 4 --np_seed 4

    echo "  test: 1000 pairs (n=100 exactly)"
    python "$SCRIPT_DIR/generate_random_sat.py" \
        --out_dir "$DATA_DIR/sr100/test" \
        --n_pairs 1000 --min_n 100 --max_n 100 --py_seed 5 --np_seed 5
fi

# ---------------------------------------------------------------------------
# SR200: eval only, n=200 exactly
# ---------------------------------------------------------------------------
if [[ "$MODE" == "--full" ]]; then
    echo ""
    echo "=== SR200 (n=200 exactly, eval only) ==="
    echo "  test: 1000 pairs"
    python "$SCRIPT_DIR/generate_random_sat.py" \
        --out_dir "$DATA_DIR/sr200/test" \
        --n_pairs 1000 --min_n 200 --max_n 200 --py_seed 10 --np_seed 10
fi

# ---------------------------------------------------------------------------
# SR400: eval only, n=400 exactly
# ---------------------------------------------------------------------------
if [[ "$MODE" == "--full" ]]; then
    echo ""
    echo "=== SR400 (n=400 exactly, eval only) ==="
    echo "  test: 1000 pairs"
    python "$SCRIPT_DIR/generate_random_sat.py" \
        --out_dir "$DATA_DIR/sr400/test" \
        --n_pairs 1000 --min_n 400 --max_n 400 --py_seed 11 --np_seed 11
fi

# ---------------------------------------------------------------------------
# 3SAT100
#   train: 40-100 vars (range used during training)
#   val:   n=100 exactly
#   test:  n=100 exactly
# ---------------------------------------------------------------------------
echo ""
echo "=== 3SAT100 ==="

if [[ "$MODE" == "--quick" ]]; then
    echo "  test only: 1000 instances (n=100 exactly)"
    python "$SCRIPT_DIR/generate_3sat.py" \
        --out_dir "$DATA_DIR/3sat100/test" \
        --n_instances 1000 --min_n 100 --max_n 100 --ratio 4.26 --py_seed 6 --np_seed 6
else
    echo "  train: 50000 instances (40-100 vars)"
    python "$SCRIPT_DIR/generate_3sat.py" \
        --out_dir "$DATA_DIR/3sat100/train" \
        --n_instances 50000 --min_n 40 --max_n 100 --ratio 4.26 --py_seed 6 --np_seed 6

    echo "  val: 1000 instances (n=100 exactly)"
    python "$SCRIPT_DIR/generate_3sat.py" \
        --out_dir "$DATA_DIR/3sat100/val" \
        --n_instances 1000 --min_n 100 --max_n 100 --ratio 4.26 --py_seed 7 --np_seed 7

    echo "  test: 1000 instances (n=100 exactly)"
    python "$SCRIPT_DIR/generate_3sat.py" \
        --out_dir "$DATA_DIR/3sat100/test" \
        --n_instances 1000 --min_n 100 --max_n 100 --ratio 4.26 --py_seed 8 --np_seed 8
fi

# ---------------------------------------------------------------------------
# 3SAT200: eval only, n=200 exactly
# ---------------------------------------------------------------------------
if [[ "$MODE" == "--full" ]]; then
    echo ""
    echo "=== 3SAT200 (n=200 exactly, ratio 4.26, eval only) ==="
    echo "  test: 1000 instances"
    python "$SCRIPT_DIR/generate_3sat.py" \
        --out_dir "$DATA_DIR/3sat200/test" \
        --n_instances 1000 --min_n 200 --max_n 200 --ratio 4.26 --py_seed 12 --np_seed 12
fi

echo ""
echo "Done. Datasets written to $DATA_DIR"
