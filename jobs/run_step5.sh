#!/bin/bash
# =====================================================================
# SLURM Job Script — Step 5: MLP Deep Learning Baseline
# =====================================================================
#
# HOW TO USE:
#   sbatch jobs/run_step5.sh
#
# MONITOR:
#   squeue -u $USER
#   tail -f logs/step5_mlp_<JOB_ID>.out
#   scancel <JOB_ID>
#
# FIRST TIME SETUP (run once on login node):
#   python3 -m venv /sci/labs/arieljaffe/dan.abergel1/fair_env
#   source /sci/labs/arieljaffe/dan.abergel1/fair_env/bin/activate
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   pip install numpy pandas scikit-learn matplotlib seaborn tqdm
# =====================================================================

# ── SLURM resource request ─────────────────────────────────────────
#SBATCH --job-name=step5-mlp
#SBATCH --gres=gpu:l40s:1        # 1x NVIDIA L40S (48 GB VRAM)
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/step5_mlp_%j.out
#SBATCH --error=logs/step5_mlp_%j.err

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────
LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
PROJECT_DIR="$LAB_DIR/FAIR"
VENV_DIR="$LAB_DIR/torch_env"

# ── Redirect caches to lab storage (home quota ~5 GB) ──────────────
export TMPDIR="$LAB_DIR/tmp"
export PIP_CACHE_DIR="$LAB_DIR/cache/pip"
export XDG_CACHE_HOME="$LAB_DIR/cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

echo "============================================================"
echo "  Step 5: MLP Deep Learning Baseline — SLURM Job"
echo "============================================================"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $(hostname)"
echo "  Date:       $(date)"
echo "  Project:    $PROJECT_DIR"
echo "  Venv:       $VENV_DIR"
echo "============================================================"
echo ""

# ── 1. Activate virtual environment ────────────────────────────────
echo "[1/4] Activating venv ..."
source "$VENV_DIR/bin/activate"
echo "  Python: $(which python3)"
echo "  Version: $(python3 --version)"
echo ""

# ── 2. Update code from GitHub ─────────────────────────────────────
echo "[2/4] Updating code ..."
cd "$PROJECT_DIR"
git fetch --all
git reset --hard origin/main
echo "  Commit: $(git rev-parse --short HEAD)"
echo "  Message: $(git log -1 --pretty=%s)"
echo ""

# ── 3. GPU check ───────────────────────────────────────────────────
echo "[3/4] GPU check ..."
python3 -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
props = torch.cuda.get_device_properties(0)
print(f'  VRAM: {props.total_memory / 1e9:.1f} GB')
"
echo ""

# ── 4. Run step5 ───────────────────────────────────────────────────
echo "[4/4] Starting step5_mlp.py ..."
mkdir -p "$PROJECT_DIR/logs"

python3 -u "$PROJECT_DIR/src/step5_mlp.py"

echo ""
echo "============================================================"
echo "  Job finished: $(date)"
echo "============================================================"
