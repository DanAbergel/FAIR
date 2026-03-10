#!/bin/bash
#SBATCH --job-name=step5_mlp
#SBATCH --output=logs/step5_mlp_%j.out
#SBATCH --error=logs/step5_mlp_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dan.abergel1@mail.huji.ac.il

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURMD_NODENAME"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "GPUs:          $SLURM_GPUS_ON_NODE"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Start time:    $(date)"
echo "Working dir:   $(pwd)"
echo "============================================"

# Load modules (adjust if Moriah uses different module names)
module load cuda
module load anaconda3

# Activate environment — adjust to your conda env or venv path
# Option A: conda
# conda activate fair

# Option B: venv on cluster
source /sci/labs/arieljaffe/dan.abergel1/fair_venv/bin/activate

# Verify GPU is visible
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU only')"

echo "============================================"
echo "Starting step5_mlp.py"
echo "============================================"

cd /sci/labs/arieljaffe/dan.abergel1/FAIR
python src/step5_mlp.py

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"
