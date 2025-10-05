#!/bin/bash
#SBATCH --j qwen_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o qwen_probe_%j.out
#SBATCH -e qwen_probe_%j.err
#SBATCH -A r00018         # SLURM account name
#SBATCH --mail-type=ALL               # Email notifications for all job events
#SBATCH --mail-user=mealieff@iu.edu   # Email address for notifications

module load python/3.10 cuda/12.1
source ~/projects/emoRep/.venv/bin/activate

python qwen_probe_q1.py

