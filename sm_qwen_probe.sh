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

cd /N/slate/mealieff/emoRep/

module load conda
conda activate qwen_probe

export HF_DATASETS_CACHE=/N/slate/mealieff/hf_cache/datasets
export HF_MODULES_CACHE=/N/slate/mealieff/hf_cache/modules
export HF_METRICS_CACHE=/N/slate/mealieff/hf_cache/metrics
mkdir -p $HF_DATASETS_CACHE $HF_MODULES_CACHE $HF_METRICS_CACHE


python qwen_probe.py

