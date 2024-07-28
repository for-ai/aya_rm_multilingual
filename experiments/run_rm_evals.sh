#!/bin/bash

export TRANSFORMERS_CACHE="./cache/"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1

# Function to display usage information
usage() {
  echo "Usage: $0 [DATASET] [SPLIT] [OUTDIR]"
  echo "  LANG - The language code to use (required)"
  echo "  DATASET - The dataset to use (optional, default is 'aya-rm-multilingual/multilingual-reward-bench')"
  echo "  OUTDIR  - The output directory (optional, default is 'output/')"
  exit 1
}

# Default values for arguments
LANG=""
DATASET="aya-rm-multilingual/multilingual-reward-bench"
OUTDIR="output/"

# Check and assign arguments if provided
if [ $# -gt 3 ]; then
  echo "Error: Too many arguments."
  usage
elif [ $# -ge 1 ]; then
  DATASET=$1
fi

if [ $# -ge 2 ]; then
  LANG=$2
fi

if [ $# -ge 3 ]; then
  OUTDIR=$3
fi

python3 scripts/rewardbench.py \
    --model openbmb/UltraRM-13b \
    --chat_template openbmb \
    --dataset $DATASET \
    --lang_code $LANG \
    --split "filtered" \
    --output_dir $OUTDIR \
    --batch_size 8 \
    --trust_remote_code \
    --force_truncation \
    --save_all 

python3 scripts/rewardbench.py \
    --model OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
    --chat_template oasst_pythia \
    --dataset $DATASET \
    --lang_code $LANG \
    --split "filtered" \
    --output_dir $OUTDIR \
    --batch_size 8 \
    --trust_remote_code \
    --force_truncation \
    --save_all 

python3 scripts/rewardbench.py \
    --model OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1 \
    --chat_template oasst_pythia \
    --dataset $DATASET \
    --lang_code $LANG \
    --split "filtered" \
    --output_dir $OUTDIR \
    --batch_size 16 \
    --trust_remote_code \
    --force_truncation \
    --save_all 