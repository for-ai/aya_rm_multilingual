#!/bin/bash

# Function to display usage information
usage() {
  echo "Usage: $0 [DATASET] [SPLIT] [OUTDIR]"
  echo "  DATASET - The dataset to use (optional, default is 'ljvmiranda921/multilingual-ultrafeedback-dpi-v0.1-test')"
  echo "  SPLIT   - The data split to use (optional, default is 'test')"
  echo "  OUTDIR  - The output directory (optional, default is 'output/')"
  exit 1
}

# Default values for arguments
DATASET="ljvmiranda921/multilingual-ultrafeedback-dpi-v0.1-test"
SPLIT="test"
OUTDIR="output/"

# Check and assign arguments if provided
if [ $# -gt 3 ]; then
  echo "Error: Too many arguments."
  usage
elif [ $# -ge 1 ]; then
  DATASET=$1
fi

if [ $# -ge 2 ]; then
  SPLIT=$2
fi

if [ $# -ge 3 ]; then
  OUTDIR=$3
fi

rewardbench \
    --model openbmb/UltraRM-13b \
    --chat_template openbmb \
    --dataset $DATASET \
    --split $SPLIT \
    --output_dir $OUTDIR \
    --batch_size 8 \
    --trust_remote_code \
    --force_truncation \
    --save_all 

rewardbench \
    --model OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
    --chat_template oasst_pythia \
    --dataset $DATASET \
    --split $SPLIT \
    --output_dir $OUTDIR \
    --batch_size 8 \
    --trust_remote_code \
    --force_truncation \
    --save_all 

rewardbench \
    --model OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1 \
    --chat_template oasst_pythia \
    --dataset $DATASET \
    --split $SPLIT \
    --output_dir $OUTDIR \
    --batch_size 16 \
    --trust_remote_code \
    --force_truncation \
    --save_all 

rewardbench \
    --model OpenAssistant/reward-model-deberta-v3-large-v2 \
    --chat_template raw \
    --dataset $DATASET \
    --split $SPLIT \
    --output_dir $OUTDIR \
    --batch_size 64 \
    --trust_remote_code \
    --force_truncation \
    --save_all 

rewardbench \
    --model berkeley-nest/Starling-RM-7B-alpha \
    --tokenizer meta-llama/Llama-2-7b-chat-hf \
    --chat_template llama-2 \
    --dataset $DATASET \
    --split $SPLIT \
    --output_dir $OUTDIR \
    --batch_size 16 \
    --trust_remote_code \
    --force_truncation \
    --save_all 

rewardbench \
    --model sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --tokenizer sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --dataset $DATASET \
    --split $SPLIT \
    --output_dir $OUTDIR \
    --batch_size 4 \
    --trust_remote_code \
    --force_truncation \
    --save_all 

rewardbench \
    --model openbmb/Eurus-RM-7b \
    --tokenizer openbmb/Eurus-RM-7b \
    --chat_template mistral \
    --dataset $DATASET \
    --split $SPLIT \
    --output_dir $OUTDIR \
    --batch_size 16 \
    --trust_remote_code \
    --force_truncation \
    --save_all 

rewardbench \
    --model allenai/tulu-v2.5-13b-preference-mix-rm \
    --tokenizer allenai/tulu-v2.5-13b-preference-mix-rm \
    --chat_template mistral \
    --dataset $DATASET \
    --split $SPLIT \
    --output_dir $OUTDIR \
    --batch_size 4 \
    --trust_remote_code \
    --force_truncation \
    --save_all 