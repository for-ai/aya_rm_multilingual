#!/bin/bash

export TRANSFORMERS_CACHE="./cache/"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1

# Function to display usage information
usage() {
  echo "Usage: $0 [MODEL] [DATASET] [OUTDIR]"
  echo "  MODEL - The model to evaluate (required)"
  echo "  DATASET - The dataset to use (optional, default is 'aya-rm-multilingual/multilingual-reward-bench')"
  echo "  OUTDIR  - The output directory (optional, default is 'output/')"
  echo "  CHAT_TEMPLATE  - The chat template to use (optional, default is 'raw')"
  echo "  BATCH_SIZE     - The batch size to use (optional, default is 8)"
  exit 1
}

# Default values for arguments
MODEL=""
DATASET="aya-rm-multilingual/multilingual-reward-bench"
OUTDIR="output/"
CHAT_TEMPLATE="raw"
BATCH_SIZE=8

# Check and assign arguments if provided
if [ $# -gt 5 ]; then
  echo "Error: Too many arguments."
  usage
elif [ $# -ge 1 ]; then
  MODEL=$1
fi

if [ $# -ge 2 ]; then
  DATASET=$2
fi

if [ $# -ge 3 ]; then
  OUTDIR=$3
fi

if [ $# -ge 4 ]; then
  CHAT_TEMPLATE=$4
fi

if [ $# -ge 5 ]; then
  BATCH_SIZE=$5
fi

# Ensure the model is provided
if [ -z "$MODEL" ]; then
  echo "Error: MODEL is required."
  usage
fi


# Define the languages and their FLORES-200 codes
declare -A languages=(
    ["arb_Arab"]="Arabic"
    ["zho_Hans"]="Chinese_Simplified"
    ["zho_Hant"]="Chinese_Traditional"
    ["ces_Latn"]="Czech"
    ["nld_Latn"]="Dutch"
    ["fra_Latn"]="French"
    ["deu_Latn"]="German"
    ["ell_Grek"]="Greek"
    ["heb_Hebr"]="Hebrew"
    ["hin_Deva"]="Hindi"
    ["ind_Latn"]="Indonesian"
    ["ita_Latn"]="Italian"
    ["jpn_Jpan"]="Japanese"
    ["kor_Hang"]="Korean"
    ["pes_Arab"]="Persian"
    ["pol_Latn"]="Polish"
    ["por_Latn"]="Portuguese"
    ["ron_Latn"]="Romanian"
    ["rus_Cyrl"]="Russian"
    ["spa_Latn"]="Spanish"
    ["tur_Latn"]="Turkish"
    ["ukr_Cyrl"]="Ukrainian"
    ["vie_Latn"]="Vietnamese"
)

# Loop through each language and run the command
for lang_code in "${!languages[@]}"; do
  python3 -m scripts.run_rewardbench \
    --model "$MODEL" \
    --chat_template "$CHAT_TEMPLATE" \
    --dataset "$DATASET" \
    --lang_code "$lang_code" \
    --split "filtered" \
    --output_dir "$OUTDIR" \
    --batch_size "$BATCH_SIZE" \
    --trust_remote_code \
    --force_truncation \
    --save_all
done
