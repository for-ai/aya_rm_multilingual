#!/bin/bash

# Function to display usage information
usage() {
  echo "Usage: $0 [MODEL] [DATASET] [OUTDIR]"
  echo "  MODEL - The model to evaluate (required)"
  echo "  DATASET - The dataset to use (optional, default is 'aya-rm-multilingual/multilingual-reward-bench')"
  echo "  OUTDIR  - The output directory (optional, default is 'output/')"
  exit 1
}

# Default values for arguments
MODEL=""
DATASET="aya-rm-multilingual/multilingual-reward-bench"
OUTDIR="output/"

# Check and assign arguments if provided
if [ $# -gt 3 ]; then
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

for lang_code in "${!languages[@]}"; do
  python3 scripts/run_generative.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --lang_code "$lang_code" \
    --split "filtered" \
    --output_dir "$OUTDIR" \
    --include_languages "${languages[$lang_code]}" "English" \
    --trust_remote_code \
    --save_all
done
