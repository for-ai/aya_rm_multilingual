#!/bin/bash

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

# Define the dataset name and columns to translate
DATASET_NAME="allenai/reward-bench"
COLUMNS_TO_TRANSLATE=("prompt" "chosen" "rejected")
MAX_LENGTH=512
SUBSET_SIZE=1000000
OUTPUT_DIR="translations"

# Get the list of GPU IDs
GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '))
NUM_GPUS=${#GPUS[@]}

# Function to run translation on a specific GPU
run_translation() {
    local lang_code=$1
    local gpu_id=$2
    local language=${languages[$lang_code]}
    echo "Translating to $language ($lang_code) on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id python translate_preference_pairs_nllb.py \
        --dataset_name "$DATASET_NAME" \
        --target_language "$lang_code" \
        --columns_to_translate "${COLUMNS_TO_TRANSLATE[@]}" \
        --max_length "$MAX_LENGTH" \
        --subset_size "$SUBSET_SIZE" \
        --output_dir "$OUTPUT_DIR" &
}

# Loop through each language in groups of 4 and assign them to GPUs
lang_codes=(${!languages[@]})
total_langs=${#lang_codes[@]}

for ((i=0; i<total_langs; i+=NUM_GPUS)); do
    for ((j=0; j<NUM_GPUS && i+j<total_langs; j++)); do
        lang_code=${lang_codes[i+j]}
        run_translation $lang_code ${GPUS[j]}
    done
    wait  # Wait for all background processes to finish before starting the next group
done

echo "All translations completed."
