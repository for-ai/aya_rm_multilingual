#!/bin/bash

# Define the languages and their FLORES-200 codes
declare -A languages=(
    ["arb_Arab"]="Arabic"
    ["zho_Hans"]="Chinese_Simplified"
    ["zho_Hant"]="Chinese_Traditional"
    ["ces_Latn"]="Czech"
    ["nld_Latn"]="Dutch"
    ["eng_Latn"]="English"
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
MAX_LENGTH=1024
BATCH_SIZE=8
OUTPUT_DIR="translations"

# Loop through each language and call the translation script
for lang_code in "${!languages[@]}"; do
    language="${languages[$lang_code]}"
    echo "Translating to $language ($lang_code)"
    
    python translate_preference_pairs_nllb.py \
        --dataset_name "$DATASET_NAME" \
        --target_language "$lang_code" \
        --columns_to_translate "${COLUMNS_TO_TRANSLATE[@]}" \
        --max_length "$MAX_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR"
    
    echo "Translation to $language ($lang_code) completed."
done
