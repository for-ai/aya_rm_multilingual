python -m scripts.convert_multilingual_uf
mkdir outputs
rewardbench \
    --model OpenAssistant/reward-model-deberta-v3-large-v2 \
    --dataset data/multilingual-ultrafeedback-dpo-v0.1.json \
    --output_dir output \
    --load_json \
    --batch_size 8 \
    --trust_remote_code \
    --force_truncation \
    --save_all 