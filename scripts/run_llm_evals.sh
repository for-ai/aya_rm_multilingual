python3 -m scripts/run_generative.py \
    --dataset_name ljvmiranda921/ultrafeedback-multilingual-dpo-test \
    --model gpt-4-turbo-2024-04-09 \
    --split test
python3 -m scripts/run_generative.py \
    --dataset_name ljvmiranda921/ultrafeedback-english-dpo-test \
    --model gpt-4-turbo-2024-04-09 \
    --split test