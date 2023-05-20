#!/bin/bash
export TRANSFORMERS_CACHE="/cm/archive/namlh31/AI4Code/cache"
export HF_DATASETS_CACHE="/cm/archive/namlh31/AI4Code/cache"
export https_proxy=http://10.16.29.10:8080

CUDA_VISIBLE_DEVICES=0 python3 test_causal.py \
    --model_name_or_path Salesforce/codegen-2B-multi \
    --seed 42 \
    --do_train \
    --do_eval \
    --preprocessing_num_workers 50 \
    --logging_strategy steps \
    --evaluation_strategy steps \
    --eval_steps 50000 \
    --save_steps 50000 \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --lora_config_path /home/namlh31/project/AI4Code/TheVault_exp/configs/lora_causal_config.yaml \
    --train_file /cm/archive/namlh31/AI4Code/codebridgedata/splits/python/train.json \
    --validation_file /cm/archive/namlh31/AI4Code/codebridgedata/splits/python/valid.json \
    --output_dir /cm/archive/namlh31/AI4Code/codebridgedata/exps/codegen-2B-multi-python-4M-1epoch \
    --save_total_limit 5 \
    --fp16 \
    --low_cpu_mem_usage \
    --gradient_checkpointing \
    --overwrite_output_dir
    # --load_in_8bit \