#!/bin/bash
export TRANSFORMERS_CACHE="/cm/archive/namlh31/AI4Code/.cache"
export HF_DATASETS_CACHE="/cm/archive/namlh31/AI4Code/.cache"
export https_proxy=http://10.16.29.10:8080


# Training with fp16/bf16/tf32. See https://huggingface.co/docs/transformers/v4.15.0/performance.
# Deepspeed issue see https://github.com/huggingface/peft/issues/306
# While using deepspeed remove line --low_cpu_mem_usage, --device_map and load_in_8bit
# Training with int8 is so slow. Can we fix?

# CUDA_VISIBLE_DEVICES=0 peft-finetuning-causal.py \


LOCAL_RANK=0,1,2,3,4 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch \
    --nproc_per_node 5 --use-env peft-finetuning-causal.py \
    --ddp_find_unused_parameters False \
    --deepspeed /home/namlh31/project/AI4Code/TheVault_exp/configs/ds_config_zero3.json \
    --model_name_or_path /cm/archive/namlh31/models/starcoder/ \
    --dataset_name_or_path Fsoft-AIC/the-vault-function \
    --do_train \
    --do_eval \
    --seed 42 \
    --num_proc 50 \
    --ignore_input_token_label \
    --padding_side left \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --auto_find_batch_size \
    --logging_strategy steps \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 100 \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --weight_decay 3e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_steps 1000 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 5 \
    --input_column docstring \
    --output_column code \
    --max_output_length 400 \
    --max_input_length 400 \
    --prefix_prompt="<comment>" \
    --postfix_prompt="<code>" \
    --output_dir /cm/archive/namlh31/AI4Code/codebridgedata/exps/starcoder-python-4M-1epoch \
    --cache_dir /cm/archive/namlh31/AI4Code/.cache \
    --lora_config_path /home/namlh31/project/AI4Code/TheVault_exp/configs/lora_causal_config.yaml \
    --bf16 \
    --gradient_checkpointing \
    --new_tokens="<comment>;<code>" \
    # --device_map auto \
    # --low_cpu_mem_usage \
    # --load_in_8bit \