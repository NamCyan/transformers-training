#!/bin/bash
export TRANSFORMERS_CACHE="/cm/archive/namlh31/AI4Code/.cache"
export HF_DATASETS_CACHE="/cm/archive/namlh31/AI4Code/.cache"
export https_proxy=http://10.16.29.10:8080

# Deepspeed issue see https://github.com/huggingface/peft/issues/306
# While using deepspeed remove line --low_cpu_mem_usage and --device_map
# Training with int8 is so slow. Can we fix?

# LOCAL_RANK=0,1,2,3 CUDA_VISIBLE_DEVICES=0,3,6,7 python -m torch.distributed.launch \
#     --nproc_per_node 4 --use-env peft-finetuning-enc-dec.py \
#     --ddp_find_unused_parameters False \
#     --deepspeed /home/namlh31/project/AI4Code/TheVault_exp/configs/ds_config_zero3.json \
CUDA_VISIBLE_DEVICES=0,3,6,7 python peft-finetuning-enc-dec.py \
    --model_name_or_path Salesforce/codet5-large  \
    --do_train \
    --do_Eval \
    --max_output_length 400 \
    --max_input_length 400 \
    --padding_side right \
    --save_total_limit 5 \
    --seed 42 \
    --num_proc 50 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --auto_find_batch_size \
    --logging_strategy steps \
    --evaluation_strategy steps \
    --eval_steps 20000 \
    --save_steps 20000 \
    --load_best_model_at_end \
    --metric_for_best_model accuracy \
    --num_train_epochs 1 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --input_column docstring \
    --output_column code \
    --lora_config_path /home/namlh31/project/AI4Code/TheVault_exp/configs/lora_enc_dec_config.yaml \
    --train_file /cm/archive/namlh31/AI4Code/codebridgedata/splits/python/train.json \
    --validation_file /cm/archive/namlh31/AI4Code/codebridgedata/splits/python/valid.json \
    --output_dir /cm/archive/namlh31/AI4Code/codebridgedata/exps/codet5-large-python4M-fp16-1epoch-peft \
    --cache_dir /cm/archive/namlh31/AI4Code/.cache \
    --gradient_checkpointing \
    --tf32 True \
    --low_cpu_mem_usage \
    --device_map auto \
    # --load_in_8bit \
    # --prefix_prompt Docstring: \
    # --postfix_prompt Code: \