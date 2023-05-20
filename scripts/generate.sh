#!/bin/bash
export TRANSFORMERS_CACHE="/cm/archive/namlh31/AI4Code/.cache"
export HF_DATASETS_CACHE="/cm/archive/namlh31/AI4Code/.cache"
export https_proxy=http://10.16.29.10:8080

CUDA_VISIBLE_DEVICES=5 python generate.py \
    --model_name_or_path /cm/archive/namlh31/AI4Code/codebridgedata/exps/codet5-large-python4M-fp16-1epoch/models/ \
    --batch_size 1 \
    --temperature 0.2 \
    --top_p 0.95 \
    --top_k 40 \
    --do_sample \
    --num_return_sequences 20 \
    --num_gen_iterations 10 \
    --padding_side right \
    --max_output_length 400 \
    --max_input_length 400 \
    --num_proc 50 \
    --input_column docstring \
    --lora_config_path /home/namlh31/project/AI4Code/TheVault_exp/configs/lora_enc_dec_config.yaml \
    --dataset_name_or_path /home/namlh31/project/AI4Code/testgen/data/human-eval/HumanEval.jsonl \
    --output_dir /home/namlh31/project/AI4Code/TheVault_exp/results/codet5-large-human-eval-temp0.2 \
    --cache_dir /cm/archive/namlh31/AI4Code/.cache \
    --low_cpu_mem_usage \
    --device_map auto \
