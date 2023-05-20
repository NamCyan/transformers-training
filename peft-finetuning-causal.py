from datasets import load_dataset, concatenate_datasets
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.trainer_utils import get_last_checkpoint
from custom_data_collator import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import os
import transformers, datasets
import logging
import sys
import argparse
import evaluate
import random
from dataclasses import dataclass, field
from typing import Optional
from training_utils import parse_config, print_summary
import yaml
import torch

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": (
                "activate 8 bit training"
            )
        },
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={"help": "Specify the device do you want to load the pretrained models"},
    )
    lora_config_path: Optional[str] = field(
        default=None,
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    input_column: Optional[str] = field(default=None, metadata={"help": "The input column training data file (a text file)."})
    output_column: Optional[str] = field(default=None, metadata={"help": "The output column training data file (a text file)."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    max_input_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "max_input_length"
            )
        },
    )
    max_output_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "max_output_length"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    label_pad_token_id: Optional[int] = field(
        default=-100,
        metadata={"help": "Token id to ignore in calculating loss"},
    )
    ignore_input_token_label: bool = field(
        default=True, metadata={"help": "ignore loss for the input sequence"}
    )
    prefix_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Prefix prompt for input"},
    )
    postfix_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Postfix prompt for input"},
    )
    padding_side: Optional[str] = field(
        default="left",
        metadata={"help": "Tokenizer padding side"},
    )
    new_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Add new tokens to tokenizer. NOTE: separate tokens with \";\" symbol"},
    )


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name_or_path', type=str)
#     parser.add_argument('--dataset_name_or_path', type=str, default= None)
#     parser.add_argument('--train_file', type=str, default= None)
#     parser.add_argument('--valid_file', type=str, default= None)
#     parser.add_argument('--output_dir', type=str)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--device', type=str, default="auto", choices= ["auto", "balanced"])

#     parser.add_argument('--input_column', type=str)
#     parser.add_argument('--output_column', type=str)
#     parser.add_argument('--max_input_length', type=int, default=512)
#     parser.add_argument('--max_output_length', type=int, default=512)
#     parser.add_argument('--prefix_prompt', type=str, default=None)
#     parser.add_argument('--postfix_prompt', type=str, default=None)
#     parser.add_argument('--ignore_input_token_label', action='store_true')
#     parser.add_argument('--padding_side', type=str, default='left')
#     parser.add_argument('--lora_config_path', type=str, default= None)
#     parser.add_argument('--label_pad_token_id', type=int, default=-100)

#     parser.add_argument('--learning_rate', type=float, default=1e-3)
#     parser.add_argument('--num_train_epochs', type=int)
#     parser.add_argument('--logging_steps', type=int, default= 500)
#     parser.add_argument('--logging_strategy', type=str, default= 'steps')
#     parser.add_argument('--evaluation_strategy', type=str, default= 'steps')
#     parser.add_argument('--eval_steps', type=int, default= 500)
#     parser.add_argument('--save_steps', type=int, default=500)
#     parser.add_argument('--metric_for_best_model', type=str, default= None)
#     parser.add_argument('--load_best_model_at_end', action='store_true')
#     parser.add_argument('--load_in_8bit', action='store_true')
#     parser.add_argument('--low_cpu_mem_usage', action='store_true')
#     parser.add_argument('--fp16', action='store_true')
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
#     parser.add_argument('--num_proc', type=int, default=10)
#     parser.add_argument('--save_total_limit', type=int, default=5)
#     return parser.parse_args()

# def preprocess_function(examples):
#     batch_size = len(examples[data_args.input_column])
#     prefix_prompt_tokens , postfix_prompt_tokens = [], []
#     if data_args.prefix_prompt is not None:
#         prefix_prompt_tokens = tokenizer.encode(data_args.prefix_prompt, add_special_tokens= False)
#     if data_args.postfix_prompt is not None:
#         postfix_prompt_tokens = tokenizer.encode(data_args.postfix_prompt, add_special_tokens= False)

#     inputs = [item.strip() for item in examples[data_args.input_column]]
#     targets = [item.strip() for item in examples[data_args.output_column]]

#     model_inputs = tokenizer(inputs, truncation=True, max_length= data_args.max_input_length - len(prefix_prompt_tokens) - len(postfix_prompt_tokens), add_special_tokens= False)
#     labels = tokenizer(targets, truncation=True, max_length= data_args.max_output_length - 1, add_special_tokens= False) # a place 
#     for i in range(batch_size):
#         sample_input_ids = prefix_prompt_tokens + model_inputs["input_ids"][i] + postfix_prompt_tokens
#         label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
#         model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
#         if data_args.ignore_input_token_label:
#             labels["input_ids"][i] = [data_args.label_pad_token_id] * len(sample_input_ids) + label_input_ids
#         else:
#             labels["input_ids"][i] = sample_input_ids + label_input_ids
#         model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
#         assert len(model_inputs["input_ids"][i]) <= (data_args.max_input_length + data_args.max_output_length)

#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Load tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.padding_side = data_args.padding_side

    if data_args.new_tokens is not None:
        new_token_list = data_args.new_tokens.split(";")
        num_added_toks = tokenizer.add_tokens(new_token_list, special_tokens=True)
        logger.info(f"Add {num_added_toks} new tokens to the Tokenizer: {new_token_list}")

    if tokenizer.pad_token_id is None:
        logger.info("Tokenizer does not has pad token. Set the pad_token to eos_token.")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if training_args.metric_for_best_model is not None:
        metric = evaluate.load("accuracy")

    if data_args.dataset_name_or_path is None and data_args.train_file is None:
        raise "Both dataset_name_or_path and train_file can not be None"


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        # ignore prediction for pad token
        
        true_label = []
        true_pred = []
        for i, label in enumerate(labels):
            pred = preds[i].argmax(-1)
            attend_predict = np.array(label) != data_args.label_pad_token_id
            true_label += list(label[attend_predict])
            true_pred += list(pred[attend_predict])

        assert len(true_pred) == len(true_label)
        return metric.compute(predictions=true_pred, references=true_label)

    # Load data
    def preprocess_function(examples):
        batch_size = len(examples[data_args.input_column])
        prefix_prompt_tokens , postfix_prompt_tokens = [], []
        if data_args.prefix_prompt is not None:
            prefix_prompt_tokens = tokenizer.encode(data_args.prefix_prompt, add_special_tokens= False)
        if data_args.postfix_prompt is not None:
            postfix_prompt_tokens = tokenizer.encode(data_args.postfix_prompt, add_special_tokens= False)

        inputs = [item.strip() for item in examples[data_args.input_column]]
        targets = [item.strip() for item in examples[data_args.output_column]]

        model_inputs = tokenizer(inputs, truncation=True, max_length= data_args.max_input_length - len(prefix_prompt_tokens) - len(postfix_prompt_tokens), add_special_tokens= False)
        labels = tokenizer(targets, truncation=True, max_length= data_args.max_output_length - 1, add_special_tokens= False)
        for i in range(batch_size):
            sample_input_ids = prefix_prompt_tokens + model_inputs["input_ids"][i] + postfix_prompt_tokens
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            if data_args.ignore_input_token_label:
                labels["input_ids"][i] = [data_args.label_pad_token_id] * len(sample_input_ids) + label_input_ids
            else:
                labels["input_ids"][i] = sample_input_ids + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
            assert len(model_inputs["input_ids"][i]) <= (data_args.max_input_length + data_args.max_output_length)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if data_args.dataset_name_or_path is not None:
        dataset = load_dataset(data_args.dataset_name_or_path, cache_dir= model_args.cache_dir, num_proc= data_args.num_proc)
    else:
        data_files = {'train': data_args.train_file, 'validation': data_args.validation_file}
        dataset = load_dataset('json', data_files= data_files, num_proc= data_args.num_proc, cache_dir= model_args.cache_dir)

    # Load model
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                                load_in_8bit=model_args.load_in_8bit, 
                                                low_cpu_mem_usage= model_args.low_cpu_mem_usage, 
                                                device_map= model_args.device_map, 
                                                torch_dtype=torch_dtype,
                                                config= config,
                                                cache_dir= model_args.cache_dir,
                                                resume_download=True)
    # Set use_cache to False to use gradient checkpointing
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    # resize embedding in case of adding new tokens
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Define LoRA Config
    if model_args.lora_config_path is not None:
        _lora_config = yaml.load(open(model_args.lora_config_path), Loader = yaml.FullLoader)
        _lora_config = parse_config(_lora_config)
        lora_config = LoraConfig(
            r=_lora_config.r,
            target_modules=_lora_config.target_modules,
            lora_alpha=_lora_config.alpha,
            lora_dropout=_lora_config.dropout,
            bias=_lora_config.bias,
            task_type=TaskType.CAUSAL_LM
        )

        # overcome gradient checkpointing error
        # see https://github.com/huggingface/transformers/issues/23170
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # prepare int-8 model for training
        if model_args.load_in_8bit:
            model = prepare_model_for_int8_training(model)

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()


    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        max_length = data_args.max_input_length + data_args.max_output_length,
        padding= True,
        label_pad_token_id=data_args.label_pad_token_id,
        pad_to_multiple_of=8
    )

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.num_proc,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

    print("="*200)
    sample_id = random.randint(0, len(tokenized_dataset['train']))
    logger.info("A sample in train set at index {}: {}".format(sample_id, tokenized_dataset['train'][sample_id]))
    logger.info("Input: \n{}".format(tokenizer.decode(tokenized_dataset['train'][sample_id]['input_ids'])))
    logger.info("Label: \n{}".format(tokenizer.decode(np.array(tokenized_dataset['train'][sample_id]['labels'])[np.array(tokenized_dataset['train'][sample_id]['labels']) != data_args.label_pad_token_id])))
    print("="*200)

    #test_collator
    # from torch.utils.data import DataLoader
    # loader_collate = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=5, collate_fn=data_collator)
    # for batch in loader_collate:
    #     print(tokenizer.decode(batch['input_ids'][0]))
    #     print(batch['attention_mask'][0])
    #     print(tokenizer.decode(batch['labels'][0][batch['labels'][0] != data_args.label_pad_token_id]))
    #     break
    # exit()

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics= compute_metrics if training_args.metric_for_best_model is not None else None
    )
    # model.config.use_cache = False # silence the warnings. Please re-enable for inference!

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # summary gpu usage
    print_summary(train_result)

    metrics = train_result.metrics

    metrics["train_samples"] = len(tokenized_dataset["train"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    peft_model_id= f"{training_args.output_dir}/models"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

if __name__ == "__main__":
    # args = get_args()

    main()

