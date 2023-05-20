from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from peft import PeftModel, PeftConfig
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, GenerationConfig
import random
import numpy as np
import jsonlines

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
    output_dir: Optional[str] = field(
        default=None, metadata={"help": "Output dir to save results."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    input_column: Optional[str] = field(default=None, metadata={"help": "The input column training data file (a text file)."})
    output_column: Optional[str] = field(default=None, metadata={"help": "The output column training data file (a text file)."})

    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
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


@dataclass
class GeneratorArguments:
    batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size."},
    )
    temperature: Optional[float] = field(
        default=1.0,
    )
    top_p: Optional[float] = field(
        default=1.0,
    )
    top_k: Optional[int] = field(
        default=0,
    )
    do_sample: bool = field(
        default=True,
    )
    num_return_sequences: Optional[int] = field(
        default=32,
        metadata={"help": "Number of sequence to generate"},
    ) 
    num_gen_iterations: Optional[int] = field(
        default=1,
    ) 



parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GeneratorArguments))
model_args, data_args, gen_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = data_args.padding_side

def preprocess_function(examples):
    batch_size = len(examples[data_args.input_column])
    prefix_prompt_tokens , postfix_prompt_tokens = [], []
    if data_args.prefix_prompt is not None:
        prefix_prompt_tokens = tokenizer.encode(data_args.prefix_prompt, add_special_tokens= False)
    if data_args.postfix_prompt is not None:
        postfix_prompt_tokens = tokenizer.encode(data_args.postfix_prompt, add_special_tokens= False)

    inputs = [item.strip() for item in examples[data_args.input_column]]
    

    # tokenize 
    num_special_tokens = 2 if tokenizer.bos_token_id else 1
    model_inputs = tokenizer(inputs, max_length=data_args.max_input_length - len(prefix_prompt_tokens) - len(postfix_prompt_tokens) - num_special_tokens, truncation=True, add_special_tokens= False)

    for i in range(batch_size):
        sample_input_ids =  prefix_prompt_tokens + model_inputs["input_ids"][i] + postfix_prompt_tokens + [tokenizer.eos_token_id]
        sample_input_ids = [tokenizer.bos_token_id] + sample_input_ids if tokenizer.bos_token_id else sample_input_ids
        model_inputs["input_ids"][i] = sample_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        
    # Tokenize targets with the `text_target` keyword argument
    if data_args.output_column:
        targets = [item.strip() for item in examples[data_args.output_column]]
        labels = tokenizer(targets, max_length=data_args.max_output_length, truncation=True)

        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else data_args.label_pad_token_id) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if "humaneval" in data_args.dataset_name_or_path.lower():
    dataset = load_dataset("custom_datasets/human-eval.py", datapath = data_args.dataset_name_or_path)

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=data_args.num_proc,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)


sample_id = random.randint(0, len(tokenized_dataset['train']))
print("A sample", tokenized_dataset['train'][sample_id])
print("Input: \n", tokenizer.decode(tokenized_dataset['train'][sample_id]['input_ids']))
print("="*200)

data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=data_args.label_pad_token_id,
        pad_to_multiple_of=8
    )

test_dataset = tokenized_dataset["train"]
test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=gen_args.batch_size, pin_memory=True, shuffle= False)

config = PeftConfig.from_pretrained(model_args.model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, 
                                              device_map=model_args.device_map,
                                              load_in_8bit=model_args.load_in_8bit, 
                                              low_cpu_mem_usage= model_args.low_cpu_mem_usage
                                             )
if model_args.lora_config_path is not None:
    model = PeftModel.from_pretrained(model, 
                                      model_args.model_name_or_path, 
                                      device_map=model_args.device_map,
                                      load_in_8bit=model_args.load_in_8bit, 
                                      low_cpu_mem_usage= model_args.low_cpu_mem_usage 
                                     )



generation_config = GenerationConfig(
                        temperature=gen_args.temperature,
                        top_p=gen_args.top_p,
                        top_k=gen_args.top_k,
                        do_sample=gen_args.do_sample,
                        num_return_sequences=gen_args.num_return_sequences,
                    )

if not os.path.exists(data_args.output_dir):
    os.mkdir(data_args.output_dir)
task_ids = dataset['train']['task_id']

model.eval()
# eval_preds = []
print("Number of iters:", gen_args.num_gen_iterations)


for iter in range(gen_args.num_gen_iterations):
    for batch_id, batch in enumerate(tqdm(test_dataloader, desc=f"Iteration {iter}:")):
        bs = len(batch['input_ids'])
        batch_task_ids = task_ids[batch_id * gen_args.batch_size: batch_id * gen_args.batch_size + bs]
        batch = {k: v.cuda() for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            outputs = model.generate(**batch, 
                                    generation_config=generation_config,
                                    max_new_tokens=data_args.max_output_length)
        preds = outputs.detach().cpu().numpy()
        generated_outputs = tokenizer.batch_decode(preds, skip_special_tokens=True)
        for i in range(bs):
            with jsonlines.open(os.path.join(data_args.output_dir, '{}.jsonl'.format(batch_task_ids[i].replace("/", "_"))), mode='a') as writer:
                writer.write_all([{"generation": gen_out} for gen_out in generated_outputs[i * gen_args.num_return_sequences: (i + 1) *gen_args.num_return_sequences]])
        # break
        
# print(eval_preds[0])