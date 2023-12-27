import os
import sys
from typing import List

import fire
import torch
import transformers
#import kosy_transformers
from datasets import load_dataset, Dataset

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.nn import functional as F

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer

#os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_stack_exchange_paired(
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        data_dir
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples):
        return {
            "prompt": ["### User:\n" + question + "\n\n### Assitant:\n" for question in samples["instruction"]],
            "chosen": samples["chosen_response"],
            "rejected": samples["rejected_response"],
        }

    return dataset.map(
        return_prompt_and_responses,
        #remove_columns=original_columns,
    )


def train(
    # model/data params
    base_model: str = "", 
    data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    lr_scheduler: str = "cosine",
    warmup_ratio: float = 0.1, 
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    #wandb_project: str = "",
    #wandb_run_name: str = "",
    #wandb_watch: str = "",  # options: false | gradients | all
    #wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",
    # NEFTune params
    noise_alpha: int = 5
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_ratio: {warmup_ratio}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            #f"wandb_project: {wandb_project}\n"
            #f"wandb_run_name: {wandb_run_name}\n"
            #f"wandb_watch: {wandb_watch}\n"
            #f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    from huggingface_hub import login
    login(token='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')
    
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1 # world_size = 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} # auto
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)
    print("############DDP:",ddp) # False


    # Check if parameter passed or if set within environ
    '''
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    '''
    
    #model = LlamaForCausalLM.from_pretrained(
    #    base_model,
    #    load_in_8bit=True, # LoRA
    #    #load_in_4bit=True, # QLoRA
    #    torch_dtype=torch.float16,
    #    device_map=device_map)

    # Original
    #tokenizer = LlamaTokenizer.from_pretrained(base_model)

    # 1. Define policy and reference models
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Q-LoRA
    model = AutoModelForCausalLM.from_pretrained(
        base_model, # location of saved SFT model
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map = device_map
        #load_in_4bit=True,
        #quantization_config=bnb_config
    )
    
    #model_ref = AutoModelForCausalLM.from_pretrained(
    #    base_model,  # same model as the main one
    #    low_cpu_mem_usage=True,
    #    torch_dtype=torch.float16,
    #    load_in_4bit=True,
    #    quantization_config=bnb_config
    #)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(type(model))
    print(model)
    print("length of tokenizer:",len(tokenizer))

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1 2 None")

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "right"

    # 2. Define dataset
    def return_prompt_and_responses(samples):
        
        return {
            "prompt": "### User:\n" + samples["question"] + "\n\n### Assitant:\n",
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }
    dataset = load_dataset(data_path)
    train_dataset = dataset.map(return_prompt_and_responses)
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= cutoff_len
        and len(x["prompt"]) + len(x["rejected"]) <= cutoff_len
    )
    train_dataset = train_dataset["train"].shuffle()
    #print(tokenizer.decode(train_dataset))
    print(train_dataset['prompt'][0])
    print(train_dataset['chosen'][0])
    print(train_dataset['rejected'][0])
    
    # 3. Define hyperparameters
    training_args = TrainingArguments(
        num_train_epochs= num_epochs,
        per_device_train_batch_size=micro_batch_size,
        #per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        #max_steps=1000,
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        #gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=learning_rate,
        #evaluation_strategy="steps",
        #eval_steps=script_args.eval_steps,
        output_dir=output_dir,
        #report_to=script_args.report_to,
        lr_scheduler_type=lr_scheduler,
        warmup_ratio=warmup_ratio,
        optim='paged_adamw_32bit', # rmsprop
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_kyujin",
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model = None, #model_ref,
        args=training_args,
        beta=0.1, # fix
        train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # train
    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)

    # save
    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)

if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)
