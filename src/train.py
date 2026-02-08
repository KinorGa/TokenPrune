"""
@File    :   train.py
@Time    :   2025/11/19 11:03:13
@Author  :   Lin
@Version :   1.0
@Desc    :   None
copyright USTC
"""

import logging
from dataclasses import dataclass

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, ListConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

IGNORE_INDEX = -100

log = logging.getLogger(__name__)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    DataCollator for supervised datasets, handling padding and conversion to tensors
    for input_ids and labels. This class ensures that the input data is properly
    formatted for use in model training.

    Attributes:
    -----------
    tokenizer: transformers.PreTrainedModel
        A tokenizer that will be used to convert input data into token IDs and provide
        special tokens such as the padding token.
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, instances: list) -> dict:
        """
        Process a list of instances and return a dictionary containing padded input IDs, labels, and attention masks

        Parameters:
        -----------
        instances: list
            A list of dictionaries where each dictionary contains input_ids and labels.

        Returns:
        --------
        dict
            A dictionary with the following keys:
            - input_ids: Tensor of padded input token IDs.
            - labels: Tensor of padded labels corresponding to the input data.
            - attention_mask: Tensor indicating which tokens should be attended to (non-padding tokens).
        """
        input_ids, labels = [], []
        for instance in instances:
            input_ids.append(torch.tensor(instance["input_ids"]))
            labels.append(torch.tensor(instance["labels"]))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@hydra.main(config_path="../config", config_name="train", version_base="1.3.1")
def train(cfg: DictConfig):
    log.info("Training started with configuration:")
    log.info(cfg)

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg["models"]["path"] + cfg["models"]["name"],
        dtype=cfg["models"].get("dtype", "bfloat16"),
        attn_implementation="flash_attention_2",
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg["models"]["path"] + cfg["models"]["name"],
        trust_remote_code=True,
    )

    # use lora
    if cfg.get("use_peft", False):
        print("Using PEFT with LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=cfg["peft"].get("inference_mode", False),
            r=cfg["peft"].get("r", 8),
            lora_alpha=cfg["peft"].get("lora_alpha", 16),
            lora_dropout=cfg["peft"].get("lora_dropout", 0.1),
            target_modules=cfg["peft"].get("target_modules", ["q_proj", "v_proj"]),
            bias=cfg["peft"].get("bias", "none"),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        if isinstance(peft_config.target_modules, ListConfig):
            peft_config.target_modules = list(peft_config.target_modules)

    def tokenize(example: dict):
        # Create input part with BOS
        if "qwen" in cfg["models"]["name"].lower():
            input_text = example["prompts"]
        else:
            input_text = tokenizer.bos_token + example["prompts"]

        # Tokenize the input to get its length
        input_tokenized = tokenizer(text=input_text, add_special_tokens=False)
        input_length = len(input_tokenized["input_ids"])

        # Create full text by adding label and EOS token
        full_text = input_text + " " + example["outputs"] + tokenizer.eos_token

        # Tokenize the full text
        full_tokenized = tokenizer(text=full_text, add_special_tokens=False)

        # Create labels with IGNORE_INDEX for input part
        labels = full_tokenized["input_ids"][:]
        labels[:input_length] = [-100] * input_length

        full_tokenized["labels"] = labels
        full_tokenized["prompt"] = input_text
        return full_tokenized

    # load train dataset
    train_dataset: Dataset = load_dataset(
        "json",
        data_files=f"{cfg['dataset']}/generate/{cfg['models']['name']}/*.jsonl",
    )["train"]
    train_dataset = train_dataset.map(
        tokenize, batched=False, num_proc=4, remove_columns=train_dataset.column_names
    )

    train_dataset = train_dataset.shuffle(seed=cfg["seed"]).select(range(150))

    # config training arguments
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"] + "/" + cfg["models"]["name"],
        logging_dir=cfg["output_dir"] + "/" + cfg["models"]["name"] + "/logs",
        num_train_epochs=cfg["num_train_epochs"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        save_strategy=cfg["save_strategy"],
        logging_strategy=cfg["logging_strategy"],
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        bf16=cfg.get("bf16", True),
        fp16=cfg.get("fp16", False),
        report_to=cfg["report_to"],
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # start train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    log.info("Training completed successfully.")


train()
