"""
@File    :   train.py
@Time    :   2025/11/19 11:03:13
@Author  :   Lin
@Version :   1.0
@Desc    :   None
copyright USTC
"""

import logging

import hydra
from datasets import Dataset
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)

"""
supervise fine tuning 
data struct (input_ids, labels, prompt)
input_ids: Index(problem + ' ' + solution)
labels: mask the part of the problem of the input_ids
prompt: str: problem
"""

log = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="train", version_base="1.3.1")
def train(cfg: DictConfig):
    Dataset.from_json

    attn_implementation = "flash_attention_2"
    if "gemma" in cfg.models.model:
        attn_implementation = "eager"
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.models.model,
        dtype="bfloat16",
        attn_implementation=attn_implementation,
    )

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(cfg.models.model)

    def tokenize(example: dict):
        # Creat input part with BOS
        if "qwen" in cfg.model_name.lower():
            input_text = example["input"]
        else:
            input_text = tokenizer.bos_token + example["input"]

        # Tokenize the input portion to get its length
        input_tokenized = tokenizer(input_text, add_special_tokens=False)
        input_length = len(input_tokenized["input_ids"])

        # Create full text by adding label and EOS token
        full_text = input_text + " " + example["label"] + tokenizer.eos_token

        # Tokenize complete text
        model_inputs = tokenizer(full_text, add_special_tokens=False)

        # Create labels with IGNORE_INDEX for input portion
        labels = model_inputs["input_ids"].copy()
        labels[:input_length] = [cfg.IGNORE_INDEX] * input_length

        model_inputs["labels"] = labels
        model_inputs["prompt"] = input_text

        return model_inputs

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
    )

    return
