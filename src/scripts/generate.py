"""
@File    :   generate.py
@Time    :   2025/11/17 16:34:34
@Author  :   Lin
@Version :   1.0
@Desc    :   None
copyright USTC
"""

import json
import logging
import os
import random

import hydra
from datasets import Dataset
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase, pipeline
from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)
r_prompt = "Surround the final answer with boxed{}"


def generate_vllm(cfg: DictConfig):
    log.info(f"cfg {cfg}")
    llm = LLM(
        model=cfg["models"]["root"] + cfg["models"]["model"],
        enable_prefix_caching=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        cfg["models"]["root"] + cfg["models"]["model"]
    )

    train_data = []
    prompt_data = []

    with open(file=cfg["prompts"], mode="r") as f:
        for qa in json.load(f):
            prompt_data.append(
                {"role": "user", "content": f"Question: {qa['question']}\nSolution:"}
            )
            prompt_data.append({"role": "assistant", "content": qa["solution"]})

    log.info("start qa preprocess")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(file=cfg["dataset"]["path"], mode="r") as f:
        for line in f.readlines():
            for _ in range(cfg["num_diverse_path"]):
                sample = json.loads(line)
                indices = list(range(len(prompt_data) // 2))
                random.shuffle(indices)
                prompt_data_shuffle = []
                for i in indices:
                    prompt_data_shuffle.extend(prompt_data[2 * i : 2 * i + 2])
                conversation = prompt_data_shuffle + [
                    {
                        "role": "user",
                        "content": f"Question: {sample['problem']}.{r_prompt}\nSolution:",
                    }
                ]
                sample["prompts"] = tokenizer.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                train_data.append(sample)
    dataset = Dataset.from_list(train_data)
    log.info("finish qa preprocess")

    sampling_params = SamplingParams(
        max_tokens=cfg["generate"]["max_tokens"],
        temperature=cfg["generate"]["temperature"],
        top_k=cfg["generate"]["top_k"],
        top_p=cfg["generate"]["top_p"],
    )
    log.info("start generation")

    chunk_size = cfg["chunk_size"] * cfg["num_diverse_path"]
    for chunk_id in range((chunk_size + len(dataset) - 1) // chunk_size):
        log.info(
            f"chunk id: {chunk_id} / {(chunk_size + len(dataset) - 1) // chunk_size}"
        )
        start = chunk_id * chunk_size
        end = min(start + chunk_size, len(dataset))

        chunk_dataset = dataset.select(indices=range(start, end))
        outputs = llm.generate(
            chunk_dataset["prompts"], sampling_params=sampling_params
        )
        cnts = []
        texts = []
        for output, sample in zip(outputs, chunk_dataset):
            cnt = len(output.outputs[0].token_ids) - 1
            text = output.outputs[0].text
            cnts.append(cnt)
            texts.append(text)
        chunk_dataset: Dataset = chunk_dataset.add_column("cnt", cnts)
        chunk_dataset = chunk_dataset.add_column("outputs", texts)
        chunk_dataset.to_json(f"{cfg['output_dir']}chunk_{chunk_id}.jsonl")
        if chunk_id > 2:
            break
    log.info("finish generation")


def generate_pipeline(cfg: DictConfig):
    log.info(f"cfg {cfg}")
    llm = pipeline(
        task="text-generation", model=cfg["models"]["root"] + cfg["models"]["model"]
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        cfg["models"]["root"] + cfg["models"]["model"]
    )

    train_data = []
    prompt_data = []

    with open(file=cfg["prompts"], mode="r") as f:
        for qa in json.load(f):
            prompt_data.append(
                {"role": "user", "content": f"Question: {qa['question']}\nSolution:"}
            )
            prompt_data.append({"role": "assistant", "content": qa["solution"]})

    log.info("start qa preprocess")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(file=cfg["dataset"]["path"], mode="r") as f:
        for line in f.readlines():
            for _ in range(cfg["num_diverse_path"]):
                sample = json.loads(line)
                indices = list(range(len(prompt_data) // 2))
                random.shuffle(indices)
                prompt_data_shuffle = []
                for i in indices:
                    prompt_data_shuffle.extend(prompt_data[2 * i : 2 * i + 2])
                conversation = prompt_data_shuffle + [
                    {
                        "role": "user",
                        "content": f"Question: {sample['problem']}\nSolution:",
                    }
                ]
                sample["prompts"] = tokenizer.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                train_data.append(sample)
    dataset = Dataset.from_list(train_data)
    log.info("finish qa preprocess")

    generation_params = {
        "max_new_tokens": cfg["generate"]["max_tokens"],
        "do_sample": False,
        "temperature": cfg["generate"]["temperature"],
        "top_k": cfg["generate"]["top_k"],
        "top_p": cfg["generate"]["top_p"],
        "num_return_sequences": 1,
        "return_full_text": False,
    }
    log.info("start generation")

    chunk_size = cfg["chunk_size"] * cfg["num_diverse_path"]
    for chunk_id in range((chunk_size + len(dataset) - 1) // chunk_size):
        log.info(
            f"chunk id: {chunk_id} / {(chunk_size + len(dataset) - 1) // chunk_size}"
        )
        start = chunk_id * chunk_size
        end = min(start + chunk_size, len(dataset))

        chunk_dataset = dataset.select(indices=range(start, end))
        outputs = []
        cnts = []
        # inputs = [item["prompts"] for item in chunk_dataset]

        # responses = llm(inputs, batch_size=4, **generation_params)
        # for item in responses:
        #     output = item[0]["generation_text"]
        #     cnts.append(len(tokenizer.encode(output, add_special_tokens=False)))
        #     outputs.append(output)
        for item in tqdm(chunk_dataset, desc="Generate"):
            output = llm(item["prompts"], **generation_params)[0]["generated_text"]
            cnts.append(len(tokenizer.encode(output, add_special_tokens=False)))
            outputs.append(output)

        chunk_dataset: Dataset = chunk_dataset.add_column("cnt", cnts)
        chunk_dataset = chunk_dataset.add_column("outputs", outputs)
        chunk_dataset.to_json(f"{cfg['output_dir']}chunk_{chunk_id}.jsonl")
        if chunk_id > 2:
            break
    log.info("finish generation")


@hydra.main(config_path="../../config", config_name="generate", version_base="1.3.1")
def generate(cfg: DictConfig):
    if cfg.use_vllm:
        generate_vllm(cfg)
    else:
        generate_pipeline(cfg)


generate()
