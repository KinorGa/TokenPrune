"""
@File    :   generate.py
@Time    :   2025/11/17 16:34:34
@Author  :   Lin
@Version :   1.0
@Desc    :   None
copyright USTC
"""

import json
import random

import hydra
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams


@hydra.main(config_path="../../config", config_name="generate", version_base="1.3.1")
def generate(cfg: DictConfig):
    print(cfg)
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

    with open(file=cfg["dataset"]["path"], mode="r") as f:
        for line in f.readlines():
            for _ in range(cfg["generate"]["num_diverse_path"]):
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
    train_data = train_data[:160]
    dataset = Dataset.from_list(train_data)

    sampling_params = SamplingParams(
        max_tokens=cfg["generate"]["max_tokens"],
        temperature=cfg["generate"]["temperature"],
        top_k=cfg["generate"]["top_k"],
        top_p=cfg["generate"]["top_p"],
    )
    outputs = llm.generate(dataset["prompts"], sampling_params=sampling_params)
    for output, sample in zip(outputs, train_data):
        cnt = len(output.outputs[0].token_ids) - 1
        text = output.outputs[0].text
        sample["count"] = cnt
        sample["output"] = text
    with open(file="outputs.json", mode="w") as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(file="outputs.jsonl", mode="w") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


generate()
