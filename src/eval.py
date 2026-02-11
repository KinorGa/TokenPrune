"""
@File    :   eval.py
@Time    :   2025/11/16 14:10:49
@Author  :   Lin
@Version :   1.0
@Desc    :   None
copyright USTC
"""

import json
import logging
import random
from copy import deepcopy
from datetime import datetime, timezone

import hydra
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

from src.build_prompts import format_question_with_prompt
from src.math_parser import compare_answers, extract_math_answer

log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="eval", version_base="1.3.1")
def evaluate(cfg: DictConfig):
    log.info("evaluation cfg: ")
    log.info(cfg)

    model_path = cfg["models"]["path"] + "/" + cfg["models"]["name"]
    if cfg.get("use_checkpoint", False):
        lora_path = cfg["checkpoint_path"] + "/" + cfg["models"]["name"] + "/checkpoint"

    llm = LLM(
        model=model_path,
        enable_prefix_caching=True,
        dtype=cfg["models"].get("dtype", "bfloat16"),
        tensor_parallel_size=1,
        enable_lora=cfg.get("use_lora", False),
        max_model_len=2048,
    )
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 加载所有的评测数据
    dataset = Dataset.from_json(cfg["dataset"])

    # 加载prompts
    all_prompts = []
    with open(cfg["prompts"], mode="r") as f:
        for qa in json.load(f):
            user_content = f"## Question\n{qa['question']}\n## Solution\n"
            all_prompts.append({"role": "user", "content": user_content})
            all_prompts.append({"role": "assistant", "content": qa["solution"]})

    # 给每个评测数据添加prompt
    def add_prompt(example: dict):
        example_prompts = []
        num_few_shot = cfg["num_few_shot"]
        indices = list(range(len(all_prompts) // 2))
        random.shuffle(indices)
        for i in range(num_few_shot):
            idx = indices[i]
            example_prompts.append(all_prompts[2 * idx])  # user
            example_prompts.append(all_prompts[2 * idx + 1])  # assistant

        example_prompts.append(
            {
                "role": "user",
                "content": format_question_with_prompt(example["problem"], prompt_id=1),
            }
        )
        example_prompt = tokenizer.apply_chat_template(
            example_prompts, add_generation_prompt=True, tokenize=False
        )

        prompt_length = len(
            tokenizer.apply_chat_template(
                example_prompts, add_generation_prompt=True, tokenize=True
            )
        )
        return {
            "prompts": example_prompt,
            "problem": example["problem"],
            "solution": example["solution"],
            "prompt_length": prompt_length,
        }

    dataset = dataset.map(add_prompt)
    # dataset = dataset.select(range(100))

    log.info("Starting evaluation...")
    sampling_params = SamplingParams(
        max_tokens=cfg["sampler"].get("max_tokens", 512),
        temperature=cfg["sampler"].get("temperature", 0.1),
        top_p=cfg["sampler"].get("top_p", 0.9),
        top_k=cfg["sampler"].get("top_k", -1),
        n=cfg["sampler"].get("n_response", 1) if cfg.get("use_sample_n", False) else 1,
    )

    items: list[list[RequestOutput]] = []

    """
      compute pass@1: if any of the n responses is correct, then it's correct
      compute avg length: the average length of correct responses, if there is no correct response, then it's 0
      save the results in a json file, including the generated response, whether it's correct, the true answer, the predicted answer, and the token length
    """
    # first generate n responses for each sample, then compute the metrics
    # split chunk generation and evaluation to accelerate the evaluation process, especially when n is large
    chunk_size = cfg.get("chunk_size", 100)
    n = cfg["sampler"].get("n_response", 1)
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))

        chunk_dataset = dataset.select(range(start, end))
        chunk_items = [[] for _ in range(end - start)]

        # generate n responses for each sample in the chunk
        for i in range(n):
            outputs: list[RequestOutput]
            outputs = llm.generate(
                prompts=chunk_dataset["prompts"],
                sampling_params=sampling_params,
                lora_request=LoRARequest("lora", 1, lora_path)
                if cfg.get("use_lora", False)
                else None,
            )
            for j, output in enumerate(outputs):
                chunk_items[j].append(output.outputs[0])
            log.info(f"Finished generation [{i + 1}/{n}] for samples [{start}:{end}]")

        items.extend(chunk_items)

    # evaluate the generated responses
    any_corrects = [False for _ in range(len(dataset))]
    all_lengths = [[] for _ in range(len(dataset))]
    all_items = []
    for i in range(len(dataset)):
        sample = dataset[i]
        problem = sample["problem"]
        label_text = sample["solution"]
        label_answer = extract_math_answer(problem, label_text)
        pred_text = None
        pred_answer = None
        is_correct = None
        token_length = 0

        for out in items[i]:
            copy_sample = deepcopy(sample)
            pred_text = out.text
            token_length = len(out.token_ids) - 1
            pred_answer = extract_math_answer(problem, pred_text)

            is_correct = compare_answers(problem, pred_answer, label_answer)
            if is_correct:
                any_corrects[i] = True
                all_lengths[i].append(token_length)
            copy_sample.update(
                {
                    "outputs": pred_text,
                    "is_correct": is_correct,
                    "true_answer": label_answer,
                    "answer": pred_answer,
                    "token_length": token_length,
                    "type": "zero-shot"
                    if cfg["num_few_shot"] == 0
                    else f"{cfg['num_few_shot']}-shot",
                    "idx": i,
                }
            )
            all_items.append(copy_sample)

    # pass@1 acc
    acc = sum(any_corrects) / len(dataset)
    # avg length of correct responses
    correct_lengths = []
    for lengths in all_lengths:
        if lengths:
            correct_lengths.append(sum(lengths) / len(lengths))
    avg_length = sum(correct_lengths) / len(correct_lengths) if correct_lengths else -1
    log.info(f"eval accuracy: {acc:.4f} ({sum(any_corrects)}/{len(dataset)})")
    log.info(f"Avg Length is {avg_length:.2f}")

    # save the results in a json file by timestamp %Y%m%d-%H%M%S
    utc_time = datetime.now(timezone.utc)
    utc_time_str = utc_time.strftime("%Y%m%d-%H%M%S")
    with open(
        file=f"{cfg['output_dir']}/{cfg['models']['name']}/n{cfg['sampler']['n_response']}/{sum(any_corrects)}%{len(dataset)}-{utc_time_str}.json",
        mode="w",
        encoding="utf-8",
    ) as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    log.info("Evaluation finished.")


evaluate()
