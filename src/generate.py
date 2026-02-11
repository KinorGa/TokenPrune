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
from copy import deepcopy

import hydra
import numpy as np
from datasets import Dataset
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

from src.build_prompts import format_question_with_prompt
from src.math_parser import compare_answers, extract_math_answer

log = logging.getLogger(__name__)


def generate_vllm(cfg: DictConfig):
    llm = LLM(
        model=cfg["models"]["path"] + cfg["models"]["name"],
        enable_prefix_caching=True,
        dtype=cfg["models"].get("dtype", "bfloat16"),
        tensor_parallel_size=1,
        max_model_len=3072,
    )

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        cfg["models"]["path"] + cfg["models"]["name"],
        trust_remote_code=True,
    )

    # load dataset
    dataset = Dataset.from_json(cfg["dataset"])

    # load external prompts
    all_prompts = []
    with open(cfg["prompts"], mode="r") as f:
        for qa in json.load(f):
            all_prompts.append(
                {
                    "role": "user",
                    "content": format_question_with_prompt(qa["question"], prompt_id=1),
                }
            )
            all_prompts.append({"role": "assistant", "content": qa["solution"]})

    def dataset_fn(example: dict, num_few_shot: int = 0):
        example_prompts = []
        # indices = list(range(len(all_prompts) // 2))
        # random.shuffle(indices)
        # np.random.shuffle(indices)
        for i in range(num_few_shot):
            # idx = indices[i]
            idx = example["prompt_indices"][i]
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
            "prompt_indices": example["prompt_indices"]
            if num_few_shot > 0 and "prompt_indices" in example
            else [],
        }

    # for chunk generation
    chunk_size = cfg["chunk_size"]
    num_diverse_path = cfg["num_diverse_path"]
    sampling_paras = SamplingParams(
        max_tokens=cfg["sampler"]["max_tokens"],
        temperature=cfg["sampler"]["temperature"],
        top_k=cfg["sampler"]["top_k"],
        top_p=cfg["sampler"]["top_p"],
    )
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        chunk_dataset = dataset.select(range(start, end))
        chunk_dataset = chunk_dataset.map(
            dataset_fn, num_proc=4, fn_kwargs={"num_few_shot": 0}
        )
        print(
            f"Processing chunk from {start} to {end}, total {len(chunk_dataset)} samples."
        )

        chunk_id = start // chunk_size

        # zero shot sample num diverse path
        zero_shot_outputs_list = [[] for _ in range(end - start)]
        zero_shot_true_or_false = [False for _ in range(end - start)]
        zero_tqdm = tqdm(range(num_diverse_path))
        num_correct = 0
        for zi in zero_tqdm:
            zero_tqdm.display(
                f"[ZERO SHOT][num diverse path {zi}/{num_diverse_path}][dataset {start}/{len(dataset)}][accuracy: {num_correct / len(chunk_dataset):.4f}][zero shot correct {sum(zero_shot_true_or_false)}/{len(zero_shot_true_or_false)}]"
            )
            log.info(
                f"[ZERO SHOT][num diverse path {zi}/{num_diverse_path}][dataset {start}/{len(dataset)}][accuracy: {num_correct / len(chunk_dataset):.4f}][zero shot correct {sum(zero_shot_true_or_false)}/{len(zero_shot_true_or_false)}]"
            )
            outputs = llm.generate(
                prompts=chunk_dataset["prompts"],
                sampling_params=sampling_paras,
            )
            num_correct = 0

            for i, (output, sample) in enumerate(zip(outputs, chunk_dataset)):
                problem = sample["problem"]
                label_text = sample["solution"]
                pred_text = output.outputs[0].text
                label_answer = extract_math_answer(problem, label_text)
                pred_answer = extract_math_answer(problem, pred_text)

                is_correct = compare_answers(problem, label_answer, pred_answer)
                item = deepcopy(sample)
                item.update(
                    {
                        "outputs": pred_text,
                        "is_correct": is_correct,
                        "true_answer": label_answer,
                        "answer": pred_answer,
                        "token_length": len(output.outputs[0].token_ids) - 1,
                        "type": "zero_shot",
                        "idx": i + start,
                    }
                )
                zero_shot_outputs_list[i].append(item)
                num_correct += int(is_correct)
                zero_shot_true_or_false[i] = zero_shot_true_or_false[i] or is_correct

        # few shot sample num diverse path
        few_shot_outputs_list = [[] for _ in range(end - start)]
        few_shot_true_or_false = [False for _ in range(end - start)]
        few_tqdm = tqdm(range(num_diverse_path))
        num_correct = 0
        for fi in few_tqdm:
            few_tqdm.display(
                f"[FEW SHOT][num diverse path {fi}/{num_diverse_path}][dataset {start}/{len(dataset)}][accuracy: {num_correct / len(chunk_dataset):.4f}][few_shot correct {sum(few_shot_true_or_false)}/{len(few_shot_true_or_false)}]"
            )
            log.info(
                f"[FEW SHOT][num diverse path {fi}/{num_diverse_path}][dataset {start}/{len(dataset)}][accuracy: {num_correct / len(chunk_dataset):.4f}][few_shot correct {sum(few_shot_true_or_false)}/{len(few_shot_true_or_false)}]"
            )

            # prepare prompts indices
            prompt_indices = np.random.randint(
                0, len(all_prompts) // 2, size=(end - start, cfg["num_few_shot"])
            )
            chunk_dataset = dataset.select(range(start, end))
            chunk_dataset = chunk_dataset.add_column(
                "prompt_indices", prompt_indices.tolist()
            )
            chunk_dataset = chunk_dataset.map(
                dataset_fn,
                num_proc=4,
                fn_kwargs={"num_few_shot": cfg["num_few_shot"]},
            )
            outputs = llm.generate(
                prompts=chunk_dataset["prompts"],
                sampling_params=sampling_paras,
            )
            num_correct = 0

            for i, (output, sample) in enumerate(zip(outputs, chunk_dataset)):
                problem = sample["problem"]
                label_text = sample["solution"]
                pred_text = output.outputs[0].text
                label_answer = extract_math_answer(problem, label_text)
                pred_answer = extract_math_answer(problem, pred_text)

                is_correct = compare_answers(problem, label_answer, pred_answer)
                item = deepcopy(sample)
                item.update(
                    {
                        "outputs": pred_text,
                        "is_correct": is_correct,
                        "true_answer": label_answer,
                        "answer": pred_answer,
                        "token_length": len(output.outputs[0].token_ids) - 1,
                        "type": "few_shot",
                        "idx": i + start,
                    }
                )
                few_shot_outputs_list[i].append(item)
                num_correct += int(is_correct)
                few_shot_true_or_false[i] = few_shot_true_or_false[i] or is_correct
        log.info(
            f"zero shot or few shot correct {sum(zero_flag or few_flag for zero_flag, few_flag in zip(zero_shot_true_or_false, few_shot_true_or_false))}/{len(chunk_dataset)}"
        )

        # save chunk results
        chunk_dataset = dataset.select(range(start, end))
        chunk_items = []
        for i in range(len(chunk_dataset)):
            for zero_sample in zero_shot_outputs_list[i]:
                item = dict()
                item.update(chunk_dataset[i])
                item.update(zero_sample)
                chunk_items.append(item)
            for few_sample in few_shot_outputs_list[i]:
                item = dict()
                item.update(chunk_dataset[i])
                item.update(few_sample)
                chunk_items.append(item)

        os.makedirs(f"{cfg['output_dir']}/{cfg['models']['name']}", exist_ok=True)
        with open(
            f"{cfg['output_dir']}/{cfg['models']['name']}/generation_chunk_{chunk_id}.json",
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(chunk_items, f, ensure_ascii=False, indent=4)
        log.info(
            f"Saved generation results of chunk {chunk_id} with {len(chunk_items)} samples to {cfg['output_dir']}/{cfg['models']['name']}/generation_chunk_{chunk_id}.json"
        )


@hydra.main(config_path="../config", config_name="generate", version_base="1.3.1")
def generate(cfg: DictConfig):
    log.info(f"Generation config: {cfg}")
    if cfg.use_vllm:
        generate_vllm(cfg)
    else:
        raise NotImplementedError("Only vLLM generation is implemented.")


generate()
