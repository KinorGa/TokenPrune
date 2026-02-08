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

import hydra
from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.math_parser import compare_answers, extract_math_answer

log = logging.getLogger(__name__)
format_prompt = (
    "Please deduce the solution step by step and enclose the final answer in \\boxed{}."
)


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
        max_model_len=4096,
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
            user_content = f"Question: {qa['question']}\nSolution: "
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
                "content": f"Question: {example['problem']}\nSolution: {format_prompt}\n",
            }
        )
        example_prompts = tokenizer.apply_chat_template(
            example_prompts, add_generation_prompt=True, tokenize=False
        )
        return {
            "prompts": example_prompts,
            "problem": example["problem"],
            "solution": example["solution"],
        }

    dataset = dataset.map(add_prompt)
    # dataset = dataset.select(range(5))

    log.info("Starting evaluation...")
    sampling_params = SamplingParams(
        max_tokens=cfg["sampler"].get("max_new_tokens", 512),
        temperature=cfg["sampler"].get("temperature", 0.1),
        top_p=cfg["sampler"].get("top_p", 0.9),
        top_k=cfg["sampler"].get("top_k", 40),
        n=cfg["sampler"].get("n_response", 1),
    )
    outputs = llm.generate(
        prompts=dataset["prompts"],
        sampling_params=sampling_params,
        lora_request=LoRARequest("lora", 1, lora_path)
        if cfg.get("use_lora", False)
        else None,
    )

    # compute evaluation metrics
    num_correct = 0
    contents = []
    is_corrects = []
    true_answers = []
    answers = []
    # check multi reponse
    for output, sample in zip(outputs, dataset):
        problem = sample["problem"]
        label_text = sample["solution"]
        label_answer = extract_math_answer(problem, label_text)
        pred_text = None
        pred_answer = None
        is_correct = None

        for out in output.outputs:
            pred_text = out.text
            pred_answer = extract_math_answer(problem, pred_text)

            is_correct = compare_answers(problem, pred_answer, label_answer)
            if is_correct:
                break

        contents.append(pred_text)
        is_corrects.append(is_correct)
        true_answers.append(str(label_answer))
        answers.append(str(pred_answer))
        num_correct += int(is_correct)

    # for output, sample in zip(outputs, dataset):
    #     problem = sample["problem"]
    #     pred_text = output.outputs[0].text
    #     label_text = sample["solution"]
    #     pred_answer = extract_math_answer(problem, pred_text)
    #     label_answer = extract_math_answer(problem, label_text)

    #     is_correct = compare_answers(problem, pred_answer, label_answer)
    #     num_correct += int(is_correct)
    #     contents.append(pred_text)
    #     is_corrects.append(is_correct)
    #     true_answers.append(str(label_answer))
    #     answers.append(str(pred_answer))

    log.info(
        f"eval accuracy: {num_correct / len(dataset):.4f} ({num_correct}/{len(dataset)})"
    )

    # save eval results
    dataset = dataset.add_column("outputs", contents)
    dataset = dataset.add_column("is_correct", is_corrects)
    dataset = dataset.add_column("true_answer", true_answers)
    dataset = dataset.add_column("answer", answers)
    dataset.to_json(
        cfg["output_dir"] + "/" + cfg["models"]["name"] + ".json",
        force_ascii=False,
        indent=2,
    )

    log.info("Evaluation finished.")


evaluate()
