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
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

from src.math_parser import compare_answers, extract_math_answer

log = logging.getLogger(__name__)


def generate_vllm(cfg: DictConfig):
    llm = LLM(
        model=cfg["models"]["path"] + cfg["models"]["name"],
        enable_prefix_caching=True,
        dtype=cfg["models"].get("dtype", "bfloat16"),
        tensor_parallel_size=1,
    )

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        cfg["models"]["path"] + cfg["models"]["name"],
        trust_remote_code=True,
    )

    # 先加载所有的提示QA
    # 以一问一答的形式处理
    all_prompts = []
    with open(cfg["prompts"], mode="r") as f:
        for qa in json.load(f):
            all_prompts.append(
                {"role": "user", "content": f"Question: {qa['question']}\nSolution:"}
            )
            all_prompts.append({"role": "assistant", "content": qa["solution"]})

    log.info("start qa preprocess")
    os.makedirs(cfg["output_dir"] + cfg["models"]["name"], exist_ok=True)

    train_data = []
    with open(cfg["dataset"], mode="r") as f:
        for line in f.readlines():
            # 每个问题重复采样多次
            for _ in range(cfg["num_diverse_path"]):
                cur_qa = json.loads(line)

                # 随机选取若干个提示QA作为上下文
                indices = list(range(len(all_prompts) // 2))
                random.shuffle(indices)
                selected_indices = indices[: cfg.num_few_shot]
                prompt_list = []
                for idx in selected_indices:
                    prompt_list.append(all_prompts[2 * idx])
                    prompt_list.append(all_prompts[2 * idx + 1])

                # 将提示QA和当前问题拼接成最终的对话输入
                conversation = prompt_list + [
                    {
                        "role": "user",
                        "content": f"Question: {cur_qa['problem']}\nSolution:",
                    }
                ]

                cur_qa["prompts"] = tokenizer.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )

                train_data.append(cur_qa)

    train_dataset = Dataset.from_list(train_data)
    log.info("finish qa preprocess")

    log.info("start generation")
    sampling_paras = SamplingParams(
        max_tokens=cfg["sampler"]["max_tokens"],
        temperature=cfg["sampler"]["temperature"],
        top_k=cfg["sampler"]["top_k"],
        top_p=cfg["sampler"]["top_p"],
    )

    # 从历史中恢复
    start_chunk_id = 0
    if cfg["resume_from_chunk"]:
        # 获取最新的chunk id
        existing_files = os.listdir(f"{cfg['output_dir']}/{cfg['models']['name']}/")
        existing_chunk_ids = [
            int(f.split("_")[-1].split(".")[0])
            for f in existing_files
            if f.startswith("generation_chunk_")
        ]
        if existing_chunk_ids:
            start_chunk_id = max(existing_chunk_ids) + 1
    print(f"Resuming from chunk id: {start_chunk_id}")

    # 对所有的问题进行分块，减少内存占用
    qa_chunk_size = cfg["chunk_size"] * cfg["num_diverse_path"]
    for chunk_id in range(
        start_chunk_id, (len(train_dataset) + qa_chunk_size - 1) // qa_chunk_size
    ):
        log.info(f"process chunk {chunk_id}")

        chunk_start = chunk_id * qa_chunk_size
        chunk_end = min((chunk_id + 1) * qa_chunk_size, len(train_dataset))

        chunk_dataset = train_dataset.select(indices=range(chunk_start, chunk_end))
        outputs = llm.generate(
            prompts=chunk_dataset["prompts"],
            sampling_params=sampling_paras,
        )

        word_cnts = []
        is_corrects = []
        contents = []
        for output, sample in zip(outputs, chunk_dataset):
            problem = sample["problem"]
            label_text = sample["solution"]
            pred_text = output.outputs[0].text
            label_answer = extract_math_answer(problem, label_text)
            pred_answer = extract_math_answer(problem, pred_text)

            is_correct = compare_answers(problem, label_answer, pred_answer)
            word_cnt = len(output.outputs[0].token_ids) - 1
            contents.append(pred_text)
            word_cnts.append(word_cnt)
            is_corrects.append(is_correct)

        chunk_dataset = chunk_dataset.add_column("outputs", contents)
        chunk_dataset = chunk_dataset.add_column("word_cnt", word_cnts)
        chunk_dataset = chunk_dataset.add_column("is_correct", is_corrects)

        # 过滤掉不正确的QA
        chunk_dataset = chunk_dataset.filter(lambda example: example["is_correct"])
        chunk_dataset.to_json(
            f"{cfg['output_dir']}/{cfg['models']['name']}/generation_chunk_{chunk_id}.jsonl",
            lines=True,
        )

        # 输出正确的个数
        print(f"Chunk {chunk_id} correct count: {sum(is_corrects)}/{len(is_corrects)}")
    log.info("finish generation")


@hydra.main(config_path="../config", config_name="generate", version_base="1.3.1")
def generate(cfg: DictConfig):
    log.info(f"Generation config: {cfg}")
    if cfg.use_vllm:
        generate_vllm(cfg)
    else:
        raise NotImplementedError("Only vLLM generation is implemented.")


generate()
