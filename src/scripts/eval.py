"""
@File    :   eval.py
@Time    :   2025/11/16 14:10:49
@Author  :   Lin
@Version :   1.0
@Desc    :   None
copyright USTC
"""

import json

from transformers import Qwen2TokenizerFast
from vllm import LLM, SamplingParams

from parser import extract_Math

tokenizer: Qwen2TokenizerFast = Qwen2TokenizerFast.from_pretrained(
    # "../concise-reasoning/models/Qwen/Qwen2.5-1.5B-Instruct"
    "../concise-reasoning/models/trained/augmented/Qwen2.5-1.5B-Instruct/gsm8k/ft_16_shortest/ckpts/checkpoint-37"
)
qas = []
with open(file="data/math/math_test.json", mode="r") as f:
    for line in f.readlines():
        qas.append(json.loads(line))

llm = LLM(
    model="../concise-reasoning/models/Qwen/Qwen2.5-1.5B-Instruct",
    enable_prefix_caching=True,
    dtype="bfloat16",
    tensor_parallel_size=1,
)

few_shots = []
with open(
    file="../concise-reasoning/data/few_shot_examples/math/few-shot-gpt4o.json",
    mode="r",
) as f:
    for qa in json.load(f):
        few_shots.append(
            {"role": "user", "content": f"Question: {qa['question']}\nSolution:"}
        )
        few_shots.append({"role": "assistant", "content": qa["solution"]})
few_shots.append(
    {
        "role": "user",
        "content": "Please reason step by step and place the final answer in \\boxed{}.",
    }
)

sampling_params = SamplingParams(max_tokens=512, temperature=0.3, top_k=8, top_p=0.8)
sampling_params = SamplingParams(max_tokens=512, temperature=0.0, top_k=-1, top_p=1.0)

# preprocess
all_inputs = []
for qa in qas:
    inputs = {
        "messages": few_shots
        + [{"role": "user", "content": f"Question: {qa['problem']}\nSolution:"}]
    }
    inputs = tokenizer.apply_chat_template(
        inputs["messages"], tokenize=False, add_generate_prompt=True
    )
    all_inputs.append(inputs)
outputs = llm.generate(all_inputs, sampling_params)
for output, qa in zip(outputs, qas):
    output = output.outputs[0].text
    real_answer = extract_Math(qa["solution"])
    pred_answer = extract_Math(output)
    print(f"{real_answer}   {pred_answer}")
