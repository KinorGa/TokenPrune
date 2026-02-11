"""
@File    :   build_prompts.py
@Time    :   2026/02/10 10:08:55
@Author  :   Lin
@Version :   1.0
@Desc    :   None
copyright USTC
"""

import sys

PROMPT1 = """
## Question\n{question}\n## Thought process\n{thought_process}\n--Based on the above thought process, provide a clear, easy-to-follow, and well-formatted solution to the question. Use the same language as the question.  \nThe solution must strictly follow these requirements: \n- Stay faithful and consistent with the given thought process. Do not add new reasoning steps or conclusions not shown in the original. \n- Show key steps leading to final answer(s) in clear, well-formatted LaTeX. \n- Use \\boxed{} for final answer(s). \n- Be clean and concise. Avoid colloquial language. Do not use phrases like "thought process" in the solution.  \nYour response should start with the solution right away, and do not include anything else. Your task is solely to write the solution based on the provided thought process. Do not try to solve the question yourself."""

PROMPT2 = (
    "Please deduce the solution step by step and enclose the final answer in \\boxed{}."
)

PROMPT3 = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def format_question_with_prompt(question, prompt_id=1, format_id=1):
    current_module = sys.modules[__name__]
    prompt = getattr(current_module, f"PROMPT{prompt_id}")
    if prompt_id == 0:
        return f"\n## Question\n{question}\n"
    elif prompt_id == 1:
        return f"\n{prompt}\n## Question\n{question}\n## Solution\n"
    elif prompt_id == 2:
        return f"## Question\n{question}\n{prompt}\n"
    else:
        raise ValueError(f"Unsupported prompt_id: {prompt_id}")
