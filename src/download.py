"""
@File    :   eval.py
@Time    :   2025/11/16 13:14:25
@Author  :   Lin
@Version :   1.0
@Desc    :   None
copyright USTC
"""

from modelscope import snapshot_download

# model_dir = snapshot_download(
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", cache_dir="models/"
# )
model_dir = snapshot_download(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", cache_dir="models/"
)
