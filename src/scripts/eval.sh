export PYTHONPATH=.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

uv run src/eval.py tensor_parallel_size=2