# vllm 部署 qwen

[qwen vllm 部署文档](https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html)

```bash
docker pull nvcr.io/nvidia/pytorch:23.10-py3

docker run -tid --name llm_test_qwen32b -p 8333:8333 -v /home/llm/Qwen2___5-VL-32B-Instruct:/root/Qwen2___5-VL-32B-Instruct --gpus all --ipc=host nvcr.io/nvidia/pytorch:23.10-py3 

docker exec -ti llm_test_qwen32b bash

pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com

pip install uv

uv venv
source .venv/bin/activate

uv pip install vllm --torch-backend=auto


nohup vllm serve /root/Qwen2___5-VL-32B-Instruct --served-model-name Qwen2.5-VL-32B-Instruct --tokenizer-mode auto --port 8333 --host 0.0.0.0 --dtype bfloat16 --max-model-len 32768 --trust-remote-code --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --limit-mm-per-prompt image=10,video=1 --api-key sensoro > vllm.log 2>&1 &
```
