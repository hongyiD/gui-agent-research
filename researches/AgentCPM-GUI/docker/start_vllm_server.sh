#!/bin/bash
# vLLM server启动脚本
# 对于不支持bfloat16的GPU（如V100），需要使用 --dtype=half

MODEL_PATH=${MODEL_PATH:-/workspace/AgentCPM-GUI/model}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-agentcpm-gui}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
DTYPE=${DTYPE:-half}  # 对于V100等GPU，使用half而不是bfloat16

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype "${DTYPE}"
