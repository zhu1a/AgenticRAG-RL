#!/bin/bash
# 启动 BGE Embedding + Reranker HTTP 服务
# 用法: RETRIEVAL_GPU=0 bash training/start_retrieval_server.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
LOG=${SCRIPT_DIR}/logs/retrieval_server.log
PID_FILE=${SCRIPT_DIR}/logs/retrieval_server.pid
RETRIEVAL_GPU=${RETRIEVAL_GPU:-0}
RETRIEVAL_PORT=${RETRIEVAL_PORT:-8790}
RETRIEVAL_DEVICE=${RETRIEVAL_DEVICE:-cuda:0}

mkdir -p "${SCRIPT_DIR}/logs"

# 只停止由本脚本记录的旧进程，避免误杀其他用户的检索服务。
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null && \
       ps -p "${OLD_PID}" -o command= | grep -q "retrieval_server.py"; then
        kill "${OLD_PID}"
        sleep 1
    fi
fi

echo "Starting retrieval server on physical GPU ${RETRIEVAL_GPU}, port ${RETRIEVAL_PORT}..."
CUDA_VISIBLE_DEVICES="${RETRIEVAL_GPU}" nohup "${PYTHON}" \
  "${SCRIPT_DIR}/training/tools/retrieval_server.py" \
  --port "${RETRIEVAL_PORT}" --device "${RETRIEVAL_DEVICE}" \
  > "${LOG}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"

echo "PID: ${SERVER_PID}"
echo "Waiting for server to load models (~30s)..."

for i in $(seq 1 60); do
    sleep 5
    if curl -fsS "http://localhost:${RETRIEVAL_PORT}/health" | grep -q "ok"; then
        echo "Retrieval server ready!"
        # 快速测试
        curl -fsS -X POST "http://localhost:${RETRIEVAL_PORT}/embed" \
          -H 'Content-Type: application/json' \
          -d '{"texts": ["测试查询"]}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Embed OK: dim={len(d[\"embeddings\"][0])}, elapsed={d[\"elapsed\"]:.3f}s')"
        exit 0
    fi
    echo "  waiting... (${i}/60)"
done

echo "ERROR: Server failed to start. Check ${LOG}"
exit 1
