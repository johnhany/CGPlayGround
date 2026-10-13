#!/usr/bin/env bash
# 启动 sea_boat 本地静态服务器（检测端口占用并清理旧进程）
set -e

PORT=38517
DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/3] 检查端口 $PORT 占用"
PIDS="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
if [ -n "$PIDS" ]; then
  echo "      端口被占用，结束旧进程: $(echo "$PIDS" | tr '\n' ' ')"
  echo "$PIDS" | xargs kill 2>/dev/null || true
  sleep 1
else
  echo "      端口空闲"
fi

echo "[2/3] 以 $DIR 为根启动静态服务器（禁缓存头，避免开发期模块缓存）"
echo "[3/3] 访问 http://127.0.0.1:$PORT"

cd "$DIR"
exec python3 "$DIR/scripts/serve.py" "$PORT"
