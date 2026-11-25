#!/usr/bin/env bash
# 在 spark-d705 上启动决策模型服务（每个模型一个 uvicorn 进程）
#   ./start_decision.sh elephant|flash|glide|decider|bison|all|stop
set -u
ROOT=${MODEL_ROOT:-/home/sakana/models/decision-models}
VENV=${VENV:-$ROOT/.venv}
LOGS=$ROOT/logs
mkdir -p "$LOGS"

export PYTHONPATH="$ROOT/serving/onejev:$ROOT/pplx-decider-v1.3/source/src:$ROOT/pplx-decider-flash-v1.3/source/src:$ROOT/GLiDE-0.4B:$ROOT/serving"
export MODEL_ROOT="$ROOT"
export TOKENIZERS_PARALLELISM=false
# Triton (flash-linear-attention) needs python headers + libcuda at build time
export C_INCLUDE_PATH=/tmp/pydev/usr/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
export LIBRARY_PATH=/usr/local/cuda/lib64:/lib/aarch64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}
export PPLX_MODE=${PPLX_MODE:-isolated}
export PPLX_FAST=${PPLX_FAST:-1}
export PPLX_WARM_TOKENS=${PPLX_WARM_TOKENS:-1000,4000,8000,16000,24000}

start() {  # name port model [device]
  local name=$1 port=$2 model=$3 device=${4:-cuda:0}
  if [ -f "$LOGS/$name.pid" ] && kill -0 "$(cat "$LOGS/$name.pid")" 2>/dev/null; then
    echo "[$name] already running (pid $(cat "$LOGS/$name.pid"))"; return
  fi
  cd "$ROOT/serving" || exit 1
  MODEL="$model" DEVICE="$device" PRELOAD="${PRELOAD:-$model}" nohup "$VENV/bin/uvicorn" app.main:app \
      --host 0.0.0.0 --port "$port" > "$LOGS/$name.log" 2>&1 &
  echo $! > "$LOGS/$name.pid"
  echo "[$name] started pid $(cat "$LOGS/$name.pid") on :$port (model=$model)"
}

stop() {
  local name=$1
  if [ -f "$LOGS/$name.pid" ]; then
    kill "$(cat "$LOGS/$name.pid")" 2>/dev/null && echo "[$name] stopped"
    rm -f "$LOGS/$name.pid"
  fi
}

case "${1:-all}" in
  elephant) start elephant 8012 elephant-1 ;;
  flash)    start flash 8014 pplx-decider-flash-v1.3 ;;
  glide)    start glide 8015 GLiDE-0.4B ;;
  decider)  start decider 8013 pplx-decider-v1.3 ;;
  bison)    start bison 8011 bison-1 ;;
  stop)     for n in bison elephant decider flash glide; do stop "$n"; done ;;
  status)
    for n in bison elephant decider flash glide; do
      if [ -f "$LOGS/$n.pid" ] && kill -0 "$(cat "$LOGS/$n.pid")" 2>/dev/null; then
        echo "$n RUNNING pid $(cat "$LOGS/$n.pid")"
      else echo "$n stopped"; fi
    done ;;
  all)
    start elephant 8012 elephant-1
    start flash 8014 pplx-decider-flash-v1.3
    start glide 8015 GLiDE-0.4B ;;
  *) echo "usage: $0 {all|elephant|flash|glide|decider|bison|status|stop}"; exit 1 ;;
esac
