#!/usr/bin/env bash
# 下载进度速览：aria2 汇总 + 各目录实际占用 + 完整性校验
cd "$(dirname "$0")" || exit 1
echo "== aria2 最新进度"
grep -E '^\[#' dl.log 2>/dev/null | tail -1 || echo "(没有 dl.log，aria2 可能已退出)"
echo
echo "== 目录占用（实际磁盘）"
du -sh */ 2>/dev/null | sort -h
echo
echo "== 完整性校验"
python3 verify.py
echo
echo "== 断点续传"
echo "python3 dl.py perplexity-ai/pplx-decider-flash-v1.3 baselabs/elephant-1 baselabs/bison-1 perplexity-ai/pplx-decider-v1.3 --jobs 4 --conn 8"
