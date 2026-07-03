#!/bin/bash
# LongMemEval Oracle 全量评测脚本
# Usage: bash evaluation/longmemeval/run_async.sh
#
# 可选环境变量控制日志级别:
#   LOG_LEVEL=WARNING REME_LOG_LEVEL=WARNING bash evaluation/longmemeval/run_async.sh
#   LOG_LEVEL=ERROR   bash evaluation/longmemeval/run_async.sh  # 只减少 reme 内部日志

# 激活 conda 环境
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate reme

# 切换到项目根目录（脚本所在目录的上两级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 带时间戳的日志文件（精确到分钟）
LOG="evaluation/longmemeval/logs/run_$(date +%Y%m%d_%H%M).log"

echo "=========================================="
echo "  LongMemEval Oracle 全量评测"
echo "  日志文件: $LOG"
echo "  启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

nohup python evaluation/longmemeval/run.py \
  --config evaluation/longmemeval/config.yaml \
  ${LOG_LEVEL:+--log-level "$LOG_LEVEL"} \
  ${REME_LOG_LEVEL:+--reme-log-level "$REME_LOG_LEVEL"} \
  > "$LOG" 2>&1 &

PID=$!
echo "后台进程 PID: $PID"
echo "查看日志: tail -f $LOG"
