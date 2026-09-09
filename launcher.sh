#!/usr/bin/env bash
# 启动 mossweb 的 Gradio 界面。
#
# 用法：
#   ./launcher.sh                       # 默认 http://127.0.0.1:7860
#   ./launcher.sh --port 8080           # 换端口
#   ./launcher.sh --host 0.0.0.0        # 允许局域网访问
#   ./launcher.sh --share --open        # 其余参数原样透传给 app.py
set -euo pipefail

# 切到脚本自身所在目录，这样从任何路径调用都能找对 model/ 和 core/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
if [ ! -f .venv/bin/activate ]; then
    echo "错误：当前目录下没有虚拟环境 .venv。" >&2
    echo "请先执行：" >&2
    echo "  uv venv --python 3.12 && uv pip install gradio" >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "工作目录：$(pwd)"
echo "解释器  ：$(command -v python)"
exec python app.py "$@"
