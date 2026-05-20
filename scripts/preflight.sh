#!/usr/bin/env bash
# REACT preflight: 检查环境就绪度，不做任何修改
# 仅读环境状态，绝不安装/修改任何依赖。任何 [FAIL] 必须先停下报告给用户。
set -u

PASS="\033[32m[PASS]\033[0m"
FAIL="\033[31m[FAIL]\033[0m"
WARN="\033[33m[WARN]\033[0m"
fails=0

echo "== REACT preflight =="

# 1) OS / Shell
if uname -a | grep -qi linux; then
  echo -e "$PASS Linux kernel"
else
  echo -e "$FAIL 非 Linux 环境，REACT 训练需 Ubuntu"
  fails=$((fails+1))
fi

# 2) Python & venv (兼容 conda 与 venv；/usr 视为系统 python，发 WARN)
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo -e "$PASS conda env: $CONDA_DEFAULT_ENV"
elif [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" != "/usr" && "$VIRTUAL_ENV" != "/usr/local" ]]; then
  echo -e "$PASS venv: $VIRTUAL_ENV"
else
  echo -e "$WARN 未检测到激活的 Python 虚拟环境（conda/venv 都没有；当前 VIRTUAL_ENV=${VIRTUAL_ENV:-未设}）"
fi
if python -c "import sys; print(f'python={sys.version.split()[0]}')"; then
  echo -e "$PASS python 可执行"
else
  echo -e "$FAIL python 不可用"
  fails=$((fails+1))
fi

# 3) CUDA / PyTorch
if python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, dev={torch.cuda.get_device_name(0)}")
PY
then
  echo -e "$PASS CUDA 可用"
else
  echo -e "$FAIL CUDA 或 torch 不可用"
  fails=$((fails+1))
fi

# 4) Flightmare / flightlib
if python -c "import flightgym" 2>/dev/null; then
  echo -e "$PASS flightgym (Flightmare Python bindings) 可 import"
else
  echo -e "$FAIL 无法 import flightgym -- 检查 FLIGHTMARE_PATH 与 build/wheel"
  fails=$((fails+1))
fi
if [[ -n "${FLIGHTMARE_PATH:-}" ]]; then
  echo -e "$PASS FLIGHTMARE_PATH=$FLIGHTMARE_PATH"
else
  echo -e "$WARN 环境变量 FLIGHTMARE_PATH 未设置（部分 launch 文件需要）"
fi

# 5) ROS（如果训练流水线依赖）
if command -v rosversion >/dev/null 2>&1; then
  echo -e "$PASS ROS: $(rosversion -d) $(rosversion roscpp 2>/dev/null)"
else
  echo -e "$WARN 未检测到 ROS -- 若仅训练可忽略，真机/launch 阶段必须装"
fi

# 6) YOPO 仓库结构（适配实际双层嵌套：外层 git 根含 Controller/Simulator/YOPO 三个兄弟，
#    Python 训练代码在内层 YOPO/ 下）
declare -a required=(
  "Controller"
  "Simulator"
  "YOPO/train_yopo.py"
  "YOPO/policy"
  "YOPO/config"
  "YOPO/loss"
)
for f in "${required[@]}"; do
  if [[ -e "$f" ]]; then
    echo -e "$PASS 仓库含 $f"
  else
    echo -e "$FAIL 缺少 $f -- 当前目录可能不是 REACT 仓库根"
    fails=$((fails+1))
  fi
done

# 7) GPU 显存（粗略）
python - <<'PY' 2>/dev/null || echo -e "$WARN 无法读取 GPU 显存"
import torch
free, total = torch.cuda.mem_get_info()
print(f"\033[32m[PASS]\033[0m GPU 显存 free/total = {free/1e9:.1f}/{total/1e9:.1f} GB")
PY

echo "== 失败项数: $fails =="
exit $fails
