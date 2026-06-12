#!/usr/bin/env bash
# 检查 GPU / PyTorch / CUDA 环境
#
# 使用：
#   bash scripts/check_env.sh
#   uv run --group trl bash scripts/check_env.sh
#   uv run --group unsloth bash scripts/check_env.sh

set -euo pipefail

echo "========== 系统环境 =========="
echo "OS:       $(uname -s) $(uname -m)"
echo "Python:   $(python3 --version 2>/dev/null || echo 'not found')"
echo ""

python3 - <<'PYEOF'
import sys, os, shutil

print(f"Python:   {sys.version}")
print(f"Prefix:   {sys.prefix}")
print()

# ── CUDA 环境变量 ──
print("========== CUDA 环境变量 ==========")
for k in ("CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
    print(f"  {k} = {os.environ.get(k, '(未设置)')}")
nvcc = shutil.which("nvcc")
print(f"  nvcc:    {nvcc or '(未找到)'}")
if nvcc:
    import subprocess
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    for line in ver.stdout.strip().splitlines():
        if "release" in line.lower():
            print(f"           {line.strip()}")
print()

# ── nvidia-smi ──
print("========== nvidia-smi ==========")
nvsmi = shutil.which("nvidia-smi")
if nvsmi:
    import subprocess
    r = subprocess.run([nvsmi], capture_output=True, text=True)
    print(r.stdout.strip() if r.returncode == 0 else f"(执行失败: {r.stderr.strip()})")
else:
    print("  nvidia-smi 未找到（无 NVIDIA GPU 或驱动未安装）")
print()

# ── PyTorch ──
print("========== PyTorch ==========")
try:
    import torch
    print(f"  torch:           {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version:    {torch.version.cuda}")
        print(f"  cuDNN version:   {torch.backends.cudnn.version()}")
        print(f"  GPU count:       {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_mem / 1024**3
            print(f"  GPU {i}: {props.name}  ({mem_gb:.1f} GB, sm_{props.major}{props.minor})")
    else:
        print("  (CUDA 不可用)")
    print(f"  MPS available:   {torch.backends.mps.is_available()}" if hasattr(torch.backends, "mps") else "")
except ImportError:
    print("  torch 未安装")
print()

# ── transformers ──
print("========== transformers ==========")
try:
    import transformers
    print(f"  transformers:  {transformers.__version__}")
except ImportError:
    print("  transformers 未安装")

# ── peft ──
try:
    import peft
    print(f"  peft:          {peft.__version__}")
except ImportError:
    pass

# ── trl ──
try:
    import trl
    print(f"  trl:           {trl.__version__}")
except ImportError:
    pass

# ── accelerate ──
try:
    import accelerate
    print(f"  accelerate:    {accelerate.__version__}")
except ImportError:
    pass

# ── unsloth ──
try:
    import unsloth
    print(f"  unsloth:       {unsloth.__version__}" if hasattr(unsloth, "__version__") else "  unsloth:       installed")
except ImportError:
    pass

# ── datasets ──
try:
    import datasets
    print(f"  datasets:      {datasets.__version__}")
except ImportError:
    pass

print()
print("========== 检查完毕 ==========")
PYEOF
