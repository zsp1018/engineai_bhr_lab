#!/usr/bin/env bash
# Project-local dependency overrides for this workstation.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
export PYTHONPATH="$repo_dir/.deps/warp_1_8_1:$repo_dir/.deps/rsl_rl_5${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$repo_dir/.deps/cuda_driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c '
import runpy, sys
from pathlib import Path
import warp
if warp.__version__ != "1.8.1":
    raise RuntimeError("Expected project-local Warp 1.8.1")
print("Using Warp", warp.__version__, warp.__file__, flush=True)
warp.init()
if not warp.is_cuda_available():
    raise RuntimeError("Warp CUDA is unavailable")
script = Path("scripts/amp/play.py").resolve()
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), *sys.argv[1:]]
runpy.run_path(str(script), run_name="__main__")
' "$@"
