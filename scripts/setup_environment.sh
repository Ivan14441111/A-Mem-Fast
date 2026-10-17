#!/usr/bin/env bash
# Build a separate CPU benchmark-client environment; never replace an existing one.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_ENV="$PROJECT_DIR/.venv-client"
PYTHON_VERSION="3.12"
DOWNLOAD_ASSETS=1

usage() {
  cat <<'EOF'
Usage: scripts/setup_environment.sh [--venv PATH] [--python VERSION] [--skip-assets]

Creates .venv-client with Python 3.12, PyTorch CPU wheels, and requirements-fast.txt.
Any existing target path is refused. The project's existing .venv is untouched.
By default, also downloads NLTK punkt_tab and the pinned MiniLM embedding model.
Use --skip-assets when those assets are already provisioned or only installing tools.

Optional GPU server (a separate environment; these commands are not run here):
  uv venv --python 3.12 .venv-serve
  uv pip install --python .venv-serve/bin/python vllm==0.11.2 transformers==4.57.6
  INFERENCE_ENV="$PWD/.venv-serve" MODEL_DIR=/path/to/Qwen3-8B GPU_ID=0 scripts/serve_qwen.sh

The GPU server requires locally downloaded Qwen weights and a compatible NVIDIA
driver. It listens on 127.0.0.1:21014; benchmark clients use its /v1 endpoint.
EOF
}

while (($#)); do
  case "$1" in
    --venv|--python)
      if (($# < 2)); then
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      fi
      if [[ "$1" == --venv ]]; then CLIENT_ENV="$2"; else PYTHON_VERSION="$2"; fi
      shift 2
      ;;
    --skip-assets) DOWNLOAD_ASSETS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -e "$CLIENT_ENV" || -L "$CLIENT_ENV" ]]; then
  printf 'Refusing to replace existing path: %s\nChoose a fresh path with --venv.\n' "$CLIENT_ENV" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  printf 'The uv executable is required. Install uv, then rerun this script.\n' >&2
  exit 1
fi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

uv venv --python "$PYTHON_VERSION" "$CLIENT_ENV"
# The explicit local version selects CPU wheels before resolving the client pins.
uv pip install --python "$CLIENT_ENV/bin/python" \
  'torch==2.9.0+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$CLIENT_ENV/bin/python" -r "$PROJECT_DIR/requirements-fast.txt"

if [[ "$DOWNLOAD_ASSETS" == 1 ]]; then
  "$CLIENT_ENV/bin/python" - "$PROJECT_DIR" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download
import nltk

root = Path(sys.argv[1])
if not nltk.download("punkt_tab", download_dir=str(root / ".cache/nltk_data"), quiet=True):
    raise RuntimeError("NLTK punkt_tab download failed")
model = snapshot_download(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    allow_patterns=["config.json", "modules.json", "sentence_bert_config.json",
                    "config_sentence_transformers.json", "1_Pooling/*", "model.safetensors",
                    "tokenizer*", "special_tokens_map.json", "vocab.txt"],
)
print("Embedding model:", model)
PY
fi

"$CLIENT_ENV/bin/python" - "$CLIENT_ENV" <<'PY'
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

target = Path(sys.argv[1])
packages = ["torch", "transformers", "sentence-transformers", "numpy", "scipy",
            "scikit-learn", "nltk", "rank-bm25", "litellm", "openai", "requests", "pytest"]
(target / "environment-client.json").write_text(json.dumps({
    "python": platform.python_version(),
    "versions": {p: version(p) for p in packages},
    "embedding_device": "cpu",
}, indent=2) + "\n")
PY

printf '\nClient environment ready: %s\n' "$CLIENT_ENV"
printf 'Set these before running the benchmark:\n'
printf 'export NLTK_DATA=%q\n' "$PROJECT_DIR/.cache/nltk_data"
printf 'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false\n'
