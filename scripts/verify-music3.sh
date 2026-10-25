#!/usr/bin/env bash
# One command for the Music 3 gates that are easy to skip.
#
# WHY THIS EXISTS: the 4-bit "MLX gap" cost a day because the tolerances had been
# loosened until the gate passed, the hash oracles were diffed by hand, and the
# MLX environment took an afternoon to rebuild each time. The real cause (MLX
# dispatches the dense depth projection and heads as one batched matmul, a
# different kernel from two single-row calls) was a one-in-eight-thousand bit
# difference that a loose tolerance can never see. See docs/audio/minimax_music3.md.
#
# Usage:
#   scripts/verify-music3.sh                      # kernel and model gates, no MLX
#   scripts/verify-music3.sh --mlx MODEL_DIR      # also diff against MLX 0.32.3
#
# MODEL_DIR is the pinned 4-bit snapshot (c7ea32923b24), for example
# ~/.turbospark/models/audio/minimax-music3-4bit.gturbo.
#
# Stage 1 (always, needs a Metal device): the golden hash oracles for the affine
# kernels, the gpu kernel gates, the audio unit tests, and the runtime Metal
# gates. A kernel change that is meant to be exact must leave all of them green.
#
# Stage 2 (--mlx): builds a scratch MLX 0.32.3 environment under
# ~/.cache/turbospark/music3-mlx (Python 3.12: 3.15 only offered mlx 0.32.4),
# dumps MLX's own linear outputs on the real depth weights and requires
# our kernels to match bit for bit, then regenerates a native MLX reference and
# runs the ignored 4-bit parity test against it (hiddens, logits, codes, flow and
# the waveform, all bit-exact). Needs network the first time.
#
# Timing is not checked here. Benchmark on AC power only: a run that started on
# battery silently halved the clocks mid-measurement.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

mlx_model=""
while [ $# -gt 0 ]; do
  case "$1" in
    --mlx) mlx_model="${2:?--mlx needs the 4-bit model directory}"; shift 2 ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

step() { printf '\n== %s\n' "$*"; }

if [ "$(uname -s)" != "Darwin" ]; then
  echo "Music 3 gates need macOS and a Metal device" >&2
  exit 2
fi

step "stage 1: kernel and model gates"
cargo test -q -p turbospark-gpu --test music3_oracle
cargo test -q -p turbospark-gpu --test music3_sequence --test music3_parity --test music3_precision
cargo test -q -p turbospark-audio --lib minimax_music3
cargo test -q -p turbospark-runtime --test music3_metal

if [ -z "$mlx_model" ]; then
  echo
  echo "stage 1 passed. Dense kernels and the MLX trajectory were NOT checked:"
  echo "  scripts/verify-music3.sh --mlx ~/.turbospark/models/audio/minimax-music3-4bit.gturbo"
  exit 0
fi

# Pins. The reference source hash and revision are asserted again by the probe.
mlx_version="0.32.3"
mlx_audio_pin="feb25a37b07923bae556e59111995071d66afa0d"
revision="c7ea32923b245fe5afc22d740a1936ad2ac590f3"
work="${TURBOSPARK_MUSIC3_MLX_DIR:-$HOME/.cache/turbospark/music3-mlx}"
mkdir -p "$work"

step "stage 2: MLX $mlx_version environment ($work)"
py312="$(command -v python3.12 || true)"
if [ -z "$py312" ]; then
  echo "python3.12 not found (python3.15 only offered mlx 0.32.4)" >&2
  exit 2
fi
if [ ! -x "$work/venv/bin/python" ] || \
   [ "$("$work/venv/bin/python" -c 'import mlx.core as m; print(m.__version__)' 2>/dev/null || true)" != "$mlx_version" ]; then
  rm -rf "$work/venv"
  "$py312" -m venv "$work/venv"
  "$work/venv/bin/pip" install -q "mlx==$mlx_version" numpy \
    "huggingface_hub>=1.0" "transformers>=5.14.0" scipy tqdm pyyaml soundfile
fi
if [ ! -d "$work/mlx-audio/.git" ]; then
  git clone -q https://github.com/Blaizzy/mlx-audio.git "$work/mlx-audio"
fi
git -C "$work/mlx-audio" checkout -q "$mlx_audio_pin"
"$work/venv/bin/pip" install -q --no-deps -e "$work/mlx-audio"

step "stage 2a: linear kernels against MLX, bit for bit"
rm -rf "$work/linear"
"$work/venv/bin/python" crates/audio/tools/gen_music3_mlx_linear_cases.py \
  --model "$mlx_model" --output "$work/linear"
cargo run -q --release -p turbospark-gpu --example music3_mlx_linear_cmp -- \
  "$work/linear" --require-exact

step "stage 2b: native 4-bit pipeline (25 frames, 4 flow steps) against MLX, waveform included"
rm -rf "$work/reference"
PYTHONPATH="$work/mlx-audio" "$work/venv/bin/python" \
  crates/audio/tools/probe_minimax_music3_reference.py \
  --model "$mlx_model" --output "$work/reference" --checkpoint-revision "$revision" \
  --frames 25 --steps 4
# The probe does not record the encoding the test's tolerance table reads.
"$work/venv/bin/python" - "$work/reference/request.json" <<'PY'
import json, sys
path = sys.argv[1]
request = json.load(open(path))
request["encoding"] = "affine4"
json.dump(request, open(path, "w"), indent=1)
PY
TURBOSPARK_MUSIC3_INSTALL_DIR="$mlx_model" \
TURBOSPARK_MUSIC3_REFERENCE_DIR="$work/reference" \
  cargo test -q -p turbospark-runtime --test music3_metal --release \
  real_music3_pinned_4bit -- --ignored

echo
echo "all Music 3 gates passed (kernels, MLX linear and conv diff, 4-bit end-to-end parity)"
