#!/usr/bin/env python3
"""Convert ACE-Step 1.5 `silence_latent.pt` to safetensors, once, at install.

The checkpoint ships the silence latent as a torch pickle. The runtime
never unpickles: this tool reads it with `torch.load(weights_only=True)`
(the restricted tensor-only unpickler upstream also uses), applies
upstream's `.transpose(1, 2)` (init_service_loader.py at the pinned
commit), and writes `silence_latent.safetensors` holding one f32 tensor
`silence_latent` shaped [frames, 64], time-major.

    ../.venv-acestep/bin/python -I crates/audio/tools/convert_ace_step15_silence.py \
        <checkpoint>/acestep-v15-turbo
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main(dit_dir: Path) -> None:
    src = dit_dir / "silence_latent.pt"
    raw = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(raw, torch.Tensor) or raw.dim() != 3 or raw.shape[0] != 1 or raw.shape[1] != 64:
        sys.exit(f"unexpected silence latent {type(raw)} {getattr(raw, 'shape', None)}")
    latent = raw.transpose(1, 2)[0].float().contiguous()
    out = dit_dir / "silence_latent.safetensors"
    save_file({"silence_latent": latent}, str(out))
    print(f"{out}: {tuple(latent.shape)} sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    print(f"source {src.name} sha256 {hashlib.sha256(src.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: convert_ace_step15_silence.py <acestep-v15-turbo dir>")
    main(Path(sys.argv[1]))
