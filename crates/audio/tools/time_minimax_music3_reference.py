#!/usr/bin/env python3
"""Time the MiniMax Music 3 Python reference on the tiny scenarios.

Mirror of the Rust timing profiles in
src/models/music/minimax_music3/timing.rs: same fixture trees, same
seeds, same shapes, so the reference wall times can sit beside the
port's CPU numbers. The reference runs on the default MLX device
(Metal on this host) unless MLX_DEVICE=cpu is set; CPU-vs-CPU is the
only like-for-like axis, and the Rust numbers are CPU f32 either way.

Run from inside the mlx-audio checkout root:

    cd ../mlx-audio
    PYTHONPATH=$PWD ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/audio/tools/time_minimax_music3_reference.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_audio.music import load
from mlx_audio.music.models.minimax_music3.ar import generate_frame_hiddens
from mlx_audio.music.models.minimax_music3.minimax_music3 import (
    _encode_tiny_text_pair,
)
from mlx_audio.music.models.minimax_music3.prompt import assemble_prompt

TEXT = "Genre: acoustic pop. BPM: 96. Warm female vocal."
LYRICS = "[verse]\nMorning light\n[chorus]\nSing with me"
FRAMES_REQUEST = 201
STEPS = 2
SEED = 7
TOOLS_DIR = Path(__file__).resolve().parent


def timed(label: str, runs: int, body):
    """Discard one warmup, then time `runs` measured passes."""
    mx.eval(body())
    walls = []
    for _ in range(runs):
        start = time.perf_counter()
        result = body()
        # MLX is lazy: force the graph before reading the clock.
        mx.eval(result)
        walls.append(time.perf_counter() - start)
    print(f"  {label}: best {min(walls):.3f} s of {['%.3f' % w for w in walls]}")
    return min(walls), result


def tiled_hiddens(config, frames: int) -> mx.array:
    fused = config.num_codebooks * config.hidden_size
    pattern = ((np.arange(frames) % 8) * 0.01 - 0.03).astype(np.float32)
    return mx.array(np.tile(pattern[:, None], [1, fused]))[None, :, :]


def text_pair(config):
    return _encode_tiny_text_pair(assemble_prompt(TEXT, LYRICS), config)


def model_config(model, tree: Path):
    cached = getattr(model, "config", None)
    if cached is not None:
        return cached
    return json.loads((tree / "config.json").read_text(encoding="ascii"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="MLX device; the env var does not work on mlx 0.32.3",
    )
    parser.add_argument(
        "--trees",
        type=Path,
        default=TOOLS_DIR.parent / "testdata" / "minimax_music3_bench",
    )
    args = parser.parse_args()

    if args.device == "cpu":
        mx.set_default_device(mx.cpu)

    testdata = TOOLS_DIR.parent / "testdata" / "minimax_music3"
    print(f"device: {mx.default_device()} mlx {mx.__version__}")

    plain_tree = testdata / "converted_plain"
    model = load(str(plain_tree))
    config = model_config(model, plain_tree)
    ids = text_pair(config)

    # 1. AR stage, natural end-token stop (27 frames with these weights).
    _, hiddens = timed(
        "AR natural (fixture tree, stops on end token)",
        args.runs,
        lambda: generate_frame_hiddens(
            model.language_model,
            model.rvq_depth_decoder,
            config,
            ids,
            max_frames=FRAMES_REQUEST,
            seed=SEED,
        ),
    )
    print(f"    emitted {hiddens.shape[1]} frames")

    # 2. Flow over tiled hiddens (sampling-free).
    timed(
        "flow 201 frames (2 chunks, 2 steps)",
        args.runs,
        lambda: model._run_flow(tiled_hiddens(config, FRAMES_REQUEST), STEPS, SEED),
    )
    timed(
        "flow 1201 frames (12 chunks, 2 steps)",
        args.runs,
        lambda: model._run_flow(tiled_hiddens(config, 1201), STEPS, SEED),
    )

    # 3. End to end through the pipeline's own entry point.
    wall, audio = timed(
        "end-to-end generate (request 201 frames at 25 fps)",
        args.runs,
        lambda: mx.array(
            next(
                model.generate(
                    text=TEXT,
                    lyrics=LYRICS,
                    duration=FRAMES_REQUEST / config.frame_rate,
                    steps=STEPS,
                    seed=SEED,
                )
            ).audio
        ),
    )
    seconds = audio.shape[-1] / config.sample_rate
    print(f"    {audio.shape[-1]} samples = {seconds:.2f} s audio, RTF {wall / seconds:.4f}")

    # 4. Forced-length AR at the frame ceiling, matching the Rust
    #    tiny_ar_scaling_profile workload.
    tiny_long = args.trees / "tiny_long"
    if tiny_long.joinpath("config.json").is_file():
        long_model = load(str(tiny_long))
        long_config = model_config(long_model, tiny_long)
        timed(
            "AR tiny_long 9000 frames (end token out of mask)",
            args.runs,
            lambda: generate_frame_hiddens(
                long_model.language_model,
                long_model.rvq_depth_decoder,
                long_config,
                text_pair(long_config),
                max_frames=9000,
                seed=SEED,
            ),
        )
    else:
        print(f"  skipping tiny_long: {tiny_long} not found")


if __name__ == "__main__":
    main()
