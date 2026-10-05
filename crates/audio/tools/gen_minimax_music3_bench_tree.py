#!/usr/bin/env python3
"""Generate synthetic MiniMax Music 3 bench trees at enlarged dims.

These trees back the timing profiles in
src/models/music/minimax_music3/timing.rs and the long-AR stress test.
They are NOT parity fixtures: the weights are the seeded MLX default
init (weight seed 1234), and `audio_end_token_id` is set to a value at
or above `vocab_size`, which removes the end token from the AR sampling
mask entirely so a run's frame count is exactly the request. Real
checkpoints keep the end token inside the vocabulary.

Variants:

- tiny_long
    ModelConfig.tiny() with the end token pushed out of the mask.
    Small enough to commit under
    crates/speech/testdata/minimax_music3_bench/tiny_long; regenerate
    it there when the fixture set regenerates.
- ar_real
    Real-width AR backbone (hidden 4096, real ffn width, 32/8 heads at
    head_dim 128, real depth decoder) at --layers layers (default 6)
    and vocab 8192, with tiny flow components. A full fp32 real-dims
    tree is ~44 GB and does not fit the bench host; the timing test
    labels its per-layer and real-vocab projections accordingly.
- flow_real
    Real DiT, vocoder, and condition output dims with a tiny AR
    backbone, so one flow chunk profiles the real DiT and vocoder.

Run from inside the mlx-audio checkout root so `mlx_audio` imports
from the pinned source:

    cd ../mlx-audio
    PYTHONPATH=$PWD ../.venv-mlxaudio/bin/python \
        ../turbospark/crates/speech/tools/gen_minimax_music3_bench_tree.py \
        --variant tiny_long \
        --out ../turbospark/crates/speech/testdata/minimax_music3_bench

The default --out is /tmp/minimax_music3_bench, sized for the large
variants; pass the testdata path for tiny_long. Disk: tiny_long is a
few MB, ar_real about 7 GB, flow_real about 12 GB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_audio.music.models.minimax_music3.config import ModelConfig
from mlx_audio.music.models.minimax_music3.minimax_music3 import Model

WEIGHT_SEED = 1234

# Reuse the official-tree writer from the fixture generator so the
# trees go through the same conversion path the parity fixtures did.
import gen_minimax_music3_fixtures as fixture_gen  # noqa: E402


def end_token_outside_vocab(config: ModelConfig) -> ModelConfig:
    """Push the end token out of the sampling mask.

    The AR stage's allowed set is the semantic block plus the end
    token id; an id at or above vocab_size is never in [0, vocab), so
    the end token can never be sampled and the run length is exactly
    the requested frame count.
    """
    config.audio_end_token_id = config.vocab_size + 1999
    return config


def tiny_long_config() -> ModelConfig:
    return end_token_outside_vocab(ModelConfig.tiny())


def ar_real_config(layers: int) -> ModelConfig:
    config = ModelConfig.tiny()
    config.hidden_size = 4096
    config.num_hidden_layers = layers
    config.intermediate_size = 12288
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.head_dim = 128
    config.max_position_embeddings = 10240
    config.rope_theta = 1_000_000.0
    config.vocab_size = 8192
    config.audio_code_offset = 2048
    config.audio_cfg_token_id = 2046
    config.semantic_vocab_size = 1024
    config.depth_num_layers = 4
    config.depth_num_heads = 16
    config.depth_intermediate_size = 6144
    config.depth_max_position_embeddings = 16
    # Flow components stay tiny: the AR profile must not pay for them.
    return end_token_outside_vocab(config)


def flow_real_config() -> ModelConfig:
    config = ModelConfig.tiny()
    # Real flow geometry.
    config.dit_in_channels = 128
    config.dit_num_layers = 36
    config.dit_num_heads = 32
    config.dit_head_dim = 64
    config.dit_ff_inner_dim = 8192
    config.dit_rotary_dim = 32
    config.dit_fourier_dim = 256
    config.condition_out_dim = 2048
    config.vocoder_input_dim = 1024
    config.vocoder_hidden_dim = 1536
    config.vocoder_upsampling_ratios = [8, 8, 4, 2]
    # The AR backbone stays tiny; the flow stage is the subject.
    return end_token_outside_vocab(config)


VARIANTS = {
    "tiny_long": lambda layers: tiny_long_config(),
    "ar_real": ar_real_config,
    "flow_real": lambda layers: flow_real_config(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/minimax_music3_bench"),
        help="output directory; use the testdata path for tiny_long",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=6,
        help="AR layer count for ar_real (default 6)",
    )
    args = parser.parse_args()

    official = args.out / f"{args.variant}_official"
    converted = args.out / args.variant
    official.mkdir(parents=True, exist_ok=True)
    converted.mkdir(parents=True, exist_ok=True)

    mx.random.seed(WEIGHT_SEED)
    config = VARIANTS[args.variant](args.layers)
    model = Model(config)
    print(
        f"{args.variant}: {sum(v.size for _, v in tree_flatten(model.parameters())) / 1e9:.2f} "
        "B parameters"
    )
    fixture_gen.write_official_tree(official, model, config)

    from mlx_audio.convert import convert

    convert(str(official), str(converted), quantize=False)
    # prepare_config drops the tiny token contract, so re-align the
    # converted config with the generating config exactly as the
    # fixture generator does.
    path = converted / "config.json"
    value = json.loads(path.read_text(encoding="ascii"))
    value["audio_code_offset"] = config.audio_code_offset
    value["audio_end_token_id"] = config.audio_end_token_id
    value["audio_cfg_token_id"] = config.audio_cfg_token_id
    value["semantic_vocab_size"] = config.semantic_vocab_size
    path.write_text(json.dumps(value, indent=1, sort_keys=True), encoding="ascii")

    import shutil

    shutil.rmtree(official)
    print(f"wrote {converted}")


if __name__ == "__main__":
    main()
