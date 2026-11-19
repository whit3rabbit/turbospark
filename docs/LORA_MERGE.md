# LoRA adapter merge at install

A PEFT LoRA repository (`adapter_config.json` plus `adapter_model.safetensors`,
no `config.json`) installs by merging the adapter into its base model while
the base streams. Example:
`harness-hub/Qwen3.8-27B-pi-adapted` (r=32, alpha=64) on `Qwen/Qwen3.8-27B`.

```sh
turbospark-model probe harness-hub/Qwen3.8-27B-pi-adapted
turbospark-model pull --repo harness-hub/Qwen3.8-27B-pi-adapted --alias qwen38-pi
```

If the merged MLX build exists, prefer it: `tozuo/Qwen3.8-27B-pi-adapted-MLX-4bit`
(catalog alias `qwen38-27b-pi`) is a 16 GB download. The adapter path downloads
the BF16 base (about 52 GiB) plus the 0.6 GB adapter, and needs that much
transfer, not that much disk: the install is about 15 GB.

## How it works

- `crates/repack/src/lora_merge.rs`: `LoraMergedSource` wraps each base
  shard's `RangeSource`. A read that overlaps a targeted tensor returns
  `base + scaling * (B @ A)`, computed in f32 and rounded to the base dtype.
  Headers are unchanged, so no writer changes.
- It stacks BEFORE `quantize_on_install`, so the quantizer sees merged
  weights, as `mlx_lm.convert` does for a merged checkpoint.
- `catalog::probe` recognizes an adapter repo, probes the BASE named in
  `base_model_name_or_path`, and sets `ProbeReport::adapter`. `InstallPlan`
  carries `adapter`; its sidecars default to the adapter repo's own tokenizer
  and chat template. `stream_mlx` downloads the adapter once, checks every
  module against the base headers, and refuses an unmatched or mis-shaped one.

## Refused, by name

DoRA, `lora_bias`, `fan_in_fan_out`, non-empty `rank_pattern` or
`alpha_pattern`, `modules_to_save`, any adapter tensor that is not a
`lora_A`/`lora_B` weight. A partial merge would read as a fluent, wrong model.

## Evidence

`crates/repack/tests/lora_merge_parity_network.rs` (ignored, network): merging
the pi adapter into the base and quantizing with the install's quantizer
reproduces the `.weight`, `.scales` and `.biases` bytes of
`tozuo/Qwen3.8-27B-pi-adapted-MLX-4bit` exactly for 64 rows each of
`layers.0.mlp.down_proj`, `layers.3.self_attn.o_proj` and
`layers.0.mlp.gate_proj`; the unmerged base differs on up to 1.0% of bytes.

Not run: a full adapter install (about 52 GiB of transfer), and no decode or
quality gate against either install. The `qwen38-27b-pi` row is `runs`, not
`verified`.
