# Pinned kernel references

Verified, reproducible settings for kernels in the engines this repository
compares against: `nace-ai/kapitsa`, `jundot/omlx`, and `ddalcu/mlx-serve`.
One JSON file per kernel entry, named `<short_name>.json` under the repo's
directory (`pinned/<repo>/`); the entry's id is `<repo>.<short_name>`. `crates/lab/src/pinned.rs` reads and validates them; every
file is also embedded into the lab binary at compile time, so the verified
baseline travels with `tslab` and a sweep can layer tweaked copies over it
through `--dir`.

The point is comparison, not admiration. Each entry holds the settings the
kernel was shipped and measured with, traceable to a source line at a pinned
revision, so a sweep can copy a set into an arm, change one field, and know
exactly what the baseline was.

## File format (`turbospark.lab.pinned/1`)

```json
{
  "schema": "turbospark.lab.pinned/1",
  "id": "mlx-serve.verify_lanes",
  "kernel": "the verify lane table (split-K, msg, matmul2d, NAX m16, crossrow)",
  "kernel_class": "verify_matmul",
  "repo": "mlx-serve",
  "summary": "One to three sentences: what runs, when, on which shapes.",
  "source": {
    "url": "https://github.com/ddalcu/mlx-serve",
    "rev": "191da71de32a65f69ebceeb6cb2e91036ab50b30",
    "rev_date": "2026-10-22",
    "license": "MIT",
    "files": ["src/transformer.zig"],
    "symbols": ["vqmmLaneForTile"]
  },
  "models": [
    {
      "name": "Qwen3.8-27B",
      "quant": "MLX affine 4-bit, group 32/64/128, bf16 scales",
      "family": "qwen3_5",
      "machine": "M4 Max 128 GB",
      "role": "primary"
    }
  ],
  "settings": [
    {
      "name": "split_kv_parts",
      "value": 2,
      "unit": "count",
      "condition": "N >= 4096",
      "kind": "measured",
      "source": "src/transformer.zig:412",
      "note": "optional; 1: 42.5 ms, 2: 42.4, 4: 44.5, 8: 49.0 per their A/B"
    }
  ],
  "gates": [
    {"condition": "M in 2..=7 and N >= 512 and K % 64 == 0 and N % 4 == 0",
     "source": "src/transformer.zig:398"}
  ],
  "measured": [
    {
      "metric": "decode_tok_s",
      "value": 109,
      "model": "Qwen3.8-27B 4-bit",
      "machine": "M4 Max 128 GB",
      "conditions": "DFlash2 tree verify at block 8, code-completion prompt, median of 3",
      "provenance": "upstream_reported",
      "source": "benchmarks.md"
    }
  ],
  "lab_mapping": [
    {"setting": "split_kv_parts",
     "maps_to": "tslab kernel --bench int4_matmul shape axis (no split axis yet)",
     "env": null,
     "note": "optional"}
  ],
  "docs": ["docs/MLX_SERVE_KERNELS.md#the-verify-lanes-which-are-the-whole-story"],
  "notes": "conflicts with our docs, unverified claims, caveats"
}
```

## Field rules (enforced by the reader and its tests)

- `schema` is exactly `turbospark.lab.pinned/1`.
- `repo` is one of `kapitsa`, `omlx`, `mlx-serve`. `id` is `<repo>.<short>`
  with a lowercase `[a-z0-9_]` short name, unique across the directory, and
  the short name is the file's stem.
- `kernel_class` is one of `gemv`, `verify_matmul`, `prefill_matmul`,
  `attention_decode`, `attention_prefill`, `gdn`, `moe`, `sampling`,
  `drafter`, `scheduler`, `fusion`, `probe`.
- `source.rev` is the 40-hex commit the entry was read at; `rev_date` is that
  commit's date. `files` are repo-relative paths that exist at that rev.
- `models` and `settings` are non-empty. `settings[].name` is snake_case and
  unique within the entry. Every `settings[].source` and `gates[].source` is
  a `path:line` into the pinned revision, and every `measured[].source`
  names the upstream file or doc the number came from.
- `role` starts with `primary` or `secondary`; a qualifier may follow it
  naming what that model measured for this entry (which A/B or ladder it
  belongs to).
- `kind` is `hardcoded` (a literal in the source), `derived` (computed from
  shape or config, formula noted), or `measured` (picked by an upstream A/B;
  put the comparison in `note`).
- `measured[].provenance` is `upstream_reported`: these are the other
  engine's own numbers under its own conditions, never ours, never
  re-measured here. A number without its `conditions` is not a number.
  `value` is a number, a string when the upstream record states the result
  as one compound comparison (a before and after, a spread), or an array of
  numbers for a curve over rows or widths (`conditions` says what each
  element is).
- `lab_mapping.setting` names the setting(s) the mapping applies to (comma
  separated when several); `env`, when present, is a `TURBOSPARK_` variable.
- Every string is ASCII (the workspace rule), and `notes` records anything
  our comparison docs claim that the source does not show.

## Pinned revisions

| Repo | Rev | Date | License |
|------|-----|------|---------|
| `nace-ai/kapitsa` | `14d1b03f306032f20426375651c8181a9105857d` | 2026-10-25 | Apache-2.0 |
| `jundot/omlx` | `78fa08bc1aef7290c93c83bb70811547763ebf33` | 2026-10-25 | Apache-2.0 |
| `ddalcu/mlx-serve` | `191da71de32a65f69ebceeb6cb2e91036ab50b30` (tag v26.10.3) | 2026-10-22 | MIT (incorporated parts Apache-2.0, per its NOTICE) |

Note: `docs/MLX_SERVE_KERNELS.md` recorded mlx-serve as Apache-2.0; the
repository's own LICENSE is MIT with `LICENSE-APACHE-2.0` for the
incorporated third-party parts, and the entries record what the LICENSE file
says.

## How entries are produced and kept honest

Each entry was written by a dedicated pass over the cloned source at the
pinned revision, with line numbers checked against the files, and
cross-read against `docs/KAPITSA_KERNELS.md` and `docs/MLX_SERVE_KERNELS.md`.
When those docs disagree with the source, the entry sides with the source
and records the conflict in `notes`.

To sweep against a pinned set: copy the JSON, change the fields the sweep
varies, and load the directory variant explicitly (`tslab pinned --dir ...`,
or `PinnedSet::load_dir` from code). Entries loaded from a directory are
validated by the same rules; an id that collides with an embedded entry
overrides it for that process so a sweep arm can hold a tweaked baseline.
