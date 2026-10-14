---
description: "What each line and column printed by turbospark-bench means, and which fields you need before you can quote a row."
icon: table
---

# Reading the output

`turbospark-bench` writes results to stdout and progress, footers, and failures to stderr. This page explains each printed field, where it comes from, and what it does not tell you. Read the [overview](README.md) for the protocol conditions and [Running the harness](running-the-harness.md) for the commands that produce this output.

A row is not usable on its own. Keep the header lines with the table. The header holds the family, window, budget, slots, KV setting, shaping, speculation, prefill path, and power settings. The source comments say that a peak or a tok/s value measured at one window says nothing about another.

## Scripted mode

```text
turbospark-bench: 3 fixed prompts, seed 42, scripted producer (see module docs)
prompt         tokens    decode_secs  tokens_per_second
0 ...
aggregate: <tokens> tokens in <secs>s = <rate> tokens/sec
```

- `tokens` is the number of generated tokens for that prompt.
- `decode_secs` is wall-clock time around the raw completion call. It includes the whole loop, not only the forward pass.
- `aggregate` sums the measured runs. The warmup run per prompt is not included.

This output measures the loop overhead of the tokenizer, sampler, and detokenizer with a fixed token stream. Do not quote it as inference speed.

## Real install mode

### Header (stdout)

The first lines name the install, the chip, and the resolved settings. Example shape, not a real result:

```text
turbospark-bench: real install <dir> on <chip>, frozen protocol real-generation-v1
  family=<family> context=<n> max_new=<n> expert_cache_slots=<n> kv_bits=<label>
  speculative=off shaping=protocol-sampled
  prefill=sequential routed_batch=unset batched_gemv=unset (both seams live only in the chunked driver; INERT here)
  power_profile=performance max_tok_s=- thermal_stepping=false
```

| Field | Meaning |
| --- | --- |
| `real install <dir> on <chip>` | The chip line is omitted if the chip name cannot be read. |
| `family`, `context`, `max_new` | Resolved from the install's manifest. The CLI cannot override them. |
| `expert_cache_slots` | The slot count you passed, or the protocol default. |
| `kv_bits` | The KV setting. `off` is FP16. |
| `speculative=off` | Non-speculative run. |
| `speculative=on drafter=<dflash2 or mtp> block=<n>` | Resolved drafter and block. `auto` shows the real block, not the word auto. |
| `shaping=protocol-sampled` | The frozen sampled setting. |
| `shaping=GREEDY (...)` | Not the frozen protocol. The printed text says the row is not comparable with the published power rows. |
| `prefill=sequential` | The sequential path. The `routed_batch` and `batched_gemv` values are echoed but inert on this path. |
| `prefill=chunked chunk_tokens=<n>` | Chunked prefill with the resolved chunk size. |
| `power_profile`, `max_tok_s`, `thermal_stepping` | The resolved power pair. `max_tok_s=-` means no cap. |

The header is printed before any warmup runs. A run that fails in the warmup prints the column labels and no rows.

### Table (stdout)

```text
case               prompt_tok  prefill_s  new_tok  decode_s    tok_s  peak_mib
short-explanation  ...
medium-review      ...
long-synthesis     ...
session peak phys_footprint: <n> MiB
```

| Column | What it is |
| --- | --- |
| `prompt_tok` | Prompt tokens after the chat template is applied. |
| `prefill_s` | Prefill seconds from the runtime's own timing, not wall time. |
| `new_tok` | Generated tokens in the measured run. |
| `decode_s` | Decode seconds from the runtime's own timing, not wall time. |
| `tok_s` | `new_tok` divided by `decode_s`, printed to three decimals. |
| `peak_mib` | The shared sampler's peak `phys_footprint` so far in this process, in MiB. |

Two details matter when you read the peak:

- The sampler is shared across all cases in the process. A later row can only show the same or a higher peak than an earlier row. It is not a per-case peak.
- `phys_footprint` is sampled before prefill, on every 8th decoded token, and once after the run. A short spike between samples is not captured.

The last line is the same sampler's session peak. The source says the protocol's number is this process-wide peak. Its value depends on how many cases ran in the process, so use `--case` runs when you need one case's peak, as described in [Running the harness](running-the-harness.md).

### Footer and markers (stderr)

Each measured case prints a Swift-format footer on stderr:

```text
[stop=endOfTurn prefill=<N>tok/<S>s new=<M>tok decode=<S>s tok/s=<X>]
```

- `stop` uses the Swift stop-reason spelling: `endOfTurn`, `toolCalls`, `eos`, `stopString`, `maxTokens`, or `cancelled`. The source comment says the protocol's validity gate requires `endOfTurn` on every case. A case that stops on `maxTokens` is not a protocol result.
- The footer repeats the table values. Its `tok/s` uses the same formula as the table.

Two marker lines bracket the measured run of each case, not the warmup:

```text
[power-window case=<id> phase=start unix_ms=<ms>]
[power-window case=<id> phase=end unix_ms=<ms>]
```

`scripts/power.sh` uses these to cut a `powermetrics` capture to the measured window. Load time, the discarded warmup, and the open are outside it. Read the power procedure in `.claude/docs/power-measurement.md` before you use the markers yourself.

A failed warmup or measured run prints `<case> warmup failed: ...` or `<case> failed: ...` on stderr and exits with status 1.

## Checklist before you quote a row

Check these in the header and the run conditions. The list follows the rules in `crates/bench/AGENTS.md` and `docs/BENCHMARKING.md`.

- The row has its family, context, budget, slots, KV setting, and chip.
- The shaping is `protocol-sampled`, not `GREEDY`.
- Speculation is off, or the row is labelled as a speculative arm and is not compared with non-speculative rows.
- The prefill path matches the arm you claim.
- The power pair matches the arm name.
- The run was on AC power, on a quiet host, with warmup discarded and arms interleaved.
- The footer says `stop=endOfTurn` for every case.
- The number is copied from the run's own output, not from memory or a previous session.

If any item is missing, the row is a timing sample. The frozen rows live in `docs/BENCHMARKS.md`, and the protocol is in `docs/BENCHMARKING.md`. This page does not restate either.
