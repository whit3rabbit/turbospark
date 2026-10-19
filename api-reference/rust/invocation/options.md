---
description: "Every command-line flag the invocation parser accepts, with its value shape, default, and the request field it fills."
icon: sliders
---

# Option reference

Use this page when you need the exact spelling, value shape, or default for a flag. The source of truth is the [`OPTIONS` table](https://github.com/whit3rabbit/turbospark/blob/main/crates/invocation/src/options.rs) and the parser in [`parser.rs`](https://github.com/whit3rabbit/turbospark/blob/main/crates/invocation/src/parser.rs). The table below was checked against both files. A flag that takes a value consumes the next token, so `Takes value` means the flag needs one token after it.

Two rules apply to every row. A flag that takes a value with a bad value fails as `InvalidValue`, and a flag at the end of the list with no value fails as `MissingValue`. Repeatable flags keep their order, and the parser does not sort or de-duplicate them.

## Required and mode

| Flag | Takes value | Default | Request field |
| --- | --- | --- | --- |
| `--model` | Yes, opaque path | Required | `model` |
| `--prompt` | Yes, text | Mode-selecting | `mode` as `Mode::Prompt` |
| `--messages-file` | Yes, path | Mode-selecting | `mode` as `Mode::MessagesFile` |
| `--humanize-file` | Yes, path or `-` | Mode-selecting | `mode` as `Mode::HumanizeFile` |
| `--chat` | No | Mode-selecting | `mode` as `Mode::Chat` |
| `--system` | Yes, text, repeatable | None | `system` |

Exactly one mode must be selected. The parser joins repeated `--system` values with a newline, in the order you gave them. `--system` is refused unless the mode is `--chat`.

## Generation and sampling

| Flag | Takes value | Default | Accepted values | Request field |
| --- | --- | --- | --- | --- |
| `--max-new` | Yes | `1024` | Positive integer | `max_new` |
| `--temperature` | Yes | `0.2` | Finite number, zero or more | `temperature` |
| `--top-k` | Yes | `64` | Integer, `0` to `256`. `0` disables rank-based shaping. | `top_k` |
| `--top-p` | Yes | `0.95` | Finite number, greater than `0` and at most `1` | `top_p` |
| `--repetition-penalty` | Yes | `1.0` | Finite number, greater than `0` | `repetition_penalty` |
| `--seed` | Yes | None | Non-negative integer, 64 bit | `seed` |
| `--stop` | Yes, repeatable | None | Any text, kept in order | `stop` |
| `--reasoning` | Yes | `off` | `off`, `low`, `medium`, `high`, `xhigh` | `reasoning` |
| `--max-tokens-per-sec` | Yes | None | Finite number, greater than `0` | `max_tokens_per_sec` |
| `--quiet` | No | Off | Flag only | `quiet` |

The `--reasoning` set is a union across checkpoints. Each checkpoint accepts its own subset, and the checkpoint's template reports a mismatch when the model opens.

## Context and memory

| Flag | Takes value | Default | Accepted values | Request field |
| --- | --- | --- | --- | --- |
| `--max-context` | Yes | `auto` | Positive integer, or `auto` | `max_context` |
| `--min-auto-context` | Yes | `0` | Non-negative integer. `0` sets no floor. | `min_auto_context` |
| `--load-guard` | Yes | `relaxed` | `off`, `relaxed`, `balanced`, `strict`, or a positive byte count | `load_guard` |
| `--expert-cache-slots` | Yes | `auto` | `8`, `16`, `24`, `32`, `48`, `64`, `96`, `128`, or `auto` | `expert_cache_slots` |
| `--expert-residency` | Yes | `auto` | `auto`, `streamed`, `mapped` (case-insensitive) | `expert_residency` |
| `--kv-bits` | Yes | `off` | `off`, `2`, `3`, `3.5`, `4` | `kv_bits` |
| `--prefill-chunk` | Yes | `128` | `32`, `64`, `128`, `256`, `512`, `1024`, `2048`, `4096`, or `auto` | `prefill_chunk` |
| `--rdadvise` | Yes | `off` | `off`, `normal`, `aggressive` | `rdadvise` |

`--max-context auto` and `--expert-cache-slots auto` stay as keywords in the request. The runtime resolves them once it knows the install and the machine. `--min-auto-context` applies only to automatic resolution, never to an explicit `--max-context` number.

`--load-guard` takes a named tier or a plain positive integer. A number becomes `LoadGuard::Custom`, which is a byte ceiling on what the engine may allocate.

## Images and batching

| Flag | Takes value | Default | Request field |
| --- | --- | --- | --- |
| `--image` | Yes, path, repeatable | None | `images`, in order |
| `--image-batch` | No | Off | `image_batch` |

All `--image` values land in one turn unless `--image-batch` is set. With `--image-batch`, the caller runs the prompt once per image. The image paths are opaque here, so decoding them is the front end's job.

## Speculation and drafting

| Flag | Takes value | Default | Accepted values | Request field |
| --- | --- | --- | --- | --- |
| `--speculative` | Yes | `auto` | `auto`, `off`, an integer from `1` to `15`, or `adaptive[:MAX]` (MAX `3` to `15`, default `4`; experimental) | `speculation` |
| `--speculative-drafter` | Yes | `auto` | `auto`, `mtp`, `dflash` | `speculative_drafter` |

`auto` for `--speculative` warns and decodes sequentially when the install cannot speculate. A named block size, and `adaptive`, fail outright in that case, so a benchmark cannot quietly run without speculation. `adaptive` moves the round size between 2 and MAX for the MTP head only and measured no faster than the fixed default (`docs/MTP_SPECULATIVE.md`); `auto` never selects it. Detecting a drafter does not switch it on, and `auto` does not enable a DFlash drafter by default.

## Steering

| Flag | Takes value | Default | Accepted values | Request field |
| --- | --- | --- | --- | --- |
| `--steering` | Yes, path, repeatable | None | Opaque path | `steering` |
| `--steering-mode` | Yes, repeatable | Per vector file, or `ablate` | `ablate`, `add`, `clamp`, `renorm` | `steering_mode` |
| `--steering-scale` | Yes, repeatable | `1.0` | Finite number | `steering_scale` |
| `--steering-layers` | Yes, repeatable | Every layer the vector covers | `START:END`, inclusive, 0-based, not inverted | `steering_layers` |
| `--steering-target` | Yes | `0.0` | Finite number, used by `clamp` only | `steering_target` |
| `--steering-gate` | Yes | `0.0` | Finite number, zero or more. `0.0` means always. | `steering_gate` |

The per-vector flags pair with `--steering` by position. The `i`th value configures the `i`th path. If a list is shorter than the path list, its last value applies to the rest. The parser refuses a per-vector value with no `--steering`, and it refuses more values than paths. `steering_knob` in the [crate docs](https://docs.rs/turbospark-invocation/latest/turbospark_invocation/fn.steering_knob.html) implements the same rule for callers that build their own requests.

## Vision sidecar

| Flag | Takes value | Default | Request field |
| --- | --- | --- | --- |
| `--vision-sidecar` | Yes, path | None | `vision_sidecar` |

The value is an opaque path. The keyword `auto` is not special in this crate, so it reaches the front end as a literal path.

## Power

| Flag | Takes value | Default | Accepted values | Request field |
| --- | --- | --- | --- | --- |
| `--power-profile` | Yes | Unset, resolved when the session opens | `performance`, `balanced`, `efficiency` | `power_profile` |

## Short-circuit flags

| Flag | Result |
| --- | --- |
| `--help` | `ParseOutcome::Help`, reached in scan order |
| `--version` | `ParseOutcome::Version`, reached in scan order |

These flags take no value and skip every whole-invocation check.

## Adding a flag

Adding a flag touches several places, and the crate's tests enforce that the table and the parser agree. Add the entry to `OPTIONS`, add a parser arm in both the scan and the value match, add the request field and its default, and update the counts in `tests/usage_and_status.rs`, which pin five required or mode-selecting options and thirty-three others. The [testing rules](https://github.com/whit3rabbit/turbospark/blob/main/docs/TESTING.md) describe the expected test coverage.

Flags also cross surfaces. The [CLI reference](https://github.com/whit3rabbit/turbospark/blob/main/docs/CLI.md) and the [tool calling reference](https://github.com/whit3rabbit/turbospark/blob/main/docs/TOOL_CALLING.md) describe how the same meanings reach users and the server.
