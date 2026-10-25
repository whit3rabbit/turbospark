---
description: Native speculative decoding with Multi-Token Prediction (MTP) heads and DFlash drafters.
icon: gauge-high
---

# Speculative Decoding and MTP

Speculative decoding accelerates token generation by drafting multiple speculative candidate tokens and verifying them in parallel with a single forward pass of the target model.

On Apple Silicon, where inference speed is primarily bound by unified memory bandwidth rather than compute ALU limits, verifying multiple tokens simultaneously yields measurable throughput increases without degrading output quality.

## Multi-Token Prediction (MTP) Heads

The primary speculative mechanism implemented in TurboSpark is **Native Multi-Token Prediction (MTP)**:

* **Architecture**: The model carries a dedicated MTP prediction head that predicts token `t + 1` concurrently with token `t`.
* **Block Speculation**: Candidates are proposed in blocks (typically block length 2).
* **Parallel Verification**: The main model's next forward pass evaluates candidate logits in parallel. Accepted tokens are emitted immediately; rejected candidates roll back to the first mismatch.

### Measured Performance Gains (Qwen 3.8 27B on M4 Max)

| Configuration | Context | Decoding Speed | Measured Footprint |
| --- | ---: | ---: | ---: |
| **Qwen 3.8 27B (Standard greedy decode)** | 4,096 | 14.7 - 17.9 tok/s | ~661 MiB |
| **Qwen 3.8 27B + MTP Head (Speculative block 2)** | 4,096 | **21.9 - 28.7 tok/s** | ~872 MiB |

*Result*: MTP speculation delivers an approximately **1.4x to 1.6x speedup** on real Apple Silicon hardware with zero loss in output fidelity.

---

## Running with Speculation

Install the MTP-enabled checkpoint:

```sh
turbospark-model pull qwen38-27b-mtp
```

Run inference with speculative decoding enabled:

```sh
turbospark-check --model qwen38-27b-mtp --chat
```

### The `--speculative` flag

| Value | Meaning |
| --- | --- |
| `auto` (default) | Enable the MTP head when the install carries one, at the measured optimum of block 2. A DFlash drafter is reported but not run. |
| `off` | Plain greedy or sampled decode. |
| `1` to `15` | A fixed block size. The measured optimum is 2 and everything above 4 loses on this engine; the range is wider than the useful part on purpose. A named block **fails at open** if the model cannot serve it, rather than silently falling back. |
| `adaptive[:MAX]` | `turbospark-check` only. Experimental: the round size moves between 2 and `MAX` (3 to 15, default 4) with the acceptance each round saw. |

`turbospark-server` takes `off`, `auto` or a block size, and applies speculation to temperature-0 requests only; other requests decode sequentially.

`--speculative-drafter auto|mtp|dflash` picks which drafter the block drives. `auto` enables an MTP head but only reports a DFlash one, because DFlash measured 0.88x on prose; name `dflash` to run it.

```sh
# Fixed block 2 (what auto chooses on an MTP install)
turbospark-check --model qwen38-27b-mtp --speculative 2 --prompt "..."

# Adaptive depth, capped at 6
turbospark-check --model qwen38-27b-mtp --speculative adaptive:6 --prompt "..."
```

The same choice is available on the C ABI (`speculation: "adaptive"`) and in Swift (`OpenOptions.Speculation.adaptive(cap:)`).

{% hint style="warning" %}
Adaptive depth is lossless but, after a retune, measured within noise of the fixed block 2 default: no gain, and one short recall case 9 to 19 percent slower. It stays opt-in and `auto` is unchanged. `docs/MTP_SPECULATIVE.md` in the repository holds the measurements.
{% endhint %}

Set `TURBOSPARK_SPEC_STATS=1` to log acceptance rates and rollback counts to stderr.

---

## Rollback and Recurrent State

A speculative round that rejects a draft has to rewind the model to the last accepted token. For attention-only models that is a KV cursor move. The Qwen GDN families (Qwen 3.5, 3.6 and 3.8 hybrids) also carry a recurrent state per layer, and the 0.5.0 release redesigned how that state is protected:

* A round checkpoint skips its snapshot when the trunk is already clean, and an eager advance captures a deferred snapshot only when it is needed. Checkpoints stay reusable across rollbacks until the round commits.
* Under the lazy GDN tape (`TURBOSPARK_GDN_LAZY_TAPE`, on by default), a verify on clean recurrent state records a rank-1 tape instead of blitting a snapshot, and the state fold commits asynchronously on the GPU so CPU drafting is not blocked on round completion.
* If an error lands between verify and commit, the producer discards its pending state so the session does not carry a half-applied round.

None of this changes the tokens produced. Set `TURBOSPARK_GDN_LAZY_TAPE=0` to snapshot on every verify for an A/B. See the [Environment Variables Reference](environment-variables.md).

---

## DFlash Drafter Research

TurboSpark also includes experimental support for **DFlash** (a block-diffusion speculative drafter). While DFlash generates fast candidates, empirical benchmarks show that native MTP heads provide significantly higher token acceptance rates with lower memory overhead on Apple Silicon unified memory. `TURBOSPARK_DFLASH_DRAFT=0` disables the drafter; `--speculative-drafter dflash` is required to run it.
