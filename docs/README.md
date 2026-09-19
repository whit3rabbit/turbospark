# Documentation guide

This directory is the developer reference for TurboSpark. Start with the
page that matches the task, then follow its links to the implementation and
the evidence that supports the claim.

## Start here

| If you need to... | Read |
|---|---|
| Build, test, or run the workspace | [`DEVELOPMENT.md`](DEVELOPMENT.md) |
| Choose the right test gate | [`TESTING.md`](TESTING.md) |
| Measure throughput, memory, power, or quality | [`BENCHMARKING.md`](BENCHMARKING.md) |
| Read the frozen Swift comparison and other measured rows | [`BENCHMARKS.md`](BENCHMARKS.md) |
| Find, probe, or install a model | [`MODELS.md`](MODELS.md) |
| Add or assess a model family | [`MODEL_FAMILY.md`](MODEL_FAMILY.md) and [`NEW_MODEL.md`](NEW_MODEL.md) |
| Use the command-line tools | [`CLI.md`](CLI.md) |
| Configure environment variables | [`ENV.md`](ENV.md) |
| Understand the text install format | [`GTURBO.md`](GTURBO.md) |
| Build a release artifact | [`RELEASE.md`](RELEASE.md) |

## Runtime and integration references

- Generation and state: [`STREAMING.md`](STREAMING.md),
  [`DECODE_BUDGET.md`](DECODE_BUDGET.md), [`MTP.md`](MTP.md),
  [`MTP_SPECULATIVE.md`](MTP_SPECULATIVE.md), [`DFLASH2.md`](DFLASH2.md),
  and [`SPECULATIVE_DECODING.md`](SPECULATIVE_DECODING.md).
- Memory and routing: [`LOAD_GUARD.md`](LOAD_GUARD.md),
  [`EXPERT_ROUTING.md`](EXPERT_ROUTING.md),
  [`EXPERT_RESIDENCY.md`](EXPERT_RESIDENCY.md),
  [`BATCHED_PREFILL.md`](BATCHED_PREFILL.md), and
  [`ACTIVATION_SPARSITY.md`](ACTIVATION_SPARSITY.md).
- Metal, power, and quantization: [`KERNELS.md`](KERNELS.md),
  [`POWER_BASELINE.md`](POWER_BASELINE.md),
  [`TRUBOQUANT.md`](TRUBOQUANT.md), and
  [`OBLITERATION.md`](OBLITERATION.md).
- Interfaces and safety: [`SWIFT_BINDINGS.md`](SWIFT_BINDINGS.md),
  [`TOOL_CALLING.md`](TOOL_CALLING.md), and
  [`FORGE_GUARDRAILS.md`](FORGE_GUARDRAILS.md),
  [`PERMISSION_GATE.md`](PERMISSION_GATE.md), and
  [`SKILL_STATE.md`](SKILL_STATE.md).
- Vision and image generation: [`VISION.md`](VISION.md),
  [`IMAGE_GENERATION.md`](IMAGE_GENERATION.md), and
  [`ZIMAGE_TURBO.md`](ZIMAGE_TURBO.md).

## Architecture and evidence records

The following pages record pinned architecture facts, parity work, and
negative findings. They are evidence records, not a work log. Read the
current-status or conclusion section first, then use the measurements and
reproduction commands when you need to verify a claim.

- [`DEEPSEEK2_PHASE0.md`](DEEPSEEK2_PHASE0.md)
- [`BONSAI2.md`](BONSAI2.md)
- [`IMAGE_GENERATION_PHASE0.md`](IMAGE_GENERATION_PHASE0.md)
- [`MINIMAX_M2_PHASE0.md`](MINIMAX_M2_PHASE0.md)
- [`QWEN3VL_PHASE0.md`](QWEN3VL_PHASE0.md)
- [`QWEN4_EXP.md`](QWEN4_EXP.md)
- [`QWEN4_PHASE0.md`](QWEN4_PHASE0.md)
- [`SPARK_PHASE0.md`](SPARK_PHASE0.md)
- [`VISION_PHASE0.md`](VISION_PHASE0.md)

These pages are useful when changing the corresponding family or feature.
They do not replace the current implementation guides above.

## How to read a claim

Use the evidence label, not the age of a heading:

- **Current** means the page describes the behavior in this checkout.
- **Measured** means a command, input, environment, and result are recorded.
- **Verified** means a checked-in test or gate asserts the behavior.
- **Experimental** means the path is opt-in, default-off, or still being
  evaluated.
- **Historical** means the page preserves reasoning or a negative result that
  still prevents a regression. It is not a promise about the default path.
- **Planned** means the implementation or gate does not exist yet.

For benchmark numbers, keep the protocol beside the result. At minimum,
record the model or install, hardware, operating system, build mode, input,
counter definition, date, and any limitations. CPU output, compilation, or a
synthetic fixture does not establish a real Metal, quality, memory, or
production claim.

## Generated and machine-readable evidence

[`verification/`](verification/) contains machine-readable captures and
measurement artifacts. Treat these files as evidence inputs. The Markdown
pages explain what each artifact proves, what it cannot prove, and how to
reproduce it. Do not turn a raw capture into a broader claim without the
matching gate.

If a page disagrees with the code, tests, or a checked-in measurement, fix the
page or mark the claim as stale. Do not add a dated session note to preserve
the disagreement.
