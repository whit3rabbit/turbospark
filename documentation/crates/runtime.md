---
description: "The decode engine: family dispatch, forward passes, state, sessions, and the Core ML pipelines, behind one generation loop."
icon: gears
---

# runtime

`runtime` (package `turbospark-runtime`) is the largest crate in the workspace.
It owns the raw-completion prefill and decode loop, the `LogitProducer`
abstraction that feeds it, and `RealForwardRunner`, the macOS/Metal engine that
turns a `.gturbo` install into next-token logits. It also hosts the Core ML
model pipelines, the native speech and audio model runners, and the power and
memory policy the decode loop obeys. The CLI, server, FFI layer, and benchmark
harnesses are all thin surfaces over this crate.

## What it does

| Responsibility | Where |
| --- | --- |
| Generation loop: prefill, decode, detokenization, stop matching, cancellation, chunked prefill | `crates/runtime/src/raw_completion.rs`, `src/raw_completion_chunked.rs` |
| Producer abstraction: `LogitProducer`, `ChunkedPrefillRunner`, `SpeculativeProducer`, and the `ScriptedLogitProducer` test mock | `crates/runtime/src/producer.rs` |
| Real model engine: `RealForwardRunner`, Metal dispatch, expert streaming, KV and GDN state | `crates/runtime/src/real_forward*.rs` |
| Speculative decoding: drafter policy plus the speculative loop | `crates/runtime/src/speculation_policy.rs`, `src/speculative.rs` |
| Directional steering dispatch | `crates/runtime/src/steering.rs` |
| Session plumbing: KV prefix reuse, a bounded session pool, rollback points | `crates/runtime/src/kv_prefix.rs`, `src/session_pool.rs`, `src/real_forward_rollback.rs` |
| Power and pacing: power profiles, thermal and memory ladders, decode-rate caps | `crates/runtime/src/power.rs`, `src/pacing.rs` |
| Core ML pipelines: Parakeet, EmbeddingGemma2, Unlimited OCR, diarization | `crates/runtime/src/coreml/` |
| Native speech and audio runners: Whisper, Qwen3 ASR, Moonshine, Kokoro, Music3, MegaTTS3, OmniVoice, Fish Drama 3 | `crates/runtime/src/whisper/`, `src/qwen3_asr.rs`, `src/kokoro.rs`, `src/music3.rs`, `src/megatts3.rs`, `src/omnivoice.rs`, `src/fish_drama3.rs` |
| Vision tower forward pass and the BERT/XLM-R encoder runner | `crates/runtime/src/vision/`, `src/encoder/` |
| Process-wide gate that keeps text, image, and audio work from sharing one Metal device at once | `crates/runtime/src/heavy.rs` |

Most of the crate is gated `#[cfg(target_os = "macos")]` because it drives
Metal. The generation loop, producer traits, generation config, power
constants, and turn splitting are portable.

## Where it sits

Depends on:

- Portable: `core` (TokenId, LogitValue), `selection` (the sampler),
  `tokenizer`.
- macOS-gated: `gpu`, `model-io`, `compute`, `streaming`, `audio`,
  `vision-io`, `coreml`.
- Dev-only: `repack` (synthetic installs for tests) and `embeddings` (one
  backend probe example).

Used by: `cli`, `server`, `ffi`, and `bench`. Two sizing policies
(`MaxContext`, `ExpertCacheSlots` and friends) live in `model-io` and are
re-exported here, because `catalog` needs the same arithmetic before an
install exists and every consumer expects to find them named `runtime::`.

## Public API tour

### The generation loop

`run_raw_completion` (`crates/runtime/src/raw_completion.rs`) is the entry
point every producer goes through. It takes a `&mut dyn LogitProducer`, a
loaded `MfTokenizer`, the prompt token ids, a `GenerationConfig`, a
`max_context`, and the vocabulary size, and it drives prefill (every prompt
token but the last through `produce_prefill`, which may skip the output head),
sampling through `selection`, streaming detokenization, and stop matching. It
reports progress through a callback carrying `RawDecodeProgress::Prefill`,
`::Token { index, id, delta }`, or `::Tail`, and returns a `RawDecodeResult`
with `new_tokens`, `prefill_seconds` / `decode_seconds`, the `StopReason`, and
the KV cache position and the token ids backing it. Variants add cancellation
(`run_raw_completion_cancellable`, polled once per token through a
`CancelFlag`) and chunked prefill (`run_raw_completion_chunked`). `StopReason`
covers `EndOfTurn`, `ToolCalls`, `Eos`, `StopString`, `MaxTokens`, and
`Cancelled`; a cancel is an ordinary result, not an error. `GenerationConfig`
(`crates/runtime/src/config.rs`) bundles a `selection::ShapingConfig` with
`max_new_tokens`, stop strings, extra stop token ids, and a `RateControl` for
decode pacing. The speculative loop (`run_raw_completion_speculative`) drives
a `SpeculativeProducer` that drafts and verifies blocks of tokens.

### RealForwardRunner and family dispatch

`RealForwardRunner` (`crates/runtime/src/real_forward.rs` and its
`real_forward_*` siblings) is the production `LogitProducer`. It opens a
`.gturbo` install directory with an expected `ArchConfig` (from `model-io`,
typically read with `turbospark_repack::peek_manifest_arch`, as
`crates/cli/src/generate/session.rs` does):

```rust
RealForwardRunner::open(dir, expecting) -> Result<Self, RealForwardError>
```

`open` matches the manifest field-by-field against `expecting` and selects the
decode flow from `ArchConfig.family` at open time (the `match` lives in
`crates/runtime/src/real_forward_open.rs`). The crate's central invariant is
that this is the ONLY selection mechanism: never tensor names. Fifteen
`ModelFamily` variants are defined in `crates/model-io/src/arch_config/family.rs`
and map onto nine private flows under `crates/runtime/src/families/`:

| Family variants | Flow module |
| --- | --- |
| `Gemma4` | `families/gemma4/` |
| `QwenGdnMoe`, `QwenGdnDense` | `families/qwen/` |
| `Llama`, `Qwen3Moe`, `Qwen3Dense`, `Qwen2Dense`, `MiniMaxM2`, `Qwen3Vl` | `families/llama/` |
| `GptOss` | `families/gptoss/` |
| `MuseGlimmer` | `families/museglimmer/` |
| `Spark25` | `families/spark/` |
| `Qwen4Exp` | `families/qwen4/` |
| `Deepseek2` | `families/deepseek2/` |
| `DeepseekV4Flash` | refused at open |
| synthetic short-name fixtures | `families/synthetic/` |

Two families sharing a flow are told apart inside it, again by
`ArchConfig.family` or `num_experts`, never by names. Several family strings
are frozen format constants in `manifest.json` that no longer match their
variant names (for example `QwenGdnMoe` writes `"qwen36"`); see
[model families](../models/model-families.md).

After opening, the runner implements `LogitProducer`, `ChunkedPrefillRunner`,
and `SpeculativeProducer`, so the same `run_raw_completion` loop serves it.
Useful accessors include `vocab_size()` (ask the runner, not the tokenizer
dialect), `supports_chunked_prefill()`, `set_prefix_reuse(bool)` for
cross-turn KV reuse, `expert_cache_slots()` and `session_pool_size()` (worth
printing beside any measurement), and `phase_counters()`. Session state is
part of correctness: `reset()` rewinds the KV cache AND the GDN recurrent
state, and a bounded pool (`crates/runtime/src/session_pool.rs`, sized at
open) parks per-session KV and recurrent state so several conversations can
share one runner.

### Load guard, power, and speculation policy

Re-exported from `model-io`: `MaxContext`, `ExpertCacheSlots`, `LoadPolicy`,
`KvQuant`, `ExpertResidency`, and the `resolve_max_context` /
`largest_context_within` / `committed_bytes` arithmetic
(`crates/model-io/src/context_policy.rs`). `crates/runtime/src/power.rs` owns
`PowerProfile`, `ThermalLevel`, `MemoryPressure`, `RateControl`, and the OS
probes (`physical_memory`, `recommended_max_working_set`). The speculation
policy (`crates/runtime/src/speculation_policy.rs`) makes three decisions in a
fixed order: `resolve_drafter` asks what the install carries, `draft_policies`
turns that into the open-time `DraftPolicies`, and `resolve_speculation`
decides whether a run may draft.

### coreml and the model runners

`runtime::coreml` (`crates/runtime/src/coreml/`) wraps the Core ML exports:
`ParakeetCoreMl` for speech-to-text, `ParakeetUnifiedCoreMl` with
`UnifiedStream` for streaming words, `EmbeddingGemma2CoreMl` for embeddings,
`UnlimitedOcrCoreMl` for OCR, and `DiarizationCoreMl` for speaker turns, plus
`detect_layout` to identify which export a directory holds. Alongside it,
`whisper::WhisperRunner`, `qwen3_asr::Qwen3AsrRunner`, `moonshine`,
`kokoro::KokoroRunner`, `music3::Music3Runner`, `megatts3`, `omnivoice`, and
`fish_drama3` run the native audio models; `embedding_probe::probe_embedding_dir`
classifies an embedding-model directory before anything loads; `heavy` is the
process-wide `HeavyWorkGuard` shared by text, image, and audio hosts.

## Examples

### Scripted generation (portable)

Adapted from `crates/runtime/tests/raw_completion.rs`, simplified. A
`ScriptedLogitProducer` stands in for a model, so the loop itself runs
anywhere; the tokenizer fixture ships at
`crates/runtime/tests/fixtures/ChatMLTokenizer`.

```rust
use std::path::PathBuf;

use turbospark_core::LogitValue;
use turbospark_runtime::{
    run_raw_completion, GenerationConfig, RawDecodeProgress, ScriptedLogitProducer,
    StopReason,
};
use turbospark_selection::ShapingConfig;
use turbospark_tokenizer::MfTokenizer;

let dir = PathBuf::from("tests/fixtures/ChatMLTokenizer");
let tokenizer = MfTokenizer::load_from_dir(&dir)?;
let vocab_size = tokenizer.vocab_size;

// Resolve ids from the loaded tokenizer, never from hardcoded numbers.
let prompt_ids = tokenizer.encode("hi", false);
let eos = tokenizer.end_of_turn_id as usize;

// The loop calls produce_prefill for every prompt token but the last, then
// produce per decoded token. A one-hot logits row samples deterministically.
let one_hot = |i: usize| {
    let mut row = vec![LogitValue::from_f32(0.0); vocab_size];
    row[i] = LogitValue::from_f32(1.0);
    row
};
let mut steps = vec![vec![LogitValue::from_f32(0.0); vocab_size]; prompt_ids.len() - 1];
steps.push(one_hot(eos)); // the first decoded token is the stop token

let mut producer = ScriptedLogitProducer::new(steps);
let config = GenerationConfig {
    shaping: ShapingConfig::new(0.0, 0, None, 1.0, None)?, // greedy
    max_new_tokens: 10,
    stop_strings: Vec::new(),
    extra_stop_tokens: Vec::new(),
    rate: Default::default(),
};

let result = run_raw_completion(
    &mut producer,
    &tokenizer,
    &prompt_ids,
    &config,
    4096, // max_context
    vocab_size,
    |event| {
        if let RawDecodeProgress::Token { delta, .. } = event {
            print!("{delta}");
        }
    },
)?;

assert_eq!(result.reason, StopReason::EndOfTurn);
```

### Open a real install and generate (macOS)

Adapted from `crates/runtime/tests/real_forward.rs` and
`crates/cli/src/generate/session.rs`. This needs a `.gturbo` install on disk;
the API shape is what matters here.

```rust
use std::path::Path;

use turbospark_runtime::{
    run_raw_completion, GenerationConfig, RawDecodeProgress, RealForwardRunner,
};
use turbospark_selection::ShapingConfig;
use turbospark_tokenizer::MfTokenizer;

let model_dir = Path::new("/Users/me/.turbospark/models/some-install.gturbo");

// The manifest's arch; open() matches it field-by-field against the install
// and refuses a mismatch or a family this build cannot serve.
let arch = turbospark_repack::peek_manifest_arch(model_dir)?;
let mut runner = RealForwardRunner::open(model_dir, arch)?;

let tokenizer = MfTokenizer::load_from_dir(model_dir)?;
let prompt_ids = tokenizer.encode("Hello", false);
let vocab_size = runner.vocab_size(); // the model's own padded row count

let config = GenerationConfig {
    shaping: ShapingConfig::new(0.0, 0, None, 1.0, None)?, // greedy
    max_new_tokens: 128,
    stop_strings: Vec::new(),
    extra_stop_tokens: Vec::new(),
    rate: Default::default(),
};

let result = run_raw_completion(
    &mut runner,
    &tokenizer,
    &prompt_ids,
    &config,
    8192, // max_context
    vocab_size,
    |event| {
        if let RawDecodeProgress::Token { delta, .. } = event {
            print!("{delta}");
        }
    },
)?;
println!("decoded {} tokens in {:.2}s", result.new_tokens, result.decode_seconds);
```

The open path has siblings that opt into more: `open_with_max_context`,
`open_with_options` (expert cache slots), `open_with_slot_policy`
(`ExpertCacheSlots::Auto`), `open_with_kv_quant`, and forms that take the
drafting, steering, and session-pool knobs. Measuring callers keep those pinned
off by default so a footprint or digest row cannot acquire a feature by
detection.

### A Core ML pipeline (macOS)

Adapted from `crates/runtime/examples/coreml_backend_probe.rs`, simplified.
The pipeline needs an exported Core ML model directory.

```rust
use std::path::Path;

use turbospark_runtime::coreml::{ParakeetCoreMl, ParakeetCoreMlOptions};

let model_dir = Path::new("models/parakeet-coreml");

// Options are derived from the export's own metadata; the weight format
// decides the default compute units.
let opts = ParakeetCoreMlOptions::for_model(model_dir)?;
let model = ParakeetCoreMl::open(model_dir, &opts)?;

// samples: f32 mono PCM, resampled to the model's declared rate.
let text = model.transcribe(&samples, model.sample_rate())?;
```

## Gotchas

{% hint style="warning" %}
Decode flow selection keys on `ArchConfig.family`, never on tensor names. Two
families that share a flow are told apart inside it by the same field or by
`num_experts`, and Gemma 4 and Qwen carry overlapping tensor names, so a
naming probe would misroute real installs while still producing fluent text.
`DeepseekV4Flash` is refused at open rather than half-served. Family strings
in `manifest.json` are an on-disk format: two of them (`"qwen36"`,
`"qwen35"`) deliberately do not match their Rust variant names, and renaming
them would strand every existing install.
{% endhint %}

{% hint style="warning" %}
Producers return raw, unnormalized logits. `selection` owns softmax,
temperature, top-k, and top-p; returning probabilities from `produce` silently
destroys sampling temperature. Relatedly, `produce_prefill` may skip the
output head for prompt tokens, but it must still advance every other
per-token side effect (KV cache, position), and a session `reset()` must
rewind the recurrent GDN state, convolution tails, and speculative state, not
just the KV cache: stale state decodes fluent but wrong text.
{% endhint %}

## Related pages

- [Crate overview](overview.md)
- [Model families](../models/model-families.md)
- [The .gturbo format](../models/gturbo-format.md)
- [Core ML models](../models/coreml.md)
- [Speculative decoding](../advanced/speculative-decoding.md)
- [Directional steering](../advanced/directional-steering.md)
- [Wiring new models](../developer/wiring-new-models.md)
- [Testing and verification](../developer/testing-and-verification.md)
