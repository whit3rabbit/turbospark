---
description: "Overview of turbospark-runtime, the low-level prefill and decode loop that drives a logit producer."
icon: box
---

# turbospark-runtime

Use this crate when you need to control the generation loop's inputs yourself. You supply the tokenizer, the prompt token IDs, the sampling settings, and a producer that computes logits. The crate samples one token at a time, detokenizes the output, and stops on the rules you set. It does not render chat templates, load models for you, or return a finished `String`.

The loop reports text as it goes through a callback, then returns a summary when the run ends. Expect to write a small function around it, not a one-line call. The signatures in the examples were checked against source. This audit did not compile them.

Real model inference uses `RealForwardRunner`, which exists only on macOS. The loop, the `LogitProducer` trait, and the test mock are not behind a macOS gate in the crate root.

## Package and import

The package is `turbospark-runtime`, and Rust code imports it as `turbospark_runtime`. Add the package from crates.io once the version you need is published. Until then, use a path dependency or a git dependency, both pointing at a checkout:

```toml
[dependencies]
turbospark-runtime = { path = "../turbospark/crates/runtime" }
turbospark-tokenizer = { path = "../turbospark/crates/tokenizer" }
turbospark-selection = { path = "../turbospark/crates/selection" }
turbospark-core = { path = "../turbospark/crates/core" }
```

Add `turbospark-core` only when you name `LogitValue` in your own producer. `TokenId` is re-exported from `turbospark_runtime`. Some crates inside the workspace use short aliases such as `foundation` and `model_io`. Those aliases exist only in workspace source. Your code uses the package names above.

## What the crate contains

| Item | Use it for |
| --- | --- |
| [`run_raw_completion`](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/fn.run_raw_completion.html) | Prefill one prompt token at a time, then decode. |
| [`run_raw_completion_chunked`](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/fn.run_raw_completion_chunked.html) | Prefill in chunks through a `ChunkedPrefillRunner`. See [the producer page](custom-producer.md#chunked-prefill). |
| `run_raw_completion_cancellable` and `run_raw_completion_chunked_cancellable` | The same loops with a cancel check. |
| [`LogitProducer`](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/trait.LogitProducer.html) | The trait a model adapter implements. |
| [`ScriptedLogitProducer`](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/struct.ScriptedLogitProducer.html) | A fixed sequence of logits for tests of your own loop code. |
| [`RealForwardRunner`](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/struct.RealForwardRunner.html) | The GPU forward pass over a `.gturbo` install. macOS only. |
| [`GenerationConfig`](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/struct.GenerationConfig.html), [`RawDecodeProgress`](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/enum.RawDecodeProgress.html), [`RawDecodeResult`](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/struct.RawDecodeResult.html), [`StopReason`](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/enum.StopReason.html), [`RuntimeError`](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/enum.RuntimeError.html) | Inputs, progress events, the result, stop reasons, and errors. |

## Requirements

Four things must match before you call the loop. The loop checks some of them and trusts the rest.

- **The tokenizer and the model come from the same install:** load the tokenizer with `MfTokenizer::load_from_dir` from the directory that holds the weights, then encode prompts with it. Token IDs are `TokenId`, a signed 32-bit integer. IDs copied from a fixture or another model produce valid-looking output that is wrong.
- **`vocab_size` equals the length of the logits the producer writes:** the loop allocates a buffer of this length and does not check it against the producer. For `RealForwardRunner`, read `runner.vocab_size()`. Do not use `tokenizer.vocab_size`, because the runner's own docs note that the tokenizer value is a dialect constant and can differ from the checkpoint's row count.
- **The prompt and the budget fit the context:** the loop rejects an empty prompt, a `max_new_tokens` of zero, and any run where `prompt_ids.len() + max_new_tokens` exceeds `max_context`. The producer must also hold that many positions. `RealForwardRunner` sizes its KV cache when you open it, and generating past the cache is an error.
- **The sampling settings are valid:** `ShapingConfig::new` checks temperature, top-k, top-p, and the repetition penalty and returns an error for bad values. A temperature of `0.0` with a repetition penalty of `1.0` gives greedy decoding. `rate: Default::default()` leaves decoding uncapped.

## Generation loop

The function below works with any producer. It takes the vocabulary size as an argument for the reason given in the requirements above:

```rust
use std::io::Write;

use turbospark_runtime::{
    run_raw_completion, GenerationConfig, LogitProducer, RawDecodeProgress, RawDecodeResult,
    RuntimeError,
};
use turbospark_selection::ShapingConfig;
use turbospark_tokenizer::MfTokenizer;

/// Generates a completion for `prompt` and streams it to stdout.
///
/// `vocab_size` must equal the length of the logits the producer writes.
/// The loop does not check it.
fn generate(
    producer: &mut dyn LogitProducer,
    tokenizer: &MfTokenizer,
    prompt: &str,
    max_context: u32,
    vocab_size: usize,
) -> Result<RawDecodeResult, RuntimeError> {
    let prompt_ids = tokenizer.encode(prompt, true);
    let config = GenerationConfig {
        // Temperature 0 with no repetition penalty is greedy decoding.
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None)?,
        max_new_tokens: 256,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };
    let result = run_raw_completion(
        producer,
        tokenizer,
        &prompt_ids,
        &config,
        max_context,
        vocab_size,
        |event| match event {
            RawDecodeProgress::Token { delta, .. } | RawDecodeProgress::Tail(delta) => {
                print!("{delta}");
            }
            RawDecodeProgress::Prefill { .. } => {}
        },
    )?;
    let _ = std::io::stdout().flush();
    println!();
    Ok(result)
}
```

The function encodes the prompt with a beginning-of-sequence token when the dialect has one. The loop does not apply a chat template. If your model expects one, render the template to text first and encode that text.

{% hint style="warning" %}
Handle `RawDecodeProgress::Tail` as well as `Token`. The stop matcher holds back text that might still complete a stop string, and the loop sends that held text as a `Tail` event when the run stops. A callback that prints only `Token` drops the last characters of some replies.
{% endhint %}

A complete program that runs is on the [custom producer page](custom-producer.md). The macOS program that opens a real install is on the [RealForwardRunner page](real-forward-runner.md).

## Progress events

The callback receives one of three events:

- `Prefill { done, total }` reports prompt tokens processed so far.
- `Token { index, id, delta }` reports one sampled token. `index` counts from 0, `id` is the token ID, and `delta` is the text this token adds.
- `Tail(String)` carries text the stop matcher was holding back. The loop sends it only when the run stops with held text, so it arrives at most once and always last.

## Result

`RawDecodeResult` reports how the run ended:

- `reason` is the [`StopReason`](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/enum.StopReason.html) that ended the run.
- `prompt_tokens` and `new_tokens` count the prompt and the generated tokens. A stop token counts as a generated token even though it produces no text.
- `kv_position` is the cache cursor after the run. For a completed decode, it equals `prompt_tokens + new_tokens - 1`, because the loop never feeds the final sampled token back to the producer.
- `kv_backed_token_ids` lists the token IDs currently held in the KV cache. Compare it with the next prompt to see how much of that prompt the cache already covers.
- `reused_prefix_tokens` reports how many prompt tokens came from the producer's existing cache. It is 0 unless the producer opted in.
- `prefill_seconds` and `decode_seconds` are wall-clock times for the two phases.

The result also has `session_slot_evicted` and `peak_memory_pressure`. Both matter only to hosts that run several sessions or watch memory, and the [`RealForwardRunner` page](real-forward-runner.md) covers them where they apply.

## Stop reasons

| Variant | Meaning |
| --- | --- |
| `EndOfTurn` | The tokenizer's end-of-turn token was sampled. |
| `ToolCalls` | The tool-call stop token was sampled. |
| `Eos` | Another stop token was sampled, including end-of-sequence and any `extra_stop_tokens` you set. |
| `StopString` | The output matched one of `stop_strings`. |
| `MaxTokens` | `max_new_tokens` was reached. |
| `Cancelled` | The cancel check returned `true`. Only the cancellable entry points return this. |

## Errors

Every loop function returns `Result<RawDecodeResult, RuntimeError>`. The variants are:

| Variant | Cause |
| --- | --- |
| `EmptyPrompt` | `prompt_ids` is empty. |
| `ZeroBudget` | `max_new_tokens` is 0. |
| `ContextOverflow { prompt, max_new, max_context }` | `prompt + max_new` is larger than `max_context`. |
| `Selection(SelectionError)` | The sampler rejected a value. |
| `Producer(String)` | The producer returned an error. The run stops with this message. |
| `SpeculationUnavailable(String)` | Speculative decoding was requested for a setup that cannot serve it. Only the speculative entry point returns this. |

`RuntimeError` implements `Display` and `std::error::Error`, so `?` into `Box<dyn Error>` works.

## Cancellation

`run_raw_completion_cancellable` takes a `CancelFlag`, which is a `&dyn Fn() -> bool`, and polls it once per prefill token and once per decoded token. A cancelled run returns `Ok` with `StopReason::Cancelled`, not an error. The partial output and the KV fields stay accurate, so a caller can keep the partial turn. In the chunked variant, prefill is checked once per chunk, so a cancel during a long chunk waits for that chunk to finish.

## Platform support

- The loop, `LogitProducer`, `ChunkedPrefillRunner`, `ScriptedLogitProducer`, the config, and the result types are not behind a macOS gate in `lib.rs`.
- `RealForwardRunner`, the family-specific runners, and the vision, encoder, and speech modules are behind `cfg(target_os = "macos")`.
- Real inference needs a Metal-capable device. Build and example checks on macOS are not verified from source.

## Next

- [Implement a custom producer](custom-producer.md) to connect your own model or a test double.
- [Run a `.gturbo` install with RealForwardRunner](real-forward-runner.md) on macOS.
- [Rust library overview](../../rust.md) for the other crates.
- [Tokenizer reference](../tokenizer/README.md) for prompt rendering and token IDs.
- The [batched prefill notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/BATCHED_PREFILL.md) cover the chunked prefill design.
- The [speculative decoding notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/SPECULATIVE_DECODING.md) cover the speculative entry point, which this guide does not document.
- The [load guard notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/LOAD_GUARD.md) explain how the memory guard decides how much context fits.
