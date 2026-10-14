---
description: "How to implement the LogitProducer trait, and the optional chunked prefill trait, so turbospark-runtime can drive your own model."
icon: code
---

# Implement a custom producer

Implement `LogitProducer` when your model can run one token at a time and write the logits for the next position. Two methods are required, `reset` and `produce`. The other methods have defaults that you can keep until you need them. Add `ChunkedPrefillRunner` only when your model can process a whole prompt chunk in one call and produce the same state as the token-at-a-time path.

A producer does not sample. It writes logits, and the loop owns temperature, top-k, top-p, the repetition penalty, and the stop rules. The project rule is that softmax lives in the sampler only, so the producer returns raw logits and never normalizes them.

## The trait

The signatures below are abbreviated. The [trait source](https://github.com/whit3rabbit/turbospark/blob/main/crates/runtime/src/producer.rs) has the full doc comments.

```rust
pub trait LogitProducer {
    // Required.
    fn reset(&mut self);
    fn produce(&mut self, token: i32, position: usize, logits: &mut [LogitValue])
        -> Result<(), String>;

    // Defaults.
    fn produce_prefill(&mut self, token: i32, position: usize, scratch: &mut [LogitValue])
        -> Result<(), String>; // calls produce
    fn try_reuse_prefix(&mut self, prompt_ids: &[TokenId]) -> usize; // returns 0
    fn session_slot_evicted(&self) -> bool; // returns false
}
```

The loop calls these in a fixed order. It calls `try_reuse_prefix` first. It calls `reset` unless that call returned a nonzero count. It then calls `produce_prefill` for each prompt token except the last, and `produce` for the last prompt token and every generated token that continues the run.

Each call uses `position` as the index where the token is written into the cache. The first call in a fresh run passes `position` 0, and each call after that passes one more than the last.

Three rules follow from that order:

- Write every one of the `vocab_size` logits on each `produce` call. The loop reads the whole buffer.
- Treat `reset` as the start of a new generation. Clear the KV cursor and any per-generation state there.
- Return `Err` for any failure. The loop stops the run and reports `RuntimeError::Producer`.

## Example

This program runs against a stand-in producer that has no model. It favors the token ID after the previous one. The `position` check makes the contract explicit. If the loop ever passed a position out of step with the cursor, the run would stop with a producer error.

```rust
use std::path::Path;

use turbospark_core::{LogitValue, TokenId};
use turbospark_runtime::{run_raw_completion, GenerationConfig, LogitProducer, RawDecodeProgress};
use turbospark_selection::ShapingConfig;
use turbospark_tokenizer::MfTokenizer;

/// Favors the token id after the previous one. There is no model behind it.
/// It shows the contract: `position` is the KV index of the token being fed,
/// and it counts up from 0.
struct NextIdProducer {
    vocab_size: usize,
    next_position: usize,
}

impl LogitProducer for NextIdProducer {
    fn reset(&mut self) {
        self.next_position = 0;
    }

    fn produce(
        &mut self,
        token: TokenId,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        if position != self.next_position {
            return Err(format!(
                "position {position} does not match the cursor {}",
                self.next_position
            ));
        }
        self.next_position += 1;
        let favored = (token as usize + 1) % self.vocab_size;
        for (id, slot) in logits.iter_mut().enumerate() {
            *slot = LogitValue::from_f32(if id == favored { 1.0 } else { 0.0 });
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("pass a directory that holds tokenizer files")?;
    let tokenizer = MfTokenizer::load_from_dir(Path::new(&dir))?;
    let vocab_size = tokenizer.vocab_size;

    let prompt_ids = tokenizer.encode("hi", true);
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None)?,
        max_new_tokens: 8,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };
    let mut producer = NextIdProducer {
        vocab_size,
        next_position: 0,
    };

    let mut text = String::new();
    let result = run_raw_completion(
        &mut producer,
        &tokenizer,
        &prompt_ids,
        &config,
        4096,
        vocab_size,
        |event| match event {
            RawDecodeProgress::Token { delta, .. } | RawDecodeProgress::Tail(delta) => {
                text.push_str(&delta)
            }
            RawDecodeProgress::Prefill { .. } => {}
        },
    )?;
    println!(
        "prompt_tokens={} new_tokens={} reason={:?} kv_position={}",
        result.prompt_tokens, result.new_tokens, result.reason, result.kv_position
    );
    println!("text={text:?}");
    Ok(())
}
```

Save the program in a binary crate that depends on the packages above, then run it against the ChatML test fixture in the repo (from the repo root):

```sh
cargo run -- crates/runtime/tests/fixtures/ChatMLTokenizer
```

This audit did not run the program, so the printed values are not verified. The `kv_position` arithmetic follows from the loop: the stop check runs before each feed, so with eight new tokens stopped at `MaxTokens`, `kv_position` is `prompt_tokens + 7`. The eighth generated token is never fed back.

The example uses the tokenizer's `vocab_size` because this producer has no model of its own. A real producer must write the model's logits length instead, as explained in the [overview requirements](README.md#requirements).

## Prefix reuse

`try_reuse_prefix` lets a producer keep the KV cache from the previous turn. It returns how many leading prompt tokens the cache already covers, and the loop skips feeding those tokens. The default returns 0, which means no reuse and a full reset every run.

{% hint style="danger" %}
A wrong reuse answer does not crash. It continues the next turn from a cache that belongs to a different conversation, and the output looks fluent. Key the count on the token IDs the producer actually fed, and return 0 whenever the cache cannot prove the match. Injected image embeddings are the case where IDs do not describe the state, so a producer that accepts such input must return 0 for it.
{% endhint %}

The loop clamps the returned count to `prompt_ids.len() - 1`, so at least one token is always fed. The loop does not call `reset` when the count is nonzero, so the producer owns the cursor from that point.

## Scripted producer for tests

`ScriptedLogitProducer::new(steps)` takes a `Vec<Vec<LogitValue>>` and returns one step per call, in order. It returns an error once the steps run out. Use it to test your own code that reads `RawDecodeResult` or the progress events, without a model.

Every prompt token consumes one step, because the loop calls `produce_prefill` or `produce` for each one. A run over an N-token prompt that samples M tokens consumes N + M - 1 steps in total. The first N steps feed the prompt, and the rest feed the sampled tokens back. The repo's [raw completion test](https://github.com/whit3rabbit/turbospark/blob/main/crates/runtime/tests/raw_completion.rs) shows a complete script.

## Chunked prefill

`ChunkedPrefillRunner` extends `LogitProducer` with a call that processes a whole prompt chunk at once:

```rust
pub trait ChunkedPrefillRunner: LogitProducer {
    fn prefill_chunk(&mut self, tokens: &[i32], start_position: usize,
        logits: &mut [LogitValue]) -> Result<(), String>;
    // Default: calls prefill_chunk and ignores the flag.
    fn prefill_chunk_with_status(&mut self, tokens: &[i32], start_position: usize,
        logits: &mut [LogitValue], is_final_prompt_chunk: bool) -> Result<(), String>;
}
```

Pass the producer to `run_raw_completion_chunked` with a chunk size. The loop splits the prompt into spans, hands each span to `prefill_chunk_with_status`, and then decodes token by token. Chunk sizes should come from `turbospark_core::ALLOWED_CHUNK_SIZES`, which is `[32, 64, 128, 256, 512, 1024, 2048, 4096]`. The default chunk size is 128.

The bar for a chunked implementation is identity with the sequential path, not agreement with another chunked run. After a chunk, the logits, and the cache state, must match what `produce_prefill` over the same tokens would leave behind. A chunked path can agree with itself and still be wrong, so compare it against the token-at-a-time loop. The repo's [chunked prefill tests](https://github.com/whit3rabbit/turbospark/blob/main/crates/runtime/tests/chunked_prefill.rs) show that comparison.

{% hint style="info" %}
`run_raw_completion_chunked` does not fall back to token-at-a-time prefill when the producer refuses a chunk. A refusal arrives as `RuntimeError::Producer`. `RealForwardRunner` refuses unsupported model families this way, so check its `supports_chunked_prefill()` first. See the [RealForwardRunner page](real-forward-runner.md#chunked-prefill).
{% endhint %}

## Next

- [Run a `.gturbo` install with RealForwardRunner](real-forward-runner.md) for the model-backed producer on macOS.
- [Overview](README.md) for the loop's requirements, results, and errors.
