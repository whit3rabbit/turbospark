---
description: >-
  Plan chunked prompt prefill and track uncommitted KV writes with
  turbospark-core.
icon: box
---

# Chunking and prefill

Use this page when your code splits a prompt into spans before a prefill. The crate picks a chunk size, plans the spans, and records whether a failed chunk left KV rows uncommitted. It does not run the forward pass. The GPU scratch buffer sizing that consumes these spans lives in `turbospark-gpu`, and this crate does only the span and state bookkeeping.

Start with the [overview](./) if you have not read it yet. It covers the package name, the import path, and the error type.

## Pick a chunk size

`resolve_automatic_chunk_size` takes an `InputLength` and returns one value from `ALLOWED_CHUNK_SIZES`. It has three cases:

* `InputLength::Known(n)` where some allowed size covers `n` returns the smallest such size.
* `InputLength::Known(n)` above 4096 returns 4096.
* `InputLength::Unknown` returns `DEFAULT_CHUNK_SIZE`, which is 128.

The function returns a `u32`, and the span planner takes a `usize`. Convert at the call site, as the example below does. A known length of 0 resolves to 32.

## Plan the spans

`prefill_chunk_spans(token_count, start_position, chunk_tokens)` returns a `Vec<PrefillChunkSpan>` in prompt order. An empty prompt returns an empty vector. Each span has four fields:

* `token_offset` is the index of the span's first token within the prompt.
* `token_count` is the number of tokens in the span.
* `start_position` is the absolute KV position of the span's first token. It equals the `start_position` argument plus `token_offset`.
* `completed_count` is the cumulative token count after the span.

{% hint style="warning" %}
The planner clamps `chunk_tokens` to the range 1 through 4096. It does not check the value against `ALLOWED_CHUNK_SIZES`. A value of 100 passes through unchanged, and a value of 0 becomes 1, which produces one-token spans without an error. Resolve or validate the chunk size before you call the planner.
{% endhint %}

```rust
use turbospark_core::{
    prefill_chunk_spans, resolve_automatic_chunk_size, InputLength, PrefillChunkCommitState,
};

fn main() {
    let prompt_tokens: usize = 300;
    let chunk = resolve_automatic_chunk_size(InputLength::Known(prompt_tokens as u64)) as usize;
    let spans = prefill_chunk_spans(prompt_tokens, 0, chunk);
    println!("chunk {chunk}, {} spans", spans.len());

    let mut state = PrefillChunkCommitState::new();
    for span in &spans {
        state.mark_dirty(span.start_position, span.token_count);
        // Write the KV rows for this span here. Commit only after they land.
        state.mark_committed();
    }
    state.require_clean("decode").expect("runner is clean");

    // A chunk that fails mid-write leaves the runner dirty until reset().
    let mut failed = PrefillChunkCommitState::new();
    failed.mark_dirty(0, 32);
    match failed.require_clean("prefill") {
        Ok(()) => println!("unexpected: clean"),
        Err(err) => println!("rejected: {err}"),
    }
    failed.reset();
    failed.require_clean("prefill").expect("clean after reset");
}
```

For this 300-token prompt, the resolver picks 512, so the planner returns one span. A longer prompt produces several spans, each at most 512 tokens.

## Track commit state

`PrefillChunkCommitState` records whether a chunk wrote KV rows without committing them. A runner that errors mid-chunk must not be reused until the state is clean again. Use the state in this order:

1. Call `mark_dirty(start_position, token_count)` before you write a chunk's KV rows.
2. Call `mark_committed()` after the rows land.
3. If the write fails, leave the state dirty.
4. Call `require_clean("operation")` before you reuse the runner. It returns `PrefillError::ChunkedRunnerDirty` if the state is dirty.
5. Call `reset()` once the failure is handled. Only `mark_committed()` and `reset()` clear the state.

`is_dirty()` and `in_flight_end_position()` report the state without changing it. The end position is the start plus the token count, and it is `None` when the state is clean.

{% hint style="warning" %}
`mark_dirty` panics when `token_count` is 0. Check for an empty span before you mark it, or skip the call.
{% endhint %}

## Limits

`MAX_CHUNK_TOKENS` is 4096 as a `usize`. It is derived from the largest value in `ALLOWED_CHUNK_SIZES`, so a wider allowed set raises it. Chunked prefill re-reads each layer's routed experts once per chunk, so expert reads scale with prompt length divided by chunk size. The ceiling exists because past a point, that re-read cost dominates.

The planner and resolver are pure functions. They do no I/O, allocate only the span vector, and need no GPU. Run them on any target the crate builds for.

## Where to go next

* [Overview](./) for the shared types, the error type, and the allowed sets.
* [Rust library overview](../) for how the other crates fit around the runtime.
* [API reference on docs.rs](https://docs.rs/turbospark-core/latest/turbospark_core/) for every signature in `chunk_sizing` and `prefill`.
