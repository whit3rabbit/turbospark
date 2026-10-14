---
description: "How to open a .gturbo install with RealForwardRunner on macOS and run it through the turbospark-runtime generation loop."
icon: microchip
---

# RealForwardRunner on macOS

`RealForwardRunner` is the model-backed `LogitProducer`. It opens a `.gturbo` install directory, runs the forward pass on the GPU, and returns logits to the same loop functions that take any other producer. It exists only on macOS, and it needs a Metal-capable device.

Open the runner once and reuse it across generations. Opening allocates the KV cache for your context limit and the activation scratch up front, so repeated opens cost memory and time.

## Requirements

- macOS with a Metal-capable device. The `turbospark-gpu` dependency is declared only for macOS in the crate manifest, so the runner cannot build on other targets.
- A `.gturbo` install directory. It must hold `manifest.json`, the weights, and the tokenizer files. Getting an install is outside this crate. See the [Rust library overview](../../rust.md) for the catalog crate.
- Enough memory for the context you choose. The KV cache is allocated when the runner opens, so a large `max_context` costs memory before the first token.

You also need `turbospark-repack` for `peek_manifest_arch`, which reads the architecture from the manifest. Add it next to the packages in the [overview](README.md#package-and-import):

```toml
turbospark-repack = { path = "../turbospark/crates/repack" }
```

## Open and generate

The example below opens an install, encodes a prompt, and streams the reply. Pass the install directory as the first argument. It builds only on macOS.

```rust
// Builds only on macOS. The crate exports RealForwardRunner only there.
use std::path::Path;

use turbospark_repack::peek_manifest_arch;
use turbospark_runtime::{
    run_raw_completion, run_raw_completion_chunked, GenerationConfig, RawDecodeProgress,
    RawDecodeResult, RealForwardRunner,
};
use turbospark_selection::ShapingConfig;
use turbospark_tokenizer::MfTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let install = std::env::args()
        .nth(1)
        .ok_or("pass a .gturbo install directory")?;
    let dir = Path::new(&install);
    // Keep this equal to the max_context passed to the loop below.
    let max_context: usize = 4096;

    let arch = peek_manifest_arch(dir)?;
    let tokenizer = MfTokenizer::load_from_dir(dir)?;
    let mut runner = RealForwardRunner::open_with_max_context(dir, arch, max_context)?;
    // The model's own logits row count. Do not substitute tokenizer.vocab_size.
    let vocab_size = runner.vocab_size();

    let prompt_ids = tokenizer.encode("Write one sentence about rivers.", true);
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None)?,
        max_new_tokens: 64,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };
    let on_progress = |event: RawDecodeProgress| {
        if let RawDecodeProgress::Token { delta, .. } = event {
            print!("{delta}");
        }
    };
    let result: RawDecodeResult = if runner.supports_chunked_prefill() {
        run_raw_completion_chunked(
            &mut runner,
            &tokenizer,
            &prompt_ids,
            &config,
            max_context as u32,
            vocab_size,
            128,
            on_progress,
        )?
    } else {
        run_raw_completion(
            &mut runner,
            &tokenizer,
            &prompt_ids,
            &config,
            max_context as u32,
            vocab_size,
            on_progress,
        )?
    };
    println!("\nnew_tokens={} reason={:?}", result.new_tokens, result.reason);
    Ok(())
}
```

The tokenizer and the runner must come from the same directory. The loop does not check that they match, so a mismatch shows up as wrong text rather than an error.

{% hint style="warning" %}
The context limit appears twice. `open_with_max_context` sizes the KV cache, and the loop's `max_context` argument checks each request against that limit. Keep the two equal. A larger loop value passes the admission check and then fails once generation runs past the cache.
{% endhint %}

## Choosing an open function

The runner has several open functions. They differ in what they read from the environment and what they let you set.

| Function | Context | Speculation | Steering | Notes |
| --- | --- | --- | --- | --- |
| `open(dir, arch)` | 4096 | Read from the environment | Off | Uses the default context. |
| `open_with_max_context(dir, arch, max_context)` | Your value | Read from the environment | Off | The example above. |
| `open_with_residency(dir, arch, max_context, slots, speculation, steering, session_slots, kv_quant, residency)` | Your value | Your value | Your value | Explicit control. The CLI and server use this one. |

`open` and `open_with_max_context` read `TURBOSPARK_MTP_DRAFT` and `TURBOSPARK_DFLASH_DRAFT` from the environment. They also let the runner choose expert residency on its own. `open_with_options` pins speculation, steering, and KV quantization off and takes only a slot count, so it is the shorter explicit form when you do not need those features.

{% hint style="warning" %}
A shell that exports `TURBOSPARK_MTP_DRAFT` or `TURBOSPARK_DFLASH_DRAFT` changes what `open_with_max_context` loads. Use `open_with_residency` with `DraftPolicies::off()` and `SteeringPolicy::off()` when your process must behave the same in every environment. Those types are exported from `turbospark_runtime`, and the `open_with_residency` signature in the [crate reference](https://docs.rs/turbospark-runtime/latest/turbospark_runtime/struct.RealForwardRunner.html) lists the rest.
{% endhint %}

## Chunked prefill

`run_raw_completion_chunked` prefills the prompt in chunks rather than one token at a time. The runner accepts it for some model families and refuses it for others. The refusal arrives as a `RuntimeError::Producer` error, not as a silent fallback, so check first with `supports_chunked_prefill()`.

The answer depends on the install. For dense Qwen, the runner reports chunked support only when no multi-token-prediction head and no DFlash drafter are loaded. Chunk sizes should come from `turbospark_core::ALLOWED_CHUNK_SIZES`. The example uses 128, which is in that list.

## Multi-turn reuse

`set_prefix_reuse(true)` lets the runner keep the KV cache from the previous turn. The next prompt then only prefills the part that differs from that cache. It is off by default, because a reusing runner is not reset between generations.

Turn reuse on only for a caller that sends the full transcript on every turn. The loop still checks the token IDs against the cache, and it resets the runner when they do not match. The [custom producer page](custom-producer.md#prefix-reuse) explains the matching rule.

## Errors

Opening returns `RealForwardError`. Its variants are:

- `Model(ModelError)` for install and manifest problems.
- `Gpu(GpuError)` for Metal device and allocation failures.
- `MissingTensor(String)` when a weight the model needs is not in the install.
- `Unsupported(String)` when the install uses a feature this runner does not support.

Generation returns `RuntimeError`. Failures inside the forward pass arrive as `RuntimeError::Producer` with the runner's message. The [overview errors table](README.md#errors) lists the rest.

## Other entry points

The runner also has vision methods, steering controls, speculative decoding, and phase counters. This page does not cover them. Speculative decoding has its own loop, `run_raw_completion_speculative`, and the [speculative decoding notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/SPECULATIVE_DECODING.md) describe its requirements. The [MTP notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/MTP.md) cover the multi-token-prediction head.

## Next

- [Overview](README.md) for the loop's requirements, results, and errors.
- [Implement a custom producer](custom-producer.md) if you need a model path other than the runner.
