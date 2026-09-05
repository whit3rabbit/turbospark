---
title: "Rust API: Core Primitives and CPU Reference"
description: "Public API of the foundation crates (turbospark-core, turbospark-compute, turbospark-selection, turbospark-window-fit, turbospark-tokenizer, turbospark-invocation), generated from cargo doc and source doc comments."
diataxisType: "reference"
---

<!-- generated: rust lane, signal: Cargo.toml -->

Workspace: 17 crates, edition 2021, MSRV 1.82. This page covers the
foundation tier: the leaf primitives crate, the CPU reference kernels, the
sampler, the window fitter, the tokenizer, and the pure CLI parser. All
signatures below are read from the source; doc text is quoted or condensed
from `///` comments. Run `cargo doc --no-deps` and open `target/doc/` for
the full rustdoc inventory (all 17 crates build there).

## turbospark-core (`crates/core`)

Core leaf crate: shared primitives and the public runtime configuration.
Downstream crates depend on it under an alias (`foundation = { package =
"turbospark-core", path = "../core" }`) to avoid colliding with the
standard library `core` in the extern prelude. 69/69 public items
documented.

Public modules: `chunk_sizing` (prefill chunk sizing and automatic chunk
size resolution), `error` (error types for foundation operations),
`prefill` (chunking strategies and runtime prefill configuration),
`primitives` (scalar primitives, type aliases, logit views),
`runtime_config` (runtime engine configuration and options builder),
`steering` (the directional-steering edit's mode, shared by the CPU
reference and the Metal dispatch).

| Symbol | Kind | Signature / doc |
|---|---|---|
| `TokenId` | type | `pub type TokenId = i32;` Token id interchange type. Signed 32-bit integer to match the downstream buffer element width. |
| `LogitValue` | type | `pub type LogitValue = Half;` Half-precision logit element (IEEE-754 binary16). |
| `RuntimeConfig` | struct | Immutable runtime configuration assembled from overrides plus defaults. |
| `Error` | enum | Shared root error. |
| `invalid_argument` / `internal` | fn | Builders for the two error kinds, from any string-like value. |
| `InputLength` | enum | Known or not-yet-known length of the input a chunk size is being chosen for. |
| `resolve_automatic_chunk_size` | fn | Resolve an automatic chunk-size request to one concrete allowed size; three-state resolution over the allowed set. |
| `PrefillError` | enum | Errors occurring during chunked-prefill configuration or execution. |
| `PrefillChunkSpan` | struct | One chunk of a chunked prefill plan. |
| `prefill_chunk_spans` | fn | Splits `token_count` tokens starting at `start_position` into spans of at most `chunk_tokens` tokens each, in order. |

## turbospark-compute (`crates/compute`)

`#![forbid(unsafe_code)]`. Destination-selected compute strategy plus CPU
reference kernels: the kernel modules are the numerical ground truth later
GPU kernels are validated against. Numerics parity with any upstream
implementation is out of scope for `ComputeStrategy` itself. 174/175 public
items documented.

Public modules (each re-exported at the crate root): `attention`, `gating`,
`gdn` (gated-DeltaNet linear attention), `hyper_connection`, `moe`, `ple`,
`qsa_indexer`, `quant`, `quant_1bit`, `quant_2bit`, `quant_gguf`,
`quant_gguf_iq`, `quant_gguf_iq_tables`, `quant_gguf_mxfp4`, `rms_norm`,
`rope`, `sampling`, `steering`, `tolerance`, `vision`, `wht`.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `ComputeStrategy` | struct | Marker for the destination-selected compute strategy. |
| `new` | fn | `pub fn new() -> Self`. Construct the default compute strategy (`crates/compute/src/lib.rs`). |
| `causal_attention` | fn | FP32 causal attention reference, ported from `Support/Reference/Attention/Attention.swift`; materializes the full attention row per Q head. |
| `causal_attention_with_sinks` | fn | The same, with attention sinks: one learned logit per query head that joins the softmax denominator and nothing else. |
| `indexed_attention` | fn | Attention over an explicit subset of key/value positions: the CPU reference for `qwen4_exp`'s QSA attention application. |
| `silu` / `sigmoid` | fn | Activation references (`x / (1 + exp(-x))` and `1 / (1 + exp(-x))`) in `gdn`. |
| `sigmoid_gate_mul` | fn | `out[i] *= sigmoid(gate[i])`. Qwen's full-attention layers gate the attention output per element. |
| `sigmoid_scalar_mul` | fn | `y[i] *= sigmoid(gate)`. One scalar logit gates the whole shared-expert output. |
| `split_q_gate` | fn | Splits a `[heads, 2 * dim]` packed projection into contiguous `[heads, dim]` query and gate halves. |

## turbospark-selection (`crates/selection`)

Choose exactly one candidate identifier from a per-candidate score vector
under a validated shaping configuration, an accumulated history, and a step
position. Numeric parity with any upstream implementation is out of scope;
only the observable input, output, ordering, and error contract is
exercised. The internal random-number algorithm is likewise out of scope.
35/35 public items documented.

Public modules: `choose`, `derive`, `penalty`, `shaping`, `truncation`.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `select` | fn | `pub fn select(scores: LogitsView<'_>, config: &ShapingConfig, history: &[TokenId], position: u64) -> Result<TokenId, SelectionError>`. Select exactly one candidate identifier; at the deterministic temperature value this always agrees with the single highest-scoring candidate under the raw score vector. |
| `ShapingConfig` | struct | A plain, caller-constructed shaping configuration; carries no state between selections. |
| `new` | fn | `ShapingConfig::new` builds and validates a shaping configuration. `top_k` of zero disables rank-based truncation; `top_p` of `None` disables probability-mass truncation. |
| `SelectionError` | struct | Shaping configuration for candidate selection, validated at construction; carries no hidden state. |
| `apply_repetition_penalty` | fn | Attenuate the score of every distinct candidate identifier present in `history` exactly once, regardless of how often it repeats. |
| `apply_presence_and_frequency_penalty` | fn | OpenAI-style presence and frequency penalties over the generated suffix of history only. |
| `derive_step_value` | fn | Deterministic per-position value derivation from an optional seed. |
| `to_unit_interval` | fn | Produce a value in `[0.0, 1.0)` from a 64-bit derived value, for use as a uniform categorical draw. |

## turbospark-window-fit (`crates/window-fit`)

Pure, deterministic conversation-window fitting: drops the oldest eligible
turns from a conversation until a caller-supplied measurement of the whole
remaining conversation is under a caller-supplied bound or nothing eligible
remains. No I/O, no state between calls. 7/8 public items documented.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `fit_conversation_window` | fn | `pub fn fit_conversation_window<T, M>(turns: &[T], has_leading_instruction: bool, bound: u64, mut measure: M) -> WindowFitOutcome<T> where T: Clone, M: FnMut(&[T]) -> u64`. Fit a conversation into a measured length bound; an optional leading instruction turn and the newest turn are never removed. |
| `WindowFitOutcome` | struct | The reported outcome of a conversation-window fit; carries the retained turns in their original order. |
| `retained_turns` | fn | The retained turns, in their original conversation order. |
| `measured_length` | fn | The length measured over the retained conversation, from the last measurement taken. |
| `removed_turn_count` | fn | How many turns were removed to reach this outcome. |
| `has_room_for_generation` | fn | Whether the retained conversation leaves room for at least one more generated unit under the bound that was fitted against. |

## turbospark-tokenizer (`crates/tokenizer`)

`#![forbid(unsafe_code)]`. Tokenizer wrapper, chat-template rendering,
streaming detokenizer, and tool-call parsing for the Gemma 4, ChatML
(Qwen), and DeepSeek-V4 model dialects. Ported from
`Sources/Mference/Tokenization` and
`Runtime/Generation/StreamingStopMatcher.swift`. 68/76 public items
documented.

The crate root re-exports: `ContentPart`, `FunctionDefinition`,
`HistoricalToolCall`, `Message`, `Role` (from `chat_template`), plus the
`detokenizer`, `dialect`, `error`, `jinja_chat_template`, `json_value`,
`reasoning`, `stop_matcher`, `structured_decoder`, and `tool_call` module
surfaces.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `MfTokenizer` | struct | Tokenizer wrapper combining Hugging Face's `tokenizers` backend with chat dialect detection, special token ID mapping, and Jinja chat templates (`dialect/mod.rs`). |
| `Role` | enum | Message sender role discriminator for chat templates. |
| `Message` | struct | Single chat message in a conversation sequence. |
| `new` | fn | Constructs a chat message with a role and text content string. |
| `with_parts` | fn | Constructs a multimodal message from ordered content parts; `content` is set to the parts' text joined. |
| `ContentPart` | enum | One ordered piece of a multimodal message's content. An image is a placeholder here and carries no pixels. |
| `FunctionDefinition` | struct | Function schema definition for tool use. |
| `HistoricalToolCall` | struct | Recorded historical tool call invocation in a chat turn. |
| `image_count` | fn | Images this message carries, in order. |

## turbospark-invocation (`crates/invocation`)

Pure translation of an ordered list of command-line argument tokens into a
validated invocation request, a help short-circuit, or one of six
distinguishable typed failures. No filesystem, network, environment, or
process access; no state between calls. 44/44 public items documented.

Public modules: `diagnostics`, `failure`, `options`, `parser`, `request`,
`usage`.

| Symbol | Kind | Signature / doc |
|---|---|---|
| `InvocationRequest` | struct | A fully populated, validated invocation (`request.rs`). |
| `ParseOutcome` | enum | One of the three possible outcomes of parsing a token list; there is no fourth outcome and no partially populated result. |
| `ParseFailure` | enum | The closed set of six distinguishable typed parsing failures; each variant carries the offending option name. |
| `OptionDecl` | struct | Single declaration table for every recognized option spelling; both the token scan (`parser`) and the usage renderer (`usage`) read this table. |
| `ExitStatus` | enum | The two distinct process exit statuses this unit ever produces. |
| `code` | fn | The documented raw process exit code for this status. |
| `StreamRouting` | struct | Where an outcome's text belongs: the primary stream, the diagnostic stream, or nowhere. |
| `exit_status` / `stream_routing` | fn | Map a parse outcome to its exit status / stream routing; help and success both map to the success status. |

## Coverage for this tier

Per-crate public-item doc coverage (excluding `pub use` re-exports, parsed
from `crates/*/src`): core 69/69, compute 174/175, selection 35/35,
window-fit 7/8, tokenizer 68/76, invocation 44/44. The full undocumented
list for the workspace is reported in the lane coverage output that
accompanies these pages.
