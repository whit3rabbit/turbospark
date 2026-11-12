---
description: "The C ABI over the inference engine for the Swift app and other native hosts: opaque handles, panic-guarded entry points, camelCase JSON wire shapes, and streaming callbacks."
icon: plug
---

# ffi

`crates/ffi` (package `turbospark-ffi`) is the C ABI layer between the Rust
engine and native GUI hosts. It builds as both a `staticlib` (the
`libturbospark_ffi.a` archive SwiftPM links) and an `rlib` (what this crate's
own tests link, so its tests exercise the same function bodies the C header
declares). The canonical surface is the hand-written header
`crates/ffi/include/turbospark.h`; the Swift package at `swift/TurboSpark`
compiles against a staged copy of it, and the macOS app drives the engine
through that package in-process, with no IPC.

## What it does

Every exported entry point lives under `crates/ffi/src/api/`, one module per
domain. Support modules carry the machinery they share: `src/abi.rs` (status
codes, the per-thread error slot, and the panic guard), `src/strings.rs`
(borrowed `const char *` in, owned `char *` out), `src/wire.rs` (the camelCase
JSON request and result shapes), and the opaque handle types
(`src/session.rs`, `src/image_session.rs`, `src/stt_session.rs`,
`src/server.rs`).

| `api/` module | Domain |
| --- | --- |
| `core.rs` | ABI version, build info, last-error read, string free, system telemetry, peak footprint |
| `session.rs` | Session open, close, cancel; session info and decode phase reports; vision tower release |
| `generate.rs` | `ts_generate` and `ts_humanize`, token counts, prompt rendering, context-window fitting |
| `tokenizer.rs` | A tokenizer opened on its own, without opening a model (text-level calls only) |
| `models.rs` | Catalog, probe, recommend, install/cancel/pause/resume, HF token and endpoint, store root, repo variants |
| `image.rs` | Native image generation sessions, catalog, install, owned PNG buffers |
| `ocr.rs` | OCR document sessions: add pages, read, cancel |
| `embedding.rs` | Embedding encode, probe, install, and cosine similarity |
| `audio.rs` | Audio model catalog, installs, sessions and jobs, the heavy-work gate |
| `audio_stream.rs` | Live streaming transcription sessions |
| `stt.rs` | Resident speech-to-text models and PCM streams (macOS; refused by name elsewhere) |
| `server.rs` | In-process HTTP server sharing already-open models |
| `daemon.rs` | Managed background daemon status, start, stop, restart |

Session open options carry the whole performance surface over the wire as
camelCase JSON. The 0.5.0 additions: `anePrefill` (`auto` | `split` | `off` |
`break`; absent means `auto`, which activates the ANE+Steel prefill split on
Apple Silicon M4+ for dense Qwen and degrades to Steel when the programs are
missing or the chip is older) with `anePrefillDir`, and
`speculation: "adaptive"` or `"adaptive:CAP"` (the same controller and the
same spellings `--speculative adaptive[:MAX]` takes in `turbospark-check`;
blocks and the cap bounds are read from `foundation`, so the front ends
cannot drift apart).
`SessionInfo` reports both resolutions: `anePrefill` with `active`, the
resolved mode, and whether the family and host support the split, and
`speculation` with why it is off when the caller might have expected it on.

## Where it sits

`turbospark-ffi` sits on top of nearly the whole workspace. Its unconditional
dependencies are `core`, `compute`, `selection`, `tokenizer`, `model-io`,
`runtime`, `server`, `catalog`, `image`, `vision-io`, and `window-fit`;
`repack` and `bench` compile only under `cfg(target_os = "macos")` (model
intake, and the footprint sampler the frozen benchmark rows use). The `audio`
crate is a dev-dependency only, for the STT end-to-end test. The crate sets
`publish = false`: nothing in the workspace depends on it. Its only consumer
is the SwiftPM package `swift/TurboSpark`, whose `CTurboSpark` target links
the archive, and the macOS app through that package.

## Public API tour

The contract, from the header and `src/lib.rs`, in four rules:

1. Errors. Every fallible call returns `TS_OK` (0) or a non-zero `TS_ERR_*`
   code. On a non-zero return, `ts_last_error` on the SAME thread holds a
   message. Read it before making another call on that thread.
2. Ownership. A `const char *` argument is borrowed for the call and never
   retained. A `char **` out-parameter receives an allocation the caller
   returns through `ts_string_free`. Nothing hands back a pointer into the
   library's own state.
3. JSON. Options and results travel as camelCase JSON strings, so adding a
   knob is never an ABI break. The per-token stream carries no JSON: it is a
   pointer and a length.
4. Threading. A session is single-threaded; `ts_generate` blocks for the
   whole turn, so call it from a background thread. The one exception is
   `ts_session_cancel`, safe from any thread and never blocks.

The header reports `TS_ABI_VERSION` (currently 2); a host compares it with
`ts_abi_version()` once at startup. Event kinds are `TS_EVENT_PREFILL`,
`TS_EVENT_CONTENT`, `TS_EVENT_REASONING`, `TS_EVENT_TOOL`, and
`TS_EVENT_FINISH`; install progress uses `TS_INSTALL_STAGE` and
`TS_INSTALL_BYTES`, whose byte callback may fire concurrently from worker
threads. A host treats an unknown event kind as a no-op: new kinds are
appended by newer libraries and that is not a version break.

Representative entry points, all prefixed `ts_`:

| Group | Entry points | Source |
| --- | --- | --- |
| Lifecycle and errors | `ts_abi_version`, `ts_build_info_json`, `ts_last_error`, `ts_string_free`, `ts_system_info_json`, `ts_peak_footprint_bytes` | `crates/ffi/src/api/core.rs` |
| Sessions | `ts_session_open`, `ts_session_close`, `ts_session_cancel`, `ts_session_info_json`, `ts_session_phases_json`, `ts_session_release_vision` | `crates/ffi/src/api/session.rs` |
| Generation | `ts_generate`, `ts_humanize`, `ts_session_count_tokens_with_tools`, `ts_session_render_prompt`, `ts_session_fit_window_with_tools_json` | `crates/ffi/src/api/generate.rs` |
| Standalone tokenizer | `ts_tokenizer_open`, `ts_tokenizer_count_text_tokens`, `ts_tokenizer_tokenize_json` | `crates/ffi/src/api/tokenizer.rs` |
| Model management | `ts_catalog_json`, `ts_probe_json`, `ts_recommend_json`, `ts_install`, `ts_install_cancel`, `ts_installed_json`, `ts_hf_token_set`, `ts_store_root_get` | `crates/ffi/src/api/models.rs` |
| Images, OCR, embeddings | `ts_image_session_open`, `ts_image_generate`, `ts_ocr_open`, `ts_ocr_read_json`, `ts_embedding_encode_json`, `ts_cosine_similarity` | `crates/ffi/src/api/image.rs`, `ocr.rs`, `embedding.rs` |
| Audio and speech | `ts_audio_session_open`, `ts_audio_job_run`, `ts_audio_stream_open`, `ts_stt_open`, `ts_stt_stream_append` | `crates/ffi/src/api/audio.rs`, `audio_stream.rs`, `stt.rs` |
| Servers | `ts_server_start`, `ts_server_attach_session`, `ts_server_stop`, `ts_daemon_status_json` | `crates/ffi/src/api/server.rs`, `daemon.rs` |

Off macOS, engine entry points such as `ts_session_open` fail with
`TS_ERR_UNSUPPORTED` and a sentence, while the catalog, probe, and install
calls work normally: the artifact is the same whether or not the machine can
run a model.

## Examples

### Errors and string ownership

Adapted from `crates/ffi/tests/c_surface.rs` and the contract at the top of
`include/turbospark.h`. A `char **` result is always released with
`ts_string_free`, and the error slot is read immediately after a non-zero
return, on the calling thread.

```c
#include <stdlib.h>
#include "turbospark.h"

char *info = NULL;
if (ts_build_info_json(&info) != TS_OK) {
    /* The return value is the message's own length, so ask for it first. */
    size_t needed = ts_last_error(NULL, 0);
    char *buf = malloc(needed + 1);
    ts_last_error(buf, needed + 1); /* truncated copies stay valid C strings */
    free(buf);
    return;
}
printf("%s\n", info);
ts_string_free(info);
```

### Stream a generation turn

Adapted from the `collect` callback and generation tests in
`crates/ffi/tests/c_surface.rs`. The callback's `text` pointer is only valid
for the duration of the call, so copy it before returning.

```c
static void collect(void *ud, int kind, const char *text, size_t len,
                    uint32_t a, uint32_t b) {
    FILE *out = (FILE *)ud;
    if (kind == TS_EVENT_CONTENT) {
        fwrite(text, 1, len, out);
    } else if (kind == TS_EVENT_PREFILL) {
        fprintf(out, "[prefill %u/%u]", a, b);
    }
    /* Unknown kinds are a no-op, by contract. */
}

TsSession *session = NULL;
if (ts_session_open("/path/to/model.gturbo", NULL, &session) != TS_OK) {
    /* read ts_last_error here */ ;
}

char *result = NULL;
const char *messages = "[{\"role\":\"user\",\"content\":\"hi\"}]";
/* From any thread while this runs: ts_session_cancel(session). */
int rc = ts_generate(session, messages, NULL, collect, stdout, &result);
if (rc == TS_OK) {
    /* {"content":"...","stopReason":"endOfTurn","promptTokens":N,...} */
    ts_string_free(result);
}
ts_session_close(session);
```

The `result_json` object carries `stopReason` (`endOfTurn`, `toolCalls`,
`maxTokens`, `cancelled`, ...), token counts, tool calls in emission order,
and the turn's content and reasoning. A null callback skips streaming and
returns the whole turn in `result_json`.

### Consume it from Swift

The Swift package wraps the raw calls in small helpers. Adapted from
`swift/TurboSpark/Sources/TurboSpark/Errors.swift`; simplified.

```swift
import CTurboSpark
import Foundation

func buildInfo() throws -> String {
    var out: UnsafeMutablePointer<CChar>?
    guard ts_build_info_json(&out) == 0, let out else {
        let needed = ts_last_error(nil, 0)
        var buffer = [CChar](repeating: 0, count: needed + 1)
        _ = ts_last_error(&buffer, buffer.count)
        throw NSError(domain: "turbospark",
                      userInfo: [NSLocalizedDescriptionKey: String(cString: buffer)])
    }
    defer { ts_string_free(out) }
    return String(cString: out)
}
```

The real binding layer keeps this in `check` and `takeString` helpers and
decodes results with `JSONDecoder` and no key transform, because the Rust
side already emits camelCase.

## Gotchas

{% hint style="warning" %}
Every `extern "C"` body in this crate must be a call to `abi::guard`,
`abi::guard_result`, or `abi::guard_value` and nothing else. Unwinding across
the FFI boundary is undefined behaviour, and the workspace cannot opt out
with `panic = "abort"` because two `Drop` impls on the unwind path are
load-bearing (Metal command encoding, and a pool claim decode blocks on). A
new entry point that skips the guard compiles fine and is still wrong; the
caught panic surfaces as `TS_ERR_PANIC` with the payload in the error slot.
{% endhint %}

{% hint style="warning" %}
The header is hand-written, not generated, and the archive and header are
staged into SwiftPM by separate copies of the same script, both gitignored.
Rust tests cannot catch header drift because they call the same bodies
through the rlib; `make swift-test` compiling Swift against the staged header
is the real check. After changing the wire contract, run `make swift-lib`
before `make swift-test`, or you may verify the previous build.
{% endhint %}

## Related pages

- [crates overview](overview.md)
- [runtime](runtime.md)
- [The local HTTP server](../command-line-and-local-api/server.md)
- [Building from source](../developer/building-from-source.md)
- [Testing and verification](../developer/testing-and-verification.md)
