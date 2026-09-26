---
description: "Open local models and stream text with the TurboSpark Swift package."
icon: code
---

# Swift bindings

The `TurboSpark` Swift package wraps the Rust C ABI. The SwiftPM package declares macOS 13 or later; running Metal model inference requires a supported Apple Silicon Mac. The desktop app itself requires macOS 14 or later.

## Build the package

From a TurboSpark checkout, build and stage the native library, then build the Swift package:

```sh
make swift-lib
cd swift/TurboSpark
swift build
```

## Open a model and stream a reply

`modelPath` accepts an installed alias or a model directory. Opening a session maps weights and prepares Metal pipelines, so reuse the session for multiple turns.

```swift
import TurboSpark

func ask() async throws {
    let session = try await TurboSparkSession(modelPath: "gemma4")
    let messages = [ChatMessage.user("Explain local model inference in one sentence.")]
    var options = GenerateOptions()
    options.maxNewTokens = 256

    for try await event in session.generate(messages, options: options) {
        switch event {
        case .content(let text):
            print(text, terminator: "")
        case .finished(let result):
            print("\n\nGenerated \(result.newTokens) tokens.")
        default:
            break
        }
    }
}
```

`GenerationEvent` also reports prefill progress, reasoning, and stop reason. It has a tool-call event case, but the current `GenerateOptions` API does not offer tools. `session.cancel()` requests a stop. The final `GenerationResult` includes the assistant text, token counts, timing, stop reason, and memory-pressure field.

Set per-turn limits and sampling with `GenerateOptions`. Its defaults are 512 new tokens, temperature 0.2, top-K 64, top-P 0.95, and reasoning off. Reasoning levels vary by model template. `OpenOptions` controls context and expert-cache sizing, load guard, power profile, speculation, and KV-cache quantization when opening a session.

## Main APIs

| API | Use it for |
| --- | --- |
| `TurboSparkSession` | Open a model; generate replies; count tokens; render prompts; tokenize and detokenize; fit a conversation to a context limit; inspect session info and phase timings. |
| `TurboSparkCatalog` | List available and installed models; estimate install size; recommend for a context; probe a Hub checkpoint; install or delete text and image models; manage the model store. Install streams report stages, byte progress, and the finished install. |
| `TurboSparkServer` | Start a server from a session, attach or detach sessions, poll request events, inspect the bound address, and stop the server. Set `ServerOptions(apiKey:)` when accepting requests beyond loopback. The server does not provide TLS. |
| `TurboSparkEmbedding` | Encode strings, compute cosine similarity, rank documents, and return top results with an encoder model. |
| `TurboSparkImageSession` | Run image generation from an installed image model and stream stage updates. |

Install progress is an async stream. It reports stages, downloaded bytes, then the installed model:

```swift
for try await event in TurboSparkCatalog.install("gemma4") {
    switch event {
    case .stage(let name): print(name)
    case .bytes(let done, let total): print("\(done) / \(total)")
    case .finished(let model): print("Installed at \(model.path)")
    }
}
```

For retrieval, `topK` returns the best matching document indices, text, and scores:

```swift
let matches = try await TurboSparkEmbedding.topK(
    query: "How do I install a model?",
    documents: helpPages,
    k: 3,
    modelPath: "my-embedding-model"
)
```

For full option fields and result types, see the [Swift source](https://github.com/whit3rabbit/turbospark/tree/main/swift/TurboSpark/Sources/TurboSpark). The existing [Swift binding notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/SWIFT_BINDINGS.md) cover ABI and implementation details.
