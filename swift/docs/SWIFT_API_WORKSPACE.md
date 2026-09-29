# Swift API workspace

The sidebar opens **API** on Text. The centered selector moves Text, Image,
TypeSafe by click, arrow keys, or horizontal swipe. Each mode retains its own
controls. Console and traffic tools remain available from the workspace.

## Text and Image

Text keeps the existing chat and completion server controls on the TurboSpark
listener, including `/v1/chat/completions`. Image attaches
one installed Z-Image or Qwen-Image model to the same TurboSpark listener,
address, and key. Its test sends `POST /v1/images/generations` through HTTP,
renders the PNG in memory, and saves to the gallery only on request.

`AppModel+ServerImages.swift` owns the served bridge to
`MLXImageGenerationSession.swift`. The Images screen and API share the
resident session and `ImageJobCoordinator`. A queue slot is released on
success, cancellation, detach, and stop. The Rust endpoint returns base64
PNG data; the traffic pane must not copy that data into text previews.

## TypeSafe

`TypeSafeAPIPaneView.swift` controls a separate loopback `openkindd` child
process through the OpenKind Swift package. The service has its own port and
Keychain key. Model load and unload buttons appear only for daemon-reported
manageable models. The app's development build uses `../openkind`;
`OPENKINDD_BINARY` can select a local daemon. Release packaging pins the
package and bundled daemon to one published revision, then checks the app
bundle and mounted DMG.

`TypeSafePlaygroundView.swift` is a native SwiftUI counterpart to OpenKind's
web playground. It has examples, a form/raw JSON switch, typed question
builders, answer cards, probability bars, response JSON, latency, usage, and
request ID. Both editors send `SystemRequest` to `/v1/systemone` through
`OpenKindServer.client.evaluate`; there is no separate playground API path.
`TypeSafePlaygroundDraft.swift` validates the form and maps it to the same
request type. Raw JSON can represent richer OpenKind requests that the form
cannot edit. Switching back to Form leaves Raw JSON selected if conversion
would lose data.

The Benchmark view uses the current request as a template and repeats it
serially for selected loaded models. It sends one warmup per model, then
records 1 to 200 client wall-clock samples. Stop skips future requests after
the current call. Results are local diagnostics; use the repository benchmark
harness for performance claims.

## Checks

After changing the Swift UI or binding, run `make compile-strings`, the
focused TypeSafe playground and localization tests, and `make swift-app`.
Server or bridge changes also need focused Rust/FFI checks and a real
installed-model HTTP smoke on Metal. Bundle and mounted-DMG checks are
required before release claims. See [API workspace](../../docs/API_WORKSPACE.md)
and [verification](../../.claude/docs/verification.md).
