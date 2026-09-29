# API workspace

The API destination opens on Text. The selector moves through Text, Image,
and TypeSafe with clicks, left and right keys, or horizontal swipes on the
selector. Text and Image use the same TurboSpark listener and API key.
TypeSafe starts a separate OpenKind daemon on `127.0.0.1` with its own
Keychain key.

## Image request

Attach one installed Z-Image or Qwen-Image MLX model in the Image tab, then
send `POST /v1/images/generations` to the TurboSpark address:

```sh
curl http://127.0.0.1:54321/v1/images/generations \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer <TurboSpark-key>' \
  -d '{"model":"<attached-image-alias>","prompt":"a fox","size":"1024x1024","n":1}'
```

Replace the sample port with the one shown in the API pane, and use its
attached image alias and API key.

The endpoint returns `data[].b64_json` PNGs. `n` is 1 through 4 and runs
serially. `size` is `WIDTHxHEIGHT`, with each side 512 through 1024 in
multiples of 16 and total area at most 1024 by 1024. Optional `seed` is a
TurboSpark extension; the resolved first seed is in `x-turbospark-seed`.
Streaming, edits, URL output, and other OpenAI Images options are outside
this prompt-to-PNG subset and rejected. Image output is omitted from text
traffic previews.

The Image tab's test sends this HTTP request. Its PNG remains in memory
until **Save to gallery** is selected. The Image tab shares one resident image
session with the Images screen; image jobs run one at a time.

## TypeSafe service

The TypeSafe tab starts bundled `openkindd` with loopback HTTP and gRPC off.
It uses the OpenKind Swift binding for a typed `/v1/systemone` request,
health, model listing, and local load/unload controls. Only models marked
manageable by the daemon show those controls. The native playground offers
form and raw JSON request editing, six examples, Noul/Choice/Score question
builders, typed answers with probability bars, and the full JSON response.
The form validates IDs, criteria, and state before constructing a
`SystemRequest`; raw JSON is decoded to the same type before sending. The
response includes latency, token usage, errors, and request ID.

The Benchmark view sends one warmup followed by 1 to 200 sequential requests
per selected loaded model. Its mean, p50, p95, minimum, and maximum are
client wall-clock times, including local HTTP and Swift client overhead.
Stopping finishes the current request and skips remaining requests. These
figures are exploratory, not model or release benchmarks. This service has
its own URL and key. See [Swift API workspace](../swift/docs/SWIFT_API_WORKSPACE.md)
for the UI and binding seams.

Development builds use the sibling OpenKind Swift package. Set
`OPENKINDD_BINARY` to a local `openkindd` executable for the TypeSafe pane.
Release bundles require `OPENKIND_RELEASE_REVISION` so the Swift binding and
daemon are built from one published commit. See [RELEASE.md](RELEASE.md).

This endpoint does not implement ComfyUI workflows or AUTOMATIC1111 routes.
Existing image quality results do not establish served HTTP quality or
packaged app readiness.
