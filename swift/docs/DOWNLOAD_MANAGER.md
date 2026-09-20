# Download manager

The macOS app has one FIFO for text and image model installs. Swift owns the
queue, durable history, controls, and presentation. Rust owns repository
resolution, network transfer, conversion, atomic publication, and final
verification.

## Lifecycle

Each row moves through the states that apply to its model:

```text
Queued -> Downloading <-> Paused -> Packing -> Verifying -> Loading -> Completed
                     \-> Cancelling -> Cancelled
```

Image installs do not load a text runtime, so they complete after verification.
Packing and verifying are not pausable. A pause is acknowledged by the native
range downloader, allows in-flight requests to finish, and prevents new
requests until resume.

Swift maps native stage messages onto these coarse states in
`AppModel+Downloads.swift`. The stage message remains the detailed user-facing
description. Download history is profile-scoped, encrypted with the rest of
the profile database, bounded, and reconciled after relaunch. Work that was
downloading, paused, packing, or verifying becomes Interrupted. Loading becomes
Completed because the native install was already committed before loading
began.

## Network path

Text GGUF and safetensors reads and curated image files share
`repack::HttpRangeSource`.

- Files are split into 16 MiB HTTP ranges.
- Up to eight HTTP/1.1 requests run concurrently. HTTP/1.1 is intentional for
  the Xet CDN path because HTTP/2 can multiplex the requests onto one edge.
- Each range has bounded retries. HTTP 429 and selected 5xx responses back off;
  permanent statuses fail immediately.
- Image ranges are written into a pre-sized sibling partial file. The complete
  file is renamed into place only after every range succeeds. Its eight
  workers hold at most one 16 MiB range each, so this path adds at most about
  128 MiB of range buffers.
- Progress callbacks may arrive from worker threads. Swift serializes their UI
  effects on the main actor and prevents out-of-order callbacks from reducing
  displayed progress.

The 16 MiB size replaces the former 64 MiB cap. Existing measurements showed
the eight-way path scaling at 16 MiB, and the smaller cap lets more dense-model
tensors enter that path while reducing retry cost. This change has not been
re-frozen as a full model-install benchmark.

## Resume and integrity

Resume is enabled only for repositories pinned to a 40-hex commit. Floating
branches do not reuse cached bytes.

Each completed range is stored under `.download-cache`, keyed by the resolved
URL and byte bounds. A cache entry contains its SHA-256 followed by its bytes.
Reads verify both length and hash. An unreadable or corrupt entry is deleted and
fetched again. The normal install manifest and receipt checks remain the final
authority.

Failed and cancelled installs retain verified ranges. Retrying the same pinned
revision reuses them, including after an app relaunch. Conversion and packing
restart because partially converted installs are never published. The cache is
removed after the completed install passes verification. Curated image source
staging is also revision-scoped and retained on failure, then removed after a
successful atomic publish.

This trades temporary disk usage for restart speed. During an interrupted
install, the verified range cache can approach the source model size. A
successful install removes it. The Swift install gate therefore budgets the
estimated download plus the final install, then preserves its normal 8 GiB
free-space headroom.

## Image packing

Image model components are packed one at a time to preserve the documented
one-heavyweight-stage-at-a-time memory envelope. The packer hashes payload
bytes while writing them. Manifest assembly reuses that digest instead of
reading every payload into memory and hashing it again. Atomic publication
still performs one full verification pass before the install becomes visible.
The FFI layer validates the published manifest but does not repeat the same
full-file hash pass immediately afterward.

## Main files

- `State/ModelDownload.swift`: request identity and durable lifecycle.
- `State/AppModel+Downloads.swift`: queue, retry, pause, progress, and stage
  mapping.
- `Components/DownloadManagerView.swift`: active work and history UI.
- `crates/repack/src/ranged_download/`: range concurrency, retry, cache, and
  partial-file publication.
- `crates/catalog/src/stream.rs`: immutable-revision cache policy for text
  installs.
- `crates/catalog/src/image.rs`: curated image source selection and transfer.
- `crates/image/src/builder.rs`: packing, verification, and atomic publication.

## Verification

Focused offline coverage includes range partitioning, concurrent placement,
retry classification, cache integrity, cached file publication, queue ordering,
stage mapping, and relaunch reconciliation. Run the normal Swift and FFI gates
after downloader changes:

```sh
make swift-lib
make swift-test
```

A live Hugging Face install is required before making an end-to-end throughput
claim. Unit tests and compilation do not establish CDN speed or real disk-use
peaks.
