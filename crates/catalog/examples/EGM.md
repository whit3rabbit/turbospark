# catalog examples

## Overview
Developer tooling for the catalog crate. `audio_pin` prints the pinned-source
fields of an audio catalog row so a new row in `src/audio_assets.json` is never
written with a digest nobody computed.

## Interface
- `audio_pin <owner/repo> <revision|main> [prefix ...]` [examples/audio_pin.rs:main]. Resolves the revision to a 40-hex commit via `turbospark_catalog::Client` and prints `source`, `assets`, and `required_files`.
- Usage string constant [examples/audio_pin.rs:USAGE].

## Invariants
- Every path, size, and SHA-256 comes from the resolved commit; non-LFS files are downloaded and hashed locally [examples/audio_pin.rs: module docs, `use sha2::Sha256`].
- Exits with status 1 and prints `audio_pin: <error>` on failure [examples/audio_pin.rs:main].

## Side Effects
- Network: Hugging Face Hub metadata requests and file downloads through `Client` [examples/audio_pin.rs:run].
- Writes only to stdout and stderr.

## Verification
- Run: `cargo test -p turbospark-catalog`
