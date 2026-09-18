#!/usr/bin/env bash
# Builds crates/ffi and copies the staticlib plus the canonical header into
# the SwiftPM package.
#
# The copy is deliberate. A SwiftPM target may not reach outside its own
# directory, and a modulemap pointing back into `crates/` works right up
# until someone builds from a different root. Both copies are gitignored;
# `crates/ffi/include/turbospark.h` is the canonical one.
#
# ARM64 ONLY, and that is a decision rather than an oversight: the engine is
# Metal on Apple Silicon and has never been run on an Intel Mac. Add a `lipo`
# step here if that ever changes.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/swift/TurboSpark/Sources/CTurboSpark"

# MUST match the `platforms:` line in both Package.swift files.
#
# Without it, cargo builds the objects for the host SDK's default (macOS 26.5
# on this machine) while SwiftPM links for 13.0, and every object file in the
# archive draws an `ld` warning. The warnings are the visible half; the real
# problem is an app that claims to support macOS 13 while containing objects
# built against a much newer SDK.
export MACOSX_DEPLOYMENT_TARGET=13.0

cargo build --release -p turbospark-ffi --target aarch64-apple-darwin --manifest-path "$root/Cargo.toml"

lib="$root/target/aarch64-apple-darwin/release/libturbospark_ffi.a"
[ -f "$lib" ] || { echo "expected $lib to exist" >&2; exit 1; }

mkdir -p "$dest"
cp "$lib" "$dest/libturbospark_ffi.a"
cp "$root/crates/ffi/include/turbospark.h" "$dest/turbospark.h"

# Keep Rust's panic personality global inside the archive. `nmedit -R` makes
# the definition static in the one std object that owns it, while the other
# Rust archive members still reference it as an external symbol. Swift then
# fails at link time with `_rust_eh_personality` undefined. The FFI archive is
# the only Rust static library linked by these Swift packages, so there is no
# duplicate runtime symbol to localize here.

# **SwiftPM DOES NOT TREAT THE ARCHIVE AS A BUILD INPUT, so without this the
# test target links the PREVIOUS staticlib and reports on code that is no
# longer in the tree.** The `-L` path arrives as an unsafe linker flag, which
# SwiftPM passes through without adding a dependency edge, so a `.a` that
# changed under an unchanged set of Swift sources triggers no relink at all.
#
# That is not a tidiness point: `swift test` is the ONLY thing that can check
# the hand-written header (crates/ffi/CLAUDE.md Gotcha 2), and a stale link
# makes it check the build before the one being tested. Measured 2026-08-21
# while mutation-checking the FFI's speculation options: two mutations of
# `open.rs` in a row read as the FIRST one's failure, twice, and the restored
# tree still read red until the Swift sources were touched by hand.
#
# Touching the sources is the cheapest correct fix. Deleting `.build` would
# also work and costs a full rebuild of the package every time.
find "$root/swift/TurboSpark/Sources/TurboSpark" \
     "$root/swift/TurboSpark/Tests" \
     "$root/swift/TurboSparkApp/Sources" \
     -name '*.swift' -exec touch {} +

printf 'staticlib %s MiB -> %s\n' \
  "$(( $(stat -f%z "$dest/libturbospark_ffi.a") / 1048576 ))" "$dest"
