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

printf 'staticlib %s MiB -> %s\n' \
  "$(( $(stat -f%z "$dest/libturbospark_ffi.a") / 1048576 ))" "$dest"
