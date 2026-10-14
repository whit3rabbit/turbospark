#!/usr/bin/env bash
# Copies syntext's hand-written C headers and its Swift wrappers into
# swift/TurboSpark/Vendor/syntext from the EXACT revision crates/ffi links.
#
# Why vendored rather than a SwiftPM dependency: a process can link only one
# Rust staticlib, so the syntext and syntext-context C ABIs are compiled into
# libturbospark_ffi.a (crates/ffi depends on syntext-context with its `ffi`
# feature). syntext's own Swift package would add a second Rust archive,
# its `SyntextFFI.xcframework`. The wrappers share `package`-access helpers,
# so `Syntext` and `SyntextContext` must live in one package: the binding
# package, which already owns the archive.
#
# The headers describe the ABI of one revision only. Run this after every
# change to the `syntext-context` rev in crates/ffi/Cargo.toml (and a
# `cargo update -p syntext-context`); `scripts/swift-lib.sh` refuses to stage
# a library whose Cargo.lock revision differs from Vendor/syntext/REVISION.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/swift/TurboSpark/Vendor/syntext"

rev="$("$root/scripts/syntext-locked-rev.sh")"

# The checkout cargo resolved for that revision. `syntext-context` is the
# `context/` member of the syntext repository, so the repository root is two
# levels above its manifest.
cargo fetch --quiet --manifest-path "$root/Cargo.toml"
manifest="$(cargo metadata --format-version 1 --manifest-path "$root/Cargo.toml" \
  | python3 -c '
import json, sys
meta = json.load(sys.stdin)
paths = [p["manifest_path"] for p in meta["packages"] if p["name"] == "syntext-context"]
if len(paths) != 1:
    sys.exit("expected exactly one syntext-context package, found %d" % len(paths))
print(paths[0])
')"
src="$(cd "$(dirname "$manifest")/.." && pwd)"
[ -f "$src/swift/Sources/CSyntext/include/syntext_context.h" ] || {
  echo "no syntext Swift sources under $src" >&2
  exit 1
}

rm -rf "$dest/CSyntext" "$dest/Syntext" "$dest/SyntextContext" "$dest/Tests"
mkdir -p "$dest/CSyntext" "$dest/Tests"
# A system-library target: the modulemap sits at the target root beside the
# headers it names, so no object file is built for it (syntext's own
# header-only `CSyntext` target needed a placeholder `CSyntext.o` under
# Swift Build).
cp "$src/swift/Sources/CSyntext/include/module.modulemap" \
   "$src/swift/Sources/CSyntext/include/syntext.h" \
   "$src/swift/Sources/CSyntext/include/syntext_context.h" \
   "$dest/CSyntext/"
cp -R "$src/swift/Sources/Syntext" "$dest/Syntext"
cp -R "$src/swift/Sources/SyntextContext" "$dest/SyntextContext"
cp -R "$src/swift/Tests/SyntextTests" "$dest/Tests/SyntextTests"
cp -R "$src/swift/Tests/SyntextContextTests" "$dest/Tests/SyntextContextTests"
printf '%s\n' "$rev" > "$dest/REVISION"

printf 'vendored syntext %s -> %s\n' "$rev" "$dest"
