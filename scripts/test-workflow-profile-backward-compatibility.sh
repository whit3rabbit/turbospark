#!/usr/bin/env bash
# Run on macOS with `scripts/test-workflow-profile-backward-compatibility.sh`.
# The script uses the historical Package.resolved and requires a pinned
# sibling OpenKind revision via WORKFLOW_COMPAT_OPENKIND_REVISION.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
historical_revision="8e99472fd6d2c68cc3192b5abfbdc1a7cd3a9a84"
temp_root="$(mktemp -d "${TMPDIR:-/tmp}/turbospark-workflow-profile-compat.XXXXXX")"
cache_root="${WORKFLOW_COMPAT_SCRATCH_ROOT:-${TMPDIR:-/tmp}/turbospark-workflow-profile-compat-build}"
current_scratch="${WORKFLOW_COMPAT_CURRENT_SCRATCH:-$cache_root/current}"
historical_scratch="${WORKFLOW_COMPAT_HISTORICAL_SCRATCH:-$cache_root/historical}"

cleanup() {
    if [[ "${WORKFLOW_COMPAT_KEEP_TMP:-0}" == "1" ]]; then
        printf 'Kept compatibility fixture and archived source at %s\n' "$temp_root"
    else
        rm -rf "$temp_root"
        if [[ -z "${WORKFLOW_COMPAT_SCRATCH_ROOT:-}" ]]; then
            rm -rf "$cache_root"
        fi
    fi
}
trap cleanup EXIT

fail() {
    printf 'workflow profile compatibility: %s\n' "$1" >&2
    exit 2
}

[[ "$(uname -s)" == "Darwin" ]] || fail "requires macOS and SQLCipher-supported app dependencies"
command -v git >/dev/null || fail "git is required"
command -v swift >/dev/null || fail "Swift is required"
command -v clang >/dev/null || fail "clang is required for SwiftPM's Syntext linker placeholder"
command -v cargo >/dev/null || fail "cargo is required to build the historical app's FFI archive"
command -v nmedit >/dev/null || fail "nmedit is required for the historical app's FFI archive"
git -C "$repo_root" cat-file -e "${historical_revision}^{commit}" \
    || fail "historical baseline ${historical_revision} is not available locally"
git -C "$repo_root" cat-file -e "${historical_revision}:swift/TurboSparkApp/Package.swift" \
    || fail "historical TurboSparkApp Swift package is missing from ${historical_revision}"
git -C "$repo_root" cat-file -e "${historical_revision}:swift/TurboSparkApp/Package.resolved" \
    || fail "historical Package.resolved is missing from ${historical_revision}"

ffi_archive="$repo_root/swift/TurboSpark/Sources/CTurboSpark/libturbospark_ffi.a"
ffi_header="$repo_root/crates/ffi/include/turbospark.h"
[[ -f "$ffi_archive" ]] || fail "missing staged FFI archive at $ffi_archive; run make swift-lib"
[[ -f "$ffi_header" ]] || fail "missing canonical FFI header at $ffi_header"

[[ -z "${OPENKIND_RELEASE_REVISION:-}" ]] \
    || fail "unset OPENKIND_RELEASE_REVISION; this harness uses the historical local-package lockfile"
openkind_checkout="$repo_root/../openkind"
[[ -f "$openkind_checkout/Package.swift" ]] \
    || fail "missing sibling ../openkind checkout; the historical package uses this local dependency"
openkind_revision="$(git -C "$openkind_checkout" rev-parse HEAD 2>/dev/null)" \
    || fail "sibling ../openkind is not a git checkout"
[[ -n "${WORKFLOW_COMPAT_OPENKIND_REVISION:-}" ]] \
    || fail "set WORKFLOW_COMPAT_OPENKIND_REVISION to the sibling OpenKind commit for a reproducible run"
if [[ "$openkind_revision" != "$WORKFLOW_COMPAT_OPENKIND_REVISION" ]]; then
    fail "sibling OpenKind revision is $openkind_revision, expected $WORKFLOW_COMPAT_OPENKIND_REVISION"
fi
mkdir -p "$temp_root/openkind"
git -C "$openkind_checkout" archive "$openkind_revision" | tar -x -C "$temp_root/openkind"
printf 'Using OpenKind source archive at %s\n' "$openkind_revision"

historical_root="$temp_root/historical"
mkdir -p "$historical_root"
git -C "$repo_root" archive "$historical_revision" swift/TurboSparkApp swift/TurboSpark \
    | tar -x -C "$historical_root"

# The historical app still links syntext's own SyntextFFI.xcframework, and
# the staged archive now carries the same `syntext_*` C symbols (crates/ffi
# links syntext-context). So the historical copy gets an archive built
# without them, and its second Rust runtime symbol made local exactly as the
# app's build did before the merge.
historical_ffi="$historical_root/swift/TurboSpark/Sources/CTurboSpark"
mkdir -p "$historical_ffi"
MACOSX_DEPLOYMENT_TARGET=13.0 cargo build --release -p turbospark-ffi --no-default-features \
    --target aarch64-apple-darwin --manifest-path "$repo_root/Cargo.toml" \
    || fail "could not build the syntext-free FFI archive for the historical app"
cp "$repo_root/target/aarch64-apple-darwin/release/libturbospark_ffi.a" "$historical_ffi/libturbospark_ffi.a"
printf '_rust_eh_personality\n' > "$temp_root/hide_symbols"
nmedit -R "$temp_root/hide_symbols" "$historical_ffi/libturbospark_ffi.a"
cp "$ffi_header" "$historical_ffi/turbospark.h"

historical_app="$historical_root/swift/TurboSparkApp"
historical_tests="$historical_app/Tests/TurboSparkAppTests"
mkdir -p "$historical_tests"
cp "$repo_root/scripts/workflow-profile-compatibility/WorkflowHistoricalProfileReaderTests.swift" \
    "$historical_tests/WorkflowHistoricalProfileReaderTests.swift"

mkdir -p "$current_scratch" "$historical_scratch"
export WORKFLOW_COMPATIBILITY_ROOT="$temp_root"

# Only the historical package needs this: its header-only `CSyntext` target
# from the syntext package has no object file, which Swift Build links
# anyway. The current app's vendored `CSyntext` is a system-library target.
prepare_syntext_linker_placeholder() {
    local scratch_path="$1"
    mkdir -p "$scratch_path/out/Products/Debug"
    MACOSX_DEPLOYMENT_TARGET=14.0 clang -c -x c /dev/null \
        -o "$scratch_path/out/Products/Debug/CSyntext.o"
}

printf 'Creating deterministic workflow-bearing profile with the current app code.\n'
swift test \
    --disable-automatic-resolution \
    --package-path "$repo_root/swift/TurboSparkApp" \
    --scratch-path "$current_scratch" \
    --jobs 1 \
    --filter WorkflowProfileCrossVersionCompatibilityTests/testCreateFixtureForHistoricalReader

printf 'Opening that fixture with historical TurboSparkApp sources from %s.\n' "$historical_revision"
prepare_syntext_linker_placeholder "$historical_scratch"
swift test \
    --disable-automatic-resolution \
    --package-path "$historical_app" \
    --scratch-path "$historical_scratch" \
    --jobs 1 \
    --filter WorkflowHistoricalProfileReaderTests/testHistoricalInitializerReadsAndWritesCoreRecords

printf 'Comparing workflow-owned schema and stored SQLite values, then reopening with current code.\n'
swift test \
    --disable-automatic-resolution \
    --package-path "$repo_root/swift/TurboSparkApp" \
    --scratch-path "$current_scratch" \
    --jobs 1 \
    --filter WorkflowProfileCrossVersionCompatibilityTests/testVerifyHistoricalReaderCompatibility

printf 'Historical reader compatibility check passed for %s.\n' "$historical_revision"
