#!/usr/bin/env bash
# Compiles swift/TurboSparkApp/Localization/Localizable.xcstrings into the
# per-language .lproj resources the app actually runs on.
#
# `swift build` and `swift run` -- the ONLY way this app is ever built, since
# no Xcode project exists -- copy a `.xcstrings` file into the resource
# bundle VERBATIM rather than compiling it: there is no SwiftPM build phase
# for the String Catalog format, only Xcode's own "Compile String Catalogs"
# step has one. Confirmed by inspecting the built resource bundle after a
# `swift build -v`: the raw 600+ KiB JSON sits there and no `.lproj`
# directory exists anywhere in the output, so every one of the 21 languages
# was inert regardless of what any `Text(...)` call site passed as `bundle:`
# (`swift/docs/SWIFT_SETTINGS_AUDIT.md` item 1). `xcstringstool` is the same
# private tool Xcode's build phase calls; running it here reproduces exactly
# what Xcode would have produced.
#
# The output lands directly inside `Sources/TurboSparkApp/Resources/`, which
# `Package.swift`'s `.process("Resources")` rule already bundles -- SwiftPM
# DOES understand plain `<language>.lproj/Localizable.strings` directories,
# unlike the `.xcstrings` source format one level up. That is why the source
# catalog lives in `Localization/` rather than under `Resources/`: it is a
# build INPUT, not a runtime resource, and shipping the raw JSON alongside
# its own compiled output would be dead weight with two sources of truth.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_catalog="$root/swift/TurboSparkApp/Localization/Localizable.xcstrings"
resources_dir="$root/swift/TurboSparkApp/Sources/TurboSparkApp/Resources"

[ -f "$source_catalog" ] || { echo "expected $source_catalog to exist" >&2; exit 1; }

tool="$(xcrun --find xcstringstool 2>/dev/null || true)"
[ -n "$tool" ] || {
  echo "xcstringstool not found (needs a full Xcode install, not just the" >&2
  echo "Command Line Tools); localized strings will not be up to date." >&2
  exit 1
}

# Remove stale output first: a language or key REMOVED from the catalog
# must not leave its last compiled .lproj behind for `.process()` to bundle.
find "$resources_dir" -maxdepth 1 -name '*.lproj' -exec rm -rf {} +

"$tool" compile "$source_catalog" --output-directory "$resources_dir"

lang_count="$(find "$resources_dir" -maxdepth 1 -name '*.lproj' | wc -l | tr -d ' ')"
printf 'compiled %s -> %s (%s languages)\n' "$source_catalog" "$resources_dir" "$lang_count"
