#!/usr/bin/env bash
# Assembles swift/TurboSparkApp's bare SwiftPM executable into a real
# TurboSpark.app bundle, with the three CLI binaries inside it.
#
# WHY THIS SCRIPT EXISTS: `swift build` emits an executable and a resource
# bundle side by side in `.build/release`, not an `.app` (swift/CLAUDE.md
# Gotcha 12). There is no Xcode project here and no Info.plist anywhere in the
# tree, so every bundle-identity fact -- identifier, version, minimum system
# version -- is invented HERE and nowhere else. Change one and it changes for
# the DMG, the cask and the user's preferences domain at the same time.
#
# THE CLI BINARIES SHIP INSIDE THE BUNDLE ON PURPOSE. A Homebrew cask has one
# `url`, so app-plus-CLI in one install means one artifact; the cask's `binary`
# stanzas point at `Contents/MacOS/` and `brew uninstall` unlinks them with the
# app. See docs/RELEASE.md.
#
# Usage:
#   scripts/make-app-bundle.sh [--out DIR] [--version X.Y.Z] [--skip-build]
#
# Env:
#   CODESIGN_IDENTITY  signing identity; default "-" (ad-hoc). Ad-hoc is the
#                      minimum an arm64 binary needs to execute AT ALL, and it
#                      is NOT notarization -- see docs/RELEASE.md's Gatekeeper
#                      section before assuming a downloaded build just opens.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="$root/dist"
version=""
skip_build=0

while [ $# -gt 0 ]; do
  case "$1" in
    --out) out_dir="$2"; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    --skip-build) skip_build=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$(uname -s)" != "Darwin" ]; then
  echo "make-app-bundle.sh is macOS only" >&2
  exit 1
fi

# One version number for the whole repo (CHANGELOG.md's preamble says so, and
# release.yml's check-version job enforces tag == this). Read it rather than
# taking it as an argument by default, so a hand-run bundle cannot disagree
# with what CI would have produced from the same tree.
if [ -z "$version" ]; then
  version="$(sed -nE 's/^version = "([^"]+)".*/\1/p' "$root/Cargo.toml" | head -1)"
fi
[ -n "$version" ] || { echo "could not resolve version from Cargo.toml" >&2; exit 1; }

bundle_id="com.whit3rabbit.turbospark"
# MUST match `platforms: [.macOS(.v14)]` in TurboSparkApp/Package.swift and the
# `depends_on macos:` line in both casks. Sonoma is 14.
min_macos="14.0"
target="aarch64-apple-darwin"
identity="${CODESIGN_IDENTITY:--}"

app="$out_dir/TurboSpark.app"
contents="$app/Contents"

if [ "$skip_build" -eq 0 ]; then
  echo "==> staging the FFI staticlib (crates/ffi -> SwiftPM)"
  TURBOSPARK_LOCALIZE_RUST_SYMBOLS=1 "$root/scripts/swift-lib.sh"

  echo "==> compiling the string catalog (Localization/Localizable.xcstrings -> .lproj)"
  "$root/scripts/compile-strings.sh"

  echo "==> cargo build --release (CLI + server)"
  cargo build --release --target "$target" \
    -p turbospark-cli -p turbospark-server --manifest-path "$root/Cargo.toml"

  echo "==> swift build -c release (TurboSparkApp)"
  (cd "$root/swift/TurboSparkApp" && swift build -c release -Xbuild-tools-swiftc -suppress-warnings)
fi

build_dir="$root/swift/TurboSparkApp/.build/release"
exe="$build_dir/TurboSparkApp"
[ -x "$exe" ] || { echo "expected $exe to exist; run without --skip-build" >&2; exit 1; }

echo "==> assembling $app"
rm -rf "$app"
mkdir -p "$contents/MacOS" "$contents/Resources"

cp "$exe" "$contents/MacOS/TurboSparkApp"

# `Bundle.module` resolves against Bundle.main.resourceURL in a real bundle, so
# Contents/Resources is where SwiftPM's generated resource bundles have to
# land. Globbed rather than named: a dependency that grows resources adds a
# second .bundle here and naming only ours would silently drop it.
found_bundle=0
for res in "$build_dir"/*.bundle; do
  [ -e "$res" ] || continue
  cp -R "$res" "$contents/Resources/"
  found_bundle=1
done
[ "$found_bundle" -eq 1 ] || {
  echo "no *.bundle in $build_dir -- Logos/ and app-prompts.json would be missing" >&2
  exit 1
}

# The CLI, inside the app. The cask links these onto PATH from here.
for bin in turbospark-check turbospark-model turbospark-server; do
  src="$root/target/$target/release/$bin"
  [ -x "$src" ] || { echo "expected $src to exist; run without --skip-build" >&2; exit 1; }
  cp "$src" "$contents/MacOS/$bin"
done

# Copy application icon if available
if [ -f "$root/assets/icons/AppIcon.icns" ]; then
  cp "$root/assets/icons/AppIcon.icns" "$contents/Resources/AppIcon.icns"
fi

cat > "$contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>TurboSpark</string>
    <key>CFBundleExecutable</key>
    <string>TurboSparkApp</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIconName</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>${bundle_id}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>TurboSpark</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${version}</string>
    <key>CFBundleVersion</key>
    <string>${version}</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
    <key>LSMinimumSystemVersion</key>
    <string>${min_macos}</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>
PLIST

printf 'APPL????' > "$contents/PkgInfo"

plutil -lint "$contents/Info.plist" >/dev/null

echo "==> codesign (identity: ${identity})"
# Inner executables first, then the bundle: a signature over the bundle seals
# what is inside it, so signing the wrapper before its contents invalidates
# itself. `--deep` is deprecated by Apple for exactly this reason.
for bin in turbospark-check turbospark-model turbospark-server; do
  codesign --force --timestamp=none --sign "$identity" "$contents/MacOS/$bin"
done
codesign --force --timestamp=none --sign "$identity" "$app"
codesign --verify --strict "$app"

echo "==> ok: $app ($(du -sh "$app" | cut -f1), version ${version}, id ${bundle_id})"
