#!/usr/bin/env bash
# Assembles swift/TurboSparkApp's bare SwiftPM executable into a real
# TurboSpark.app bundle, with the three CLI binaries and openkindd inside it.
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
#   OPENKINDD_INSTALL_ROOT
#                      where `cargo install openkind-server` lands; default a
#                      temp dir discarded with the run. A persistent root is
#                      what ci.yml caches, because cargo install builds in a
#                      target directory of its own that no workspace cache
#                      sees, and the mlx feature (mlx-sys, candle, mlx-rs) was
#                      15 minutes of every package-macos leg. A matching
#                      version already in the root makes the install a no-op.
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

# Sparkle update feed + trust. The feed is the appcast.xml asset attached to
# every GitHub release, reached through the evergreen latest-release URL; the
# DMG is itself the update archive Sparkle installs. Trust is EdDSA (the
# public half of the key whose private half lives in the keychain on the
# release machine and as the SPARKLE_ED_PRIVATE_KEY repo secret -- see
# docs/RELEASE.md). The app is ad-hoc signed, so there is no Team ID for
# Sparkle's Apple-identity comparison; the EdDSA signature is what makes an
# update verifiable. This literal is PUBLIC on purpose -- local bundles and
# the ci.yml packaging job must never need the secret.
update_feed_url="https://github.com/whit3rabbit/turbospark/releases/latest/download/appcast.xml"
sparkle_public_ed_key="O+w+eDlUPQQa1Y+quYgz9BiJljUdS01KeafmpSzoZ3g="

app="$out_dir/TurboSpark.app"
contents="$app/Contents"
openkind_revision="${OPENKIND_RELEASE_REVISION:-}"
openkind_server_version="${OPENKIND_SERVER_VERSION:-0.3.0}"
openkind_bin="${OPENKINDD_BINARY:-}"
openkind_build_dir=""

if [ "$skip_build" -eq 0 ]; then
  [[ "$openkind_revision" =~ ^[0-9a-f]{40}$ ]] || {
    echo "OPENKIND_RELEASE_REVISION must be an exact published 40-character commit for the Swift package" >&2
    exit 1
  }
  if ! command -v protoc >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then
      echo "==> protoc not found; installing protobuf via Homebrew"
      HOMEBREW_NO_AUTO_UPDATE=1 brew install protobuf
    else
      echo "error: protoc is required to build openkind-server; install protobuf (e.g. brew install protobuf)" >&2
      exit 1
    fi
  fi
  # openkind-server's `mlx` feature builds mlx-sys, whose bindgen step runs
  # libclang. Under Xcode 27 (seen 2026-10-14, Command Line Tools also
  # installed) libclang found no SDK on its own and failed with "'stdlib.h'
  # file not found". Point both at the active SDK unless the caller did.
  if [ -z "${SDKROOT:-}" ]; then
    SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
    export SDKROOT
  fi
  if [ -z "${BINDGEN_EXTRA_CLANG_ARGS:-}" ]; then
    export BINDGEN_EXTRA_CLANG_ARGS="-isysroot $SDKROOT"
  fi
  if [ -n "${OPENKINDD_INSTALL_ROOT:-}" ]; then
    openkind_install_root="$OPENKINDD_INSTALL_ROOT"
  else
    openkind_build_dir="$(mktemp -d)"
    trap 'rm -rf "$openkind_build_dir"' EXIT
    openkind_install_root="$openkind_build_dir/install"
  fi
  # Idempotent: cargo's bookkeeping in the root ("Ignored package ... is
  # already installed") is what makes a cached root cheap, and it is
  # version-checked, which a bare `[ -x ]` test would not be.
  cargo install --locked --version "$openkind_server_version" --target "$target" --features mlx \
    --root "$openkind_install_root" openkind-server
  openkind_bin="$openkind_install_root/bin/openkindd"
fi
[ -x "$openkind_bin" ] || { echo "openkindd binary missing; set OPENKINDD_BINARY for --skip-build" >&2; exit 1; }

if [ "$skip_build" -eq 0 ]; then
  echo "==> staging the FFI staticlib (crates/ffi -> SwiftPM)"
  TURBOSPARK_LOCALIZE_RUST_SYMBOLS=1 "$root/scripts/swift-lib.sh"

  echo "==> compiling the string catalog (Localization/Localizable.xcstrings -> .lproj)"
  "$root/scripts/compile-strings.sh"

  # The SAME deployment target swift-lib.sh exported for the staticlib.
  # Without it the two release builds disagree on MACOSX_DEPLOYMENT_TARGET
  # and cargo rebuilds 22 shared crates (runtime, gpu, audio, catalog,
  # repack, image and their deps) a second time: 4m45s of the package-macos
  # leg, measured 2026-10-26. The CLI binaries' floor becomes 13.0 instead
  # of rustc's 11.0 default; the bundle's floor is 14.0 either way.
  echo "==> cargo build --release (CLI + server)"
  MACOSX_DEPLOYMENT_TARGET=13.0 cargo build --release --target "$target" \
    -p turbospark-cli -p turbospark-server --manifest-path "$root/Cargo.toml"

  echo "==> swift build -c release (TurboSparkApp)"
  (cd "$root/swift/TurboSparkApp" && swift build -c release -Xbuild-tools-swiftc -suppress-warnings)
fi

# MLX's GPU kernels. mlx-swift ships them as .metal sources that only Xcode's
# build system compiles (xcodebuild, or SwiftPM's swiftbuild): the native
# `swift build` above, which every Xcode the release and CI pin uses, emits
# no mlx-swift_Cmlx.bundle at all. Without it MLX's first GPU op hits
# mlx-c's default error handler, which calls exit(-1): v0.6.0 quit on the
# first Z-Image generation with no crash report. So build that one bundle
# with xcodebuild, from a copy of the SAME resolved checkout the app linked,
# so kernels and C++ never disagree. The copy keeps Xcode's .swiftpm state
# out of SwiftPM's checkout. Under .build, so CI's SwiftPM cache keeps it and
# --skip-build reuses it.
mlx_checkout="$root/swift/TurboSparkApp/.build/checkouts/mlx-swift"
mlx_kernels="$root/swift/TurboSparkApp/.build/mlx-kernels"
mlx_bundle="$mlx_kernels/derived/Build/Products/Release/mlx-swift_Cmlx.bundle"
if [ "$skip_build" -eq 0 ]; then
  [ -f "$mlx_checkout/Package.swift" ] || {
    echo "no mlx-swift checkout at $mlx_checkout; the swift build should have resolved it" >&2
    exit 1
  }
  if ! xcrun --find metal >/dev/null 2>&1; then
    # Xcode 26 and later ship the Metal compiler as a separate component.
    # CI fetches it; on a workstation that download is the user's call.
    if [ -n "${CI:-}" ]; then
      echo "==> Metal toolchain missing; xcodebuild -downloadComponent MetalToolchain"
      xcodebuild -downloadComponent MetalToolchain
    else
      echo "the Metal compiler is missing; run: xcodebuild -downloadComponent MetalToolchain" >&2
      exit 1
    fi
  fi
  echo "==> xcodebuild mlx-swift's Metal kernels (mlx-swift_Cmlx.bundle)"
  mkdir -p "$mlx_kernels"
  rsync -a --delete --exclude .git --exclude .swiftpm "$mlx_checkout/" "$mlx_kernels/src/"
  (cd "$mlx_kernels/src" && xcodebuild build -quiet -scheme MLX -configuration Release \
    -destination 'generic/platform=macOS' -derivedDataPath "$mlx_kernels/derived" \
    -skipPackagePluginValidation ARCHS=arm64 ONLY_ACTIVE_ARCH=NO)
fi
[ -s "$mlx_bundle/Contents/Resources/default.metallib" ] || {
  echo "expected MLX kernels at $mlx_bundle/Contents/Resources/default.metallib; run without --skip-build" >&2
  exit 1
}

build_dir="$root/swift/TurboSparkApp/.build/release"
exe="$build_dir/TurboSparkApp"
[ -x "$exe" ] || { echo "expected $exe to exist; run without --skip-build" >&2; exit 1; }

echo "==> assembling $app"
rm -rf "$app"
mkdir -p "$contents/MacOS" "$contents/Resources" "$contents/Frameworks"

cp "$exe" "$contents/MacOS/TurboSparkApp"
cp "$openkind_bin" "$contents/MacOS/openkindd"
if [ "$skip_build" -eq 0 ]; then
  printf '%s\n' "$openkind_revision" > "$contents/Resources/OpenKindRevision.txt"
  printf '%s\n' "$openkind_server_version" > "$contents/Resources/OpenKindServerVersion.txt"
fi

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
# MLX looks for it in every loaded bundle's resources, the main bundle's
# Contents/Resources included. A build that already made one (swiftbuild)
# is replaced, so the kernels always come from the step above.
rm -rf "$contents/Resources/mlx-swift_Cmlx.bundle"
cp -R "$mlx_bundle" "$contents/Resources/"

# The CLI, inside the app. The cask links these onto PATH from here.
# `turbospark` is the front end the Server pane's Launch buttons run
# (AgentTerminalLaunch.findCLI looks next to the app executable first).
for bin in turbospark turbospark-check turbospark-model turbospark-server; do
  src="$root/target/$target/release/$bin"
  [ -x "$src" ] || { echo "expected $src to exist; run without --skip-build" >&2; exit 1; }
  cp "$src" "$contents/MacOS/$bin"
done

# Dynamic frameworks ride in as SwiftPM binary targets, and SwiftPM drops
# each linked .framework next to the executable in .build/release. Without
# this copy the assembled bundle is missing them entirely and dies in dyld
# before main -- and the DMG verify checks signatures and file presence, not
# launchability, so a missing framework would ship silently. SQLCipher
# predates the updater and shipped with exactly that bug; Sparkle made it
# visible because it was the second dynamic dependency.
for fw in "$build_dir"/*.framework; do
  [ -e "$fw" ] || continue
  cp -R "$fw" "$contents/Frameworks/"
done
for required_framework in SQLCipher.framework Sparkle.framework; do
  [ -d "$contents/Frameworks/$required_framework" ] || {
    echo "missing $required_framework in $build_dir -- the packaged app would fail before main" >&2
    exit 1
  }
done

# SwiftPM bakes rpaths for ITS layout (@loader_path next to the executable in
# .build/release, where SwiftPM also drops the framework). Inside the bundle
# the framework lives one directory up in Contents/Frameworks, which none of
# the baked rpaths cover -- the exe links @rpath/Sparkle.framework and would
# die in dyld before main on a machine that never had this repo's .build.
# Bundle convention is @executable_path/../Frameworks. This edit invalidates
# the exe's existing (build-time) signature, which the codesign pass below
# then re-establishes.
if ! otool -l "$contents/MacOS/TurboSparkApp" | grep -A2 LC_RPATH | grep -q "@executable_path/../Frameworks"; then
  install_name_tool -add_rpath @executable_path/../Frameworks \
    "$contents/MacOS/TurboSparkApp"
fi

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
    <key>NSAppleEventsUsageDescription</key>
    <string>TurboSpark opens Terminal to run the coding agent you launch from the Server pane.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>TurboSpark records microphone audio for meetings and local transcription when you start recording.</string>
    <key>NSAudioCaptureUsageDescription</key>
    <string>TurboSpark captures system and app audio for meeting recordings and local transcription without screen recording.</string>
    <key>NSScreenCaptureUsageDescription</key>
    <string>TurboSpark captures audio from the apps you select for a meeting. Screen frames are not saved.</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>SUEnableAutomaticChecks</key>
    <true/>
    <key>SUFeedURL</key>
    <string>${update_feed_url}</string>
    <key>SUPublicEDKey</key>
    <string>${sparkle_public_ed_key}</string>
</dict>
</plist>
PLIST

printf 'APPL????' > "$contents/PkgInfo"

# The in-app language picker writes AppleLanguages, which AppKit honors only
# for languages the main bundle declares. One list, shared with the SwiftPM
# executable that embeds it: swift/TurboSparkApp/Localization/Info.plist.
plutil -replace CFBundleLocalizations -json \
  "$(plutil -extract CFBundleLocalizations json -o - "$root/swift/TurboSparkApp/Localization/Info.plist")" \
  "$contents/Info.plist"

plutil -lint "$contents/Info.plist" >/dev/null

echo "==> codesign (identity: ${identity})"
# Inner executables first, then the bundle: a signature over the bundle seals
# what is inside it, so signing the wrapper before its contents invalidates
# itself. `--deep` is deprecated by Apple for exactly this reason.
for bin in turbospark turbospark-check turbospark-model turbospark-server openkindd; do
  codesign --force --timestamp=none --sign "$identity" "$contents/MacOS/$bin"
done

# Sparkle's nested helpers BEFORE the framework root that seals them,
# innermost first. The Downloader XPC keeps its entitlements (Sparkle's own
# docs prescribe --preserve-metadata=entitlements for it); the others carry
# none. Sparkle ships these helpers ad-hoc signed, so re-signing with the
# bundle's (default ad-hoc) identity keeps the whole chain one identity.
fw_b="$contents/Frameworks/Sparkle.framework/Versions/B"
if [ -d "$fw_b" ]; then
  for helper in \
    "$fw_b/XPCServices/Installer.xpc" \
    "$fw_b/XPCServices/Downloader.xpc" \
    "$fw_b/Autoupdate" \
    "$fw_b/Updater.app"; do
    codesign --force --timestamp=none --preserve-metadata=entitlements \
      --sign "$identity" "$helper"
  done
fi

# Then every framework root (SQLCipher has no nested code; a root signing
# seals it), then the app itself.
for fw in "$contents/Frameworks"/*.framework; do
  codesign --force --timestamp=none --sign "$identity" "$fw"
done
codesign --force --timestamp=none --sign "$identity" "$app"
codesign --verify --deep --strict "$app"

echo "==> ok: $app ($(du -sh "$app" | cut -f1), version ${version}, id ${bundle_id})"
