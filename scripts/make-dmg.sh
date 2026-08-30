#!/usr/bin/env bash
# Wraps a built TurboSpark.app into the release DMG, then MOUNTS IT AND CHECKS
# what came out. The check is the point: `hdiutil create` succeeds on a
# staging directory that is missing the resource bundle or carries a broken
# signature, and the failure surfaces as an app that launches with no logos or
# refuses to open at all, on a user's machine, weeks later.
#
# Usage:
#   scripts/make-dmg.sh [--app PATH] [--out DIR] [--version X.Y.Z]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="$root/dist"
app=""
version=""

while [ $# -gt 0 ]; do
  case "$1" in
    --app) app="$2"; shift 2 ;;
    --out) out_dir="$2"; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$(uname -s)" != "Darwin" ]; then
  echo "make-dmg.sh is macOS only" >&2
  exit 1
fi

[ -n "$app" ] || app="$out_dir/TurboSpark.app"
[ -d "$app" ] || { echo "no app bundle at $app; run scripts/make-app-bundle.sh first" >&2; exit 1; }

if [ -z "$version" ]; then
  version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist")"
fi
[ -n "$version" ] || { echo "could not resolve version" >&2; exit 1; }

# README.md and both casks name this file. Changing it is a three-file change.
dmg="$out_dir/TurboSpark-${version}-arm64.dmg"

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

cp -R "$app" "$staging/TurboSpark.app"
# The drag-to-install target. Without it the DMG opens onto a lone app icon
# and the user is expected to know where to put it.
ln -s /Applications "$staging/Applications"

# DMG custom volume icon if available
if [ -f "$root/assets/icons/AppIcon.icns" ]; then
  cp "$root/assets/icons/AppIcon.icns" "$staging/.VolumeIcon.icns"
  if command -v SetFile >/dev/null 2>&1; then
    SetFile -a C "$staging" || true
    SetFile -a V "$staging/.VolumeIcon.icns" || true
  fi
fi

echo "==> hdiutil create $dmg"
mkdir -p "$out_dir"
rm -f "$dmg"
hdiutil create \
  -volname "TurboSpark ${version}" \
  -srcfolder "$staging" \
  -fs HFS+ \
  -format UDZO \
  -quiet \
  "$dmg"

echo "==> hdiutil verify"
hdiutil verify -quiet "$dmg"

echo "==> mount and check contents"
mount_point="$(mktemp -d)"
hdiutil attach "$dmg" -readonly -nobrowse -noautoopen -quiet -mountpoint "$mount_point"
# Detach even if a check below fails, or the next run inherits a stale mount.
trap 'hdiutil detach "$mount_point" -quiet -force >/dev/null 2>&1 || true; rm -rf "$staging" "$mount_point"' EXIT

mounted="$mount_point/TurboSpark.app"
[ -d "$mounted" ] || { echo "TurboSpark.app missing from the image" >&2; exit 1; }
[ -L "$mount_point/Applications" ] || { echo "/Applications symlink missing from the image" >&2; exit 1; }

# Each of these has its own way of going missing, so each is named.
[ -x "$mounted/Contents/MacOS/TurboSparkApp" ] || { echo "main executable missing" >&2; exit 1; }
for bin in turbospark-check turbospark-model turbospark-server; do
  [ -x "$mounted/Contents/MacOS/$bin" ] || { echo "CLI binary $bin missing from the bundle" >&2; exit 1; }
done
# Bundle.module's resources. An app without these starts and then renders no
# provider logos and no built-in prompts -- it does not crash, which is why
# this is asserted rather than left to a launch test.
ls "$mounted/Contents/Resources"/*.bundle >/dev/null 2>&1 \
  || { echo "no resource .bundle in Contents/Resources" >&2; exit 1; }
codesign --verify --strict "$mounted" \
  || { echo "signature does not verify on the mounted copy" >&2; exit 1; }

echo "==> ok: $dmg ($(du -h "$dmg" | cut -f1))"
