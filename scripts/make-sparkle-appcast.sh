#!/usr/bin/env bash
# Builds dist/appcast.xml for one release: the Sparkle 2 feed that turns the
# release's DMG into an in-app update.
#
# WHY A SCRIPT: the same signing step runs in release.yml (from the
# SPARKLE_ED_PRIVATE_KEY secret) and in the local two-version update smoke
# (from a keychain-exported key file), and hand-writing appcast XML in two
# places is how the two feeds drift. generate_appcast reads the DMG's
# CFBundleShortVersionString/CFBundleVersion itself, so the feed cannot
# disagree with the bundle it points at.
#
# Usage:
#   scripts/make-sparkle-appcast.sh [--version X.Y.Z] [--dist DIR]
#       [--download-url-prefix URL] [--sparkle-version V]
#
# The EdDSA private key comes from the first hit:
#   SPARKLE_ED_KEY_FILE     path to an exported key file (local runs)
#   SPARKLE_ED_PRIVATE_KEY  the key file's contents (the CI repo secret)
# With neither, the script fails: an unsigned appcast is an update feed the
# app must refuse, so there is no "--skip-signing" to reach for.
#
# Env:
#   SPARKLE_TOOLS_DIR  directory with Sparkle's bin/ (generate_appcast).
#                      When unset, the pinned release below is downloaded.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dist="$root/dist"
version=""
url_prefix=""
sparkle_version="2.10.0"

while [ $# -gt 0 ]; do
  case "$1" in
    --version) version="$2"; shift 2 ;;
    --dist) dist="$2"; shift 2 ;;
    --download-url-prefix) url_prefix="$2"; shift 2 ;;
    --sparkle-version) sparkle_version="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Same single-version rule as make-app-bundle.sh: a hand-run appcast must not
# be able to disagree with what CI would have produced from the same tree.
if [ -z "$version" ]; then
  version="$(sed -nE 's/^version = "([^"]+)".*/\1/p' "$root/Cargo.toml" | head -1)"
fi
[ -n "$version" ] || { echo "could not resolve version from Cargo.toml" >&2; exit 1; }

if [ -z "$url_prefix" ]; then
  url_prefix="https://github.com/whit3rabbit/turbospark/releases/download/v${version}/"
fi

dmg="$dist/TurboSpark-${version}-arm64.dmg"
[ -f "$dmg" ] || { echo "expected $dmg to exist; run make app-bundle && make dmg first" >&2; exit 1; }

key_file="${SPARKLE_ED_KEY_FILE:-}"
tmp_key=""
cleanup() {
  [ -n "$tmp_key" ] && rm -f "$tmp_key"
  rm -rf "$stage_dir"
}
trap cleanup EXIT
stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/turbospark-appcast.XXXXXX")"

if [ -z "$key_file" ]; then
  [ -n "${SPARKLE_ED_PRIVATE_KEY:-}" ] || {
    echo "need SPARKLE_ED_KEY_FILE or SPARKLE_ED_PRIVATE_KEY (see the header comment)" >&2
    exit 1
  }
  tmp_key="$(mktemp "${TMPDIR:-/tmp}/turbospark-ed25519.XXXXXX")"
  chmod 600 "$tmp_key"
  printf '%s' "$SPARKLE_ED_PRIVATE_KEY" > "$tmp_key"
  key_file="$tmp_key"
fi

if [ -n "${SPARKLE_TOOLS_DIR:-}" ]; then
  tools="$SPARKLE_TOOLS_DIR"
  [ -x "$tools/bin/generate_appcast" ] || {
    echo "SPARKLE_TOOLS_DIR=$tools has no bin/generate_appcast" >&2
    exit 1
  }
else
  tools="${TMPDIR:-/tmp}/turbospark-sparkle-tools-${sparkle_version}"
  if [ ! -x "$tools/bin/generate_appcast" ]; then
    echo "==> fetching Sparkle ${sparkle_version} tools"
    mkdir -p "$tools"
    curl -fsSL --retry 3 \
      -o "$tools/sparkle.tar.xz" \
      "https://github.com/sparkle-project/Sparkle/releases/download/${sparkle_version}/Sparkle-${sparkle_version}.tar.xz"
    tar -xf "$tools/sparkle.tar.xz" -C "$tools"
  fi
fi

# generate_appcast stages the artifacts it indexes; the copy keeps the real
# dist/ clean and makes the feed's enclosures the ONLY thing in the stage,
# so a stray file cannot sneak into the feed.
cp "$dmg" "$stage_dir/"

echo "==> generate_appcast (version ${version})"
"$tools/bin/generate_appcast" "$stage_dir" \
  --ed-key-file "$key_file" \
  --download-url-prefix "$url_prefix" \
  -o "$dist/appcast.xml"

# Fail loudly rather than shipping a feed the app will refuse at the
# signature check: the edSignature attribute is the whole point of this file.
grep -q 'sparkle:edSignature=' "$dist/appcast.xml" || {
  echo "appcast.xml has no sparkle:edSignature; refusing to ship it" >&2
  exit 1
}
grep -q "url=\"${url_prefix}" "$dist/appcast.xml" || {
  echo "appcast.xml does not point at ${url_prefix}; refusing to ship it" >&2
  exit 1
}

echo "==> ok: $dist/appcast.xml"
