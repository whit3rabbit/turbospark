#!/usr/bin/env bash
# Zips the tslab skill (crates/lab/skill) with the release tslab and
# tslab-worker binaries and the lab's workloads, so an agent on another Apple
# Silicon Mac can measure TurboSpark installs without the source tree. It is
# the same skill a checkout links: its doctor script picks bin/ when present.
#
# The zip unpacks to one `tslab/` folder, which is a skill directory: put it
# in ~/.claude/skills/ or ~/.agents/skills/ (scripts/install-tslab-skill.sh --zip does that), or
# upload the zip where a skill zip is accepted. The binaries are unsigned:
# a browser download is quarantined, and the skill's doctor script says so.
#
# Usage:
#   scripts/package-tslab-skill.sh [--out DIR] [--no-build]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="$root/target/dist"
build=1

while [ $# -gt 0 ]; do
    case "$1" in
        --out) out_dir="$2"; shift 2 ;;
        --no-build) build=0; shift ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "package-tslab-skill: builds macOS arm64 binaries; run it on Apple Silicon" >&2
    exit 1
fi

cd "$root"
if [ "$build" = 1 ]; then
    cargo build --release -p turbospark-lab
fi
for b in tslab tslab-worker; do
    [ -x "target/release/$b" ] || { echo "target/release/$b missing (drop --no-build)" >&2; exit 1; }
done

rev="$(git rev-parse --short HEAD)"
dirty=""
git diff --quiet HEAD -- crates || dirty="-dirty"
name="tslab-skill-${rev}${dirty}-macos-arm64"
stage="$root/target/tslab-skill-stage"
skill="$stage/tslab"

# The staging directory belongs to this script alone.
rm -rf "$stage"
mkdir -p "$skill/bin" "$skill/references" "$skill/scripts" "$skill/schema" "$out_dir"

cp crates/lab/skill/SKILL.md "$skill/SKILL.md"
cp crates/lab/skill/scripts/doctor.sh crates/lab/skill/scripts/preflight.sh "$skill/scripts/"
chmod +x "$skill/scripts/doctor.sh" "$skill/scripts/preflight.sh"
# All references ship, so SKILL.md's links resolve; it marks extending.md
# as checkout only.
cp crates/lab/skill/references/*.md "$skill/references/"
cp target/release/tslab target/release/tslab-worker "$skill/bin/"
cp crates/lab/schema/tslab-1.schema.json "$skill/schema/"
cp LICENSE "$skill/LICENSE"

# Workflows find their jobs in workloads/ beside bin/ (workflow::assets_dir).
cp -R crates/lab/workloads "$skill/workloads"
# The chat workload names its document relative to itself: ship the document
# beside it and point there.
mkdir -p "$skill/chats"
cp docs/MTP.md "$skill/chats/MTP.md"
python3 -I - "$root/crates/lab/chats/long-doc.chat.json" "$skill/chats/long-doc.chat.json" <<'PY'
import json, sys
chat = json.load(open(sys.argv[1]))
chat["document"] = "MTP.md"
json.dump(chat, open(sys.argv[2], "w"), indent=2)
open(sys.argv[2], "a").write("\n")
PY

{
    echo "tslab skill build"
    echo "source: TurboSpark $(git rev-parse HEAD)${dirty}"
    echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "rustc: $(rustc --version)"
    echo "platform: macOS arm64"
    echo
    (cd "$skill" && shasum -a 256 bin/tslab bin/tslab-worker)
} > "$skill/BUILD.txt"

zip_path="$out_dir/$name.zip"
rm -f "$zip_path"
(cd "$stage" && zip -qrX "$zip_path" tslab)

echo "tslab skill: $zip_path"
echo "  $(du -h "$zip_path" | cut -f1)  sha256 $(shasum -a 256 "$zip_path" | cut -d' ' -f1)"
if [ -n "$dirty" ]; then
    echo "  note: built from uncommitted changes under crates/; BUILD.txt says so" >&2
fi
