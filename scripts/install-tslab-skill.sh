#!/usr/bin/env bash
# Installs a tslab skill for Claude Code.
#
#   --repo           Link this checkout's repository skill
#                    (crates/lab/skill, name turbospark-lab) into
#                    .claude/skills/turbospark-lab. .claude/ is not tracked, so
#                    each checkout or worktree links it once.
#   --zip PATH       Unpack a portable skill zip (scripts/package-tslab-skill.sh)
#                    into DEST/tslab. DEST defaults to ~/.claude/skills.
#   --dest DIR       Where --zip installs.
#   --force          Replace an existing install.
#
# Usage:
#   scripts/install-tslab-skill.sh --repo
#   scripts/install-tslab-skill.sh --zip target/dist/tslab-skill-<rev>-macos-arm64.zip
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode=""
zip=""
dest="$HOME/.claude/skills"
force=0

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) mode=repo; shift ;;
        --zip) mode=zip; zip="$2"; shift 2 ;;
        --dest) dest="$2"; shift 2 ;;
        --force) force=1; shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$mode" in
    repo)
        target="$root/.claude/skills/turbospark-lab"
        mkdir -p "$root/.claude/skills"
        if [ -e "$target" ] || [ -L "$target" ]; then
            if [ "$force" = 1 ] && [ -L "$target" ]; then
                rm "$target"
            else
                echo "$target exists (pass --force to replace a link; a real directory is left alone)" >&2
                exit 1
            fi
        fi
        ln -s ../../crates/lab/skill "$target"
        echo "linked $target -> crates/lab/skill"
        ;;
    zip)
        [ -f "$zip" ] || { echo "no such zip: $zip" >&2; exit 1; }
        target="$dest/tslab"
        if [ -e "$target" ]; then
            if [ "$force" = 1 ]; then
                rm -rf "$target"
            else
                echo "$target exists (pass --force to replace it)" >&2
                exit 1
            fi
        fi
        mkdir -p "$dest"
        tmp="$(mktemp -d)"
        trap 'rm -rf "$tmp"' EXIT
        unzip -q "$zip" -d "$tmp"
        [ -f "$tmp/tslab/SKILL.md" ] || { echo "$zip does not hold a tslab/ skill folder" >&2; exit 1; }
        mv "$tmp/tslab" "$target"
        echo "installed $target"
        "$target/scripts/doctor.sh" || true
        ;;
    *)
        echo "give --repo or --zip PATH (see --help)" >&2
        exit 2
        ;;
esac
