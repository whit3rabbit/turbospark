# tslab skill

## Overview
The agent skill that packages `tslab` (SKILL.md, evals, references) and its host preflight scripts. It is either unpacked from a zip beside `bin/`, or used in place as `crates/lab/skill` in a checkout [skill/scripts/doctor.sh:MODE]. The folder holds no Rust source.

## Subdirectories & Boundaries
- scripts: [./scripts/EGM.md](./scripts/EGM.md) - read-only `doctor.sh` and `preflight.sh` checks.
- Other assets (not contract-bearing): `SKILL.md`, `evals/`, `references/`.
