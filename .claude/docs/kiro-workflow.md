# Kiro spec-driven workflow

Moved from the root AGENTS.md. Read it before spec or steering work. The skills
live in `.agents/skills/kiro-*/SKILL.md` and carry the detailed protocols.

## Paths

- Steering: `.kiro/steering/` holds project-wide policy (architecture,
  naming, security, tech stack, API standards). Default files are `product.md`,
  `tech.md`, and `structure.md`. Load them for spec and implementation work,
  and load extra steering only when the task needs it.
- Specs: `.kiro/specs/<feature>/` holds one spec per feature, with its own
  notes. Check it for active specifications.
- Folder-level context belongs in a local `AGENTS.md`, not in steering.

## Language

Markdown written into a spec (`requirements.md`, `design.md`, `tasks.md`,
`research.md`, validation reports) uses the language set in `spec.json.language`.

## Workflow

- Phase 0 (optional): `$kiro-steering`, `$kiro-steering-custom`.
- Discovery: `$kiro-discovery "idea"` picks the action path and writes
  `brief.md` and `roadmap.md` for multi-spec projects.
- Phase 1, specification:
  - Single spec: `$kiro-spec-quick {feature} [--auto]`, or step by step:
    - `$kiro-spec-init "description"`
    - `$kiro-spec-requirements {feature}`
    - `$kiro-validate-gap {feature}` (optional, for an existing codebase)
    - `$kiro-spec-design {feature} [-y]`
    - `$kiro-validate-design {feature}` (optional design review)
    - `$kiro-spec-tasks {feature} [-y]`
  - Multi-spec: `$kiro-spec-batch` creates every spec in `roadmap.md` in
    parallel, by dependency wave.
- Phase 2, implementation: `$kiro-impl {feature} [tasks] [--review required|inline|off]`.
  - Without task numbers: autonomous mode (a subagent per task, independent
    review, final validation).
  - With task numbers: manual mode (tasks run in the main context, still
    reviewer-gated before completion).
  - `--review off` skips task-local review. Use it deliberately and keep
    `$kiro-validate-impl {feature}` as the final quality gate.
- Re-validate any time with `$kiro-validate-impl {feature}`. Check progress with
  `$kiro-spec-status {feature}`.

## Supporting skills

- `kiro-review`: task-local adversarial review used by reviewer subagents.
- `kiro-debug`: root-cause-first debugging used by debugger subagents.
- `kiro-verify-completion`: fresh-evidence gate before any success claim.

Keep steering current and check it against `$kiro-spec-status`.
