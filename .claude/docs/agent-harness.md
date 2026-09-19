# Agent harness and repository workflow

This page contains occasional agent-harness, code-discovery, measurement-script, and PR workflow detail moved out of the always-loaded root instructions.

## References and measurement scripts

`scripts/` holds the measurement surfaces that cannot be a `cargo test`:
two need the Swift engine built next door, `kld.py` needs a 14.6 GB
reference checkpoint plus a Python environment, and `kld_llamacpp.py` needs
a 26.9 GB GGUF plus a llama.cpp install (brew's; it compiles
`llamacpp_logits.c` against that header on first run and caches the binary
in `/tmp`). `kld.py` runs mlx-lm under `uv run --with mlx-lm`, an ephemeral
env, so no Python dependency is installed globally or enters this
workspace; `kld_llamacpp.py` needs only numpy and reuses `kld.py`'s
divergence and perplexity functions rather than restating them.

**FINDING A REFERENCE: CHECK mlx-vlm AS WELL AS mlx-lm, AND CHECK WHETHER THE
COMPONENT SHIPS ALONE.** Every script above uses mlx-lm, which makes it the
obvious place to look and is not always the right one: mlx-lm 0.31.3 has no
`qwen3_5_mtp`, while mlx-vlm 0.6.14 implements it at
`speculative/drafters/qwen3_5_mtp/`. A sub-component may also be published as
its own checkpoint (`mlx-community/Qwen3.8-27B-MTP-4bit`, 239 MB), far cheaper
to load than its parent and named by the config's `model_type`. READING a
reference settles convention questions that measuring them cannot -- 2026-08-14
for mrope, 2026-08-18 for the MTP head's five design choices. Note
`safetensors.numpy` CANNOT decode BF16 (`TypeError: data type 'bfloat16' not
understood`); parse the container directly, which is ~15 lines and keeps the
decoder independent anyway (Gotcha 48).
And READ THE PRIOR ART YOUR OWN DOCS NAME. A "take no source" note is a
LICENSING decision about copying and never an instruction not to look: the MTP
head's norm convention sat one grep away in the project whose headline result
`docs/MTP_SPECULATIVE.md`'s first sentence quotes, and was rediscovered by
two hours of bisection instead.
**AND A TRUE FACT ABOUT A REFERENCE IS NOT A READING OF IT.** The control
vector numbering was DERIVED from two correct facts about llama.cpp and came
out one block off, under an honest `UNVERIFIED` that made it look
measured-open rather than reasoned-and-wrong; the refutation was internal from
the first commit (`crates/repack/AGENTS.md` Gotcha 11). Check a derived
convention against the line that IMPLEMENTS it -- brew ships llama.cpp's
headers to `/opt/homebrew/include` and its sources are one
`raw.githubusercontent.com` fetch away, so this class of question costs no
download at all.

## Code discovery, search, and explore agent harness

This repository uses Syntext as its indexed code search engine (see `swift/docs/SYNTEXT.md` and `swift/docs/SWIFT_TOOLS.md`).

- **Default search tool**: Use `grep_search` for code discovery, symbol lookup, and pattern matching. It uses the project's Syntext index for sub-millisecond regex and literal search with line numbers and context lines in ripgrep format.
- **Do not shell out for searching**: Never shell out to `grep`, `find`, or `ripgrep` via Bash or terminal execution when `grep_search` or `Glob` is available.
- **File locating and reading**: Use `Glob` (`list_directory`) for file path patterns and `FileRead` (`read_file`) for inspecting specific files or line ranges.
- **Explore subagent contract**:
  - The `explore` agent (`AgentManager+BuiltIns.swift` and `.turbospark/agents/explore.md`) is strictly read-only.
  - Allowed tools: `FileRead`, `Glob`, `Grep`, `grep_search`, `Bash` (strictly read-only commands: `ls`, `git status`, `git log`, `git diff`), `WebFetch`, `WebSearch`.
  - Disallowed tools: all write and edit tools (`write_file`, `edit_file`, `apply_patch`, `notebook_edit`), subagent creation (`agent`, `subagent`, `task`), planning mode tools (`enter_plan_mode`, `exit_plan_mode`), and artifact/worktree tools (`todowrite`, `enter_worktree`, `exit_worktree`).
  - Project instructions are omitted (`omitsProjectInstructions` / `omit_claude_md: true`) to preserve context window and reduce prefill latency for fast search fan-out.
  - All findings must report absolute paths and avoid emojis.
- **Plan subagent contract**:
  - The `plan` agent (`AgentManager+BuiltIns.swift` and `.turbospark/agents/plan.md`) is strictly read-only for designing architectural and implementation plans.
  - Allowed tools: `FileRead`, `Glob`, `Grep`, `grep_search`, `Bash` (strictly read-only commands: `ls`, `git status`, `git log`, `git diff`), `WebFetch`, `WebSearch`.
  - Disallowed tools: all write and edit tools (`write_file`, `edit_file`, `apply_patch`, `notebook_edit`), subagent creation (`agent`, `subagent`, `task`), planning mode tools (`enter_plan_mode`, `exit_plan_mode`), and artifact/worktree tools (`todowrite`, `enter_worktree`, `exit_worktree`).
  - Structured process: Understand Requirements, Explore Thoroughly (using Syntext `grep_search` and `Glob`), Design Solution, Detail the Plan.
  - Required output ending: Concludes with "### Critical Files for Implementation" listing 3-5 critical files. All findings avoid emojis.
- **General-purpose subagent contract**:
  - The `general-purpose` agent (`AgentManager+BuiltIns.swift`, `.turbospark/agents/general-purpose.md`, and `.claude/agents/general-purpose.md`) is the fallback execution and research subagent.
  - Unlike `explore` and `plan`, it is unconstrained (has no tool ceiling and no disallowed tools), enabling multi-step task execution, file editing, and command running.
  - Retains project instructions (`omitsProjectInstructions == false`) for full codebase and architectural context.
  - Uses Syntext `grep_search` for fast indexed code and pattern search across large codebases.
  - Completes tasks fully without gold-plating or leaving half-done, returning a concise report with essentials.

## PR batch merge workflow

- Treat every PR head as untrusted code. Do not check it out, merge it into a local
  checkout, or run Cargo, tests, scripts, or other PR-provided code on a maintainer
  host before it has passed review and been merged remotely.
- Validate each PR with the required GitHub Actions checks on an isolated
  GitHub-hosted runner without maintainer credentials. A local build is not a
  substitute for those checks.
- Run `cargo build --workspace` on the trusted `main` branch once before opening
  the queue, and again only after reviewed PRs have been merged remotely and the
  updated `main` has been fetched.
- Process a deterministic PR list one by one.
- Merge remotely only after review approval and all required CI checks pass:
  - `gh pr merge <num> --merge --delete-branch` for clean merges.
  - `gh pr merge <num> --auto --merge --delete-branch` when queueing through conflict checks.
- For merge conflicts or CI failures, record the PR as blocked; do not locally
  merge it for diagnosis or silently stack additional PRs on a broken head.
- Use `git revert -m 1 <merge-sha>` to back out a bad merge commit after fetching
  trusted `main` (safe rollback) instead of destructive history rewrites.
- If repeated build failures show the same pre-existing error in untouched files, treat it as queue- or branch-state health and stop attributing one-to-one PR blame.
- Record each PR status and first failure line in the task log; skip and continue on blocked PRs.
- If repeated build failures show the same pre-existing error in untouched code, treat it as a shared branch health issue and stop attributing it to each PR individually.

<!-- BEGIN AGENT-CONFIG:mf -->
Before exploring this codebase, run `mf search "<question>" --field notes`.
Before finishing, write what you learned as a page with `mf write <draft> --field notes`, or stage it with `mf raw add --field notes`.
<!-- END AGENT-CONFIG:mf -->
