# Stage 7: Critic Review report (OC-11)

Target: docs/designs/opencode2-parity/OC-11-formatter-pass.md (full file). Standard: feature-spec-standard for design and structural. One round only, no remediation, so convergence was not assessed. Reviewers egm-design (DC), egm-structural (ST), egm-style (SY) all completed.

## Design (DC): 2 Blocker, 9 Major, 4 Nit

DC-1 (Blocker, whole doc, 36-166). No Alternatives Considered section. Unweighed choices: in-process pass vs model-run formatter, shared hook seam with OC-10 vs bespoke hook, built-in registry vs user-only rows, format-on-write vs turn end, project toggle vs per-formatter toggle. Fix: add the section with at least four rejected options and concrete reasons.

DC-2 (Blocker, 117-128). No Risks and Mitigations section; failure modes are scattered across NFRs. Fix: add a table covering DC-3 to DC-8.

DC-3 (Major, 12-14, 55-56, 121-124). "Toggle plus write permission cover the act" is asserted. Write permission approves content, not running another executable. prettier, ruff, rustfmt read project config (.prettierrc and JS variants, rustfmt.toml, pyproject.toml, .clang-format) that can execute code, and the agent can write those files the same turn. Fix: state the ladder position, refuse the pass for a config-file write in the same turn or after a config change since enabling, and require per-command approval for user rows.

DC-4 (Major, 45-46, 134-136). Login-shell PATH resolution unspecified: how spawned, cached, how it differs from shellEnvironment(); repo-local binaries (node_modules/.bin) are not on PATH; a relative path in a user row could let the agent plant the formatter. Fix: define resolution order, forbid in-project executables unless an absolute path is configured, cache with invalidation.

DC-5 (Major, 48-56, 99-100). In-place rewrite lacks atomicity and race handling (timeout mid-write, user editor, file watcher, next edit's staleness hash). Fix: format a temp copy, atomic replace on exit 0 only, re-sync the edit tool's staleness record, state behavior if the file changed during the run.

DC-6 (Major, 51-53, 85-86). What the model sees is unspecified; its next str_replace uses old text. Error tail has no size cap and no ANSI stripping (ShellOutputFormatting.stripANSI and compact exist) and is untrusted output. Fix: say what the model receives, cap and sanitize via ShellOutputFormatting.

DC-7 (Major, 112, 119-120). "Bounded time" has no number, kill policy, output cap, sync/async, cancellation, or concurrency bound. Fix: specify timeout, SIGTERM then SIGKILL, output cap, cancel on turn stop, bounded parallelism.

DC-8 (Major, 41-47, 138-139; INDEX line 83). OC-10 dependency stated in INDEX but not in the doc; ordering undefined (diagnostics must run after the formatter). Fix: add a Dependencies line on OC-10's seam and define format first, then diagnostics.

DC-9 (Major, 38-41, 70-73). Registry underspecified: no command lines (swift-format format -i, gofmt -w, clang-format -i), swift-format is not guaranteed present (Xcode 16+), clang-format usually absent, no precedence for multiple matches, no handling of extensionless or shebang files. Fix: concrete v1 row table, precedence rule, match key.

DC-10 (Major, 55-56). Project-root and "written this turn" rules have no enforcement: no symlink or path-traversal check (reuse PathContainment), no exclude for ignored, vendored, generated, minified or lock files. Fix: require PathContainment with symlink resolution and an exclude list.

DC-11 (Major, 94-100). No acceptance criteria beyond a manual journey; tests cover only the availability gate (none for timeout, partial write, failure path, root bound, toggle off); CUJ 2 contradicts line 44 ("pass log" undefined). Fix: define or drop the pass log; test each CUJ and safety property.

DC-12 (Nit, 125-126). Names Localizable.xcstrings but no gate command. Fix: name the make target.
DC-13 (Nit, 3-8). Parity links point at `dev`, which drifts; INDEX audit commit is 7ce35879b8. Fix: pin to that commit.
DC-14 (Nit, 19-23). "Small and high-leverage" and "OpenCode formats after every agent write" lack evidence or a quoted code path. Fix: add one sentence of evidence or soften.
DC-15 (Nit, 144-145). No schema, migration, or statement that a missing key decodes to off. Fix: add one line.

## Structural (ST): 2 Blocker, 4 Major, 4 Nit

ST-1 (Blocker, whole file). No Alternatives Considered; open questions are not rejections. Missing: format on save/watch, leave to model or explicit tool, whole-project pass, LSP formatting via OC-10, bundled formatter. Fix: add H2 with reasons.
ST-2 (Blocker, whole file). No Risks and Mitigations; only a Safety bullet (121-124). Unaddressed: write/format race, stale undo or checkpoint, transcript vs model content, hangs and huge files, user commands as code-exec vector (storage location unsaid), symlinks escaping root (line 56). Fix: add H2 with mitigations.
ST-3 (Major, 28, 31). Relative links to swift/TurboSparkApp/... resolve under docs/designs/opencode2-parity/ and point nowhere. Fix: prefix with ../../../.
ST-4 (Major, 38-41 vs 70-73, 48 vs 74-76). Goals state decided what Open Questions leave open. Fix: mark Goals as proposed or resolve the questions.
ST-5 (Major, 102-115, 140-141). No reference to OC-10's post-edit seam (INDEX 83-84). Fix: add Dependencies note, make item 3 consume the seam.
ST-6 (Major, 121-124). No AppToolPermissionEngine ladder position (INDEX 112-114). Fix: state it and whether the formatter run is a separate approval.
ST-7 (Nit, 104-115). Diagram checks match and availability before the toggle; failure branch crammed. Fix: toggle first, split branches.
ST-8 (Nit, 134-136, 144-145). Registry file and persistence item both own "per-project records". Fix: name one owner.
ST-9 (Nit, 65). "format-on-agent-prompt" unclear. Fix: "no formatting of files touched only by read or other non-edit tools".
ST-10 (Nit, INDEX 95, 99). OC-11 listed in step 3 before OC-10 and again in step 7. Fix: remove from step 3 or split out the seam step.
(The structural reviewer read only lines 1-136 of OPENCODE2_COMPARISON.md; heading hierarchy and blocks are well formed.)

## Style (SY): 0 Blocker, 5 Major, 8 Nit

SY-1 (Major, 38-41 vs 70-73). A decision is stated settled and listed open; unclear whether prettier, ruff, rustfmt are shipped rows or examples. Fix: point Goals to the Open Question; mark them examples.
SY-2 (Major, 42-43 vs 49-50, 121-124). "Login-shell PATH" appears to conflict with "no shell". Fix: say a PATH value is captured once from the login shell and the formatter run is an argument vector with no shell.
SY-3 (Major, 49, 112, 119-120). Time bound has no value. Fix: give seconds or an open question with a default.
SY-4 (Major, 121-124). Safety claim outruns evidence; hardening rules unnamed; no ladder position; the rewrite is a second write the approved edit did not perform. Fix: name the rules by file, state permission treatment and ladder position.
SY-5 (Major, INDEX 83-84, item 3 at 140). OC-10 dependency missing; INDEX steps 3 and 7 inconsistent. Fix: Dependencies note, "register with the OC-10 seam".
SY-6 (Nit, 19-21). "Guaranteed ... high-leverage" is subjective and conflicts with silent skip. Fix: replace with a measurable statement.
SY-7 (Nit, 26-32). Absence claim not calibrated. Fix: "No pipeline found by grep (audit 2026-10-31)".
SY-8 (Nit, 44, 118). "Pass log" undefined. Fix: define or drop.
SY-9 (Nit, 51-53, 61, 99-100). "Short note", "error tail", "one-line" lack limits. Fix: specify a cap.
SY-10 (Nit, 55, 65). Ambiguous scope wording. Fix: "format only paths returned by successful edit-tool calls in the current turn".
SY-11 (Nit, 48, 74, 108). Trigger named three ways; "extension/language" vs "extension or language". Fix: one term each.
SY-12 (Nit, 70-73). "Lean:" and "toolchain-adjacent" are informal. Fix: "Recommendation:" and plainer wording.
SY-13 (Nit, 117-120, 146-151). "At most the formatter's own runtime" ignores PATH resolution and spawn; no runner tests. Fix: calibrate the claim, add FormatterRunnerTests. (Also line 134 vs 144: split registry and persistence responsibilities.)
(The style reviewer did not verify Swift paths or OpenCode links, nor open OPENCODE2_COMPARISON.md.)
