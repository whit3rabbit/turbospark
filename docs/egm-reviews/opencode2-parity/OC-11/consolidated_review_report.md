# Consolidated review report: OC-11 formatter pass

Target: docs/designs/opencode2-parity/OC-11-formatter-pass.md (full file, 164 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed (the first Stage 5 spawn attempts hit the concurrent subagent limit and succeeded on retry, one at a time).
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.
Caveat: ST and IC read only lines 1-136 of OPENCODE2_COMPARISON.md; SY did not open it. Reuse targets named below (LoginShellPath, ProcessExecutor, PathContainment) were found by code search in Stage 5 and not re-verified later.

Counts after deduplication: Blocker 2, Major 10, Nit 9.

## Blocker

B1. Whole file (36-76, 102-115). No Alternatives Considered section; the standard fails a single-path spec. Fix: add three or four bullets with one-line rejection reasons (leave to model, shared OC-10 seam vs bespoke hook, swift-format only vs user rows, turn-end vs per-edit). [DC-1, ST-1, S8-B1]
B2. Lines 117-128. No Risks and Mitigations section. Fix: add a compact table (stale model view, config-file code execution, write race, hang, root escape). [DC-2, ST-2, S8-B2]

## Major

M1. Lines 18-22, 134-141. OC-10 post-edit seam and reuse targets not named; INDEX lists OC-11 in step 3 and step 7. Fix: add a Dependencies line (register on OC-10's seam, run before diagnostics, reuse LoginShellPath and ProcessExecutor) and fix INDEX ordering. [ZO-1, ZO-2, IC-6, DC-8, ST-5, ST-10, SY-5, S8-M1]
M2. Lines 18-19, 45-47, 119-124. Safety claim asserted; no AppToolPermissionEngine ladder position; user rows could plant commands; formatter config can execute code. Fix: state the pass is app-initiated outside the ladder, user rows live in the app archive not the repo, toggle is the trust grant, config is trusted input; record residual risk. [ZO-3, IC-5, DC-3, ST-6, SY-4, CO-9, S8-M2]
M3. Lines 38-41, 70-73. Goals state decided what Open Questions leave open; v1 scope too wide; command lines missing. Fix: resolve now (swift-format only, fixed command, extension match, user rows as examples). [ZO-4, IC-8, DC-9, ST-4, SY-1, S8-M3]
M4. Lines 48-56, 99-100. What the model sees after the rewrite is unspecified; next edit may anchor on stale content; undo and OC-10 ordering unstated. Fix: tool result carries post-format content, refresh the edit tool's recorded hash, undo captures pre-format write. [ZO-5, IC-1, IC-9, CO-6, DC-6, S8-M4]
M5. Lines 42-44, 49-50. "Login-shell PATH" appears to conflict with "no shell"; toolchain and project-local binaries unresolved. Fix: capture PATH once via LoginShellPath, cache, resolve absolute executable, run with an argument vector, no in-project relative executables in v1. [IC-3, CO-2, DC-4, SY-2, S8-M5]
M6. Lines 48-53, 99-100. In-place rewrite is not atomic and can race with user saves. Fix: format a temp copy, replace on exit 0 only, skip with a failure note if the file hash changed. [IC-4, DC-5, S8-M6]
M7. Lines 49, 112, 119-120. "Bounded time" has no value, kill, or output cap. Fix: give a timeout (for example 10 s), kill on expiry, cap error tail, cancel on turn stop. [CO-1, DC-7, SY-3, S8-M7]
M8. Lines 44, 88-93, 118. "Pass log" is undefined and contradicts CUJ 2. Fix: drop it; a missing formatter is a silent skip. [CO-7, DC-11, SY-8, IC-8, S8-M8]
M9. Lines 55-56. "Files written this turn" and project-root rule have no mechanism. Fix: define as paths from successful edit-tool calls; require PathContainment with symlink resolution. [CO-5, CO-8, IC-2, DC-10, SY-10, S8-M9]
M10. Lines 28, 31, 48, 74-76. Triggering tools unnamed (named three ways); relative Swift links resolve to nowhere. Fix: list concrete tool names, resolve the Open Question, prefix links with ../../../. [CO-4, SY-11, ST-3, S8-M10, S8-N3]

## Nit

N1. Lines 26-31. ShellOutputFormatting citation is irrelevant; TerminalCommandClassifier already auto-approves formatters. Fix: drop one, note the other; calibrate "no pipeline" as grep evidence. [ZO-6, SY-7, S8-N1]
N2. Lines 51-54, 99-100. Success note adds noise and localization cost; error tail uncapped. Fix: note on failure only, capped and ANSI-stripped via ShellOutputFormatting. [ZO-8, SY-9, DC-6, S8-N2]
N3. Lines 125-128, 146-151, 155-157. Program constraints and test mutation detail over-specified; runner has no tests. Fix: one-line references to INDEX.md; add one FormatterRunnerTests line (timeout, failure keeps original, root bound). [ZO-7, CO-10, CO-11, DC-12, SY-13, S8-N4, S8-M13]
N4. Lines 19-21, 65, 70-73. Wording: "guaranteed", "Lean:", "format-on-agent-prompt", "toolchain-adjacent". Fix: measurable wording, "Recommendation:", plainer phrases. [SY-6, SY-12, ST-9, S8-N5]
N5. Lines 144-145. No statement that a missing persistence key decodes to off. Fix: add one line. [DC-15, S8-N6]
N6. Lines 134-136, 144-145. Registry and persistence both own "per-project records". Fix: name one owner. [ST-8, S8-N7]
N7. Lines 104-115. Diagram checks toggle after match; failure branch crammed. Fix: toggle first, split branches. [ST-7]
N8. Lines 3-8, 19-23. Parity links point at drifting `dev`; motivation uncited. Fix: pin to 7ce35879b8 and soften or cite. [DC-13, DC-14]
N9. Lines 51-52, 99-100. A broken formatter pays its full bound and appends an error tail on every write. Fix: optional per-formatter circuit breaker (S8 recommends rejecting for v1). [IC-7]

## Stage 8 scope guidance

Recommended to reject as scope bloat: per-command approval prompts, same-turn config-write refusal, diff return, turn-end batching, dirty-set coalescing, retry-on-miss, stdin piping, watcher handling, concurrency bounds, exclude lists, ordered-subscriber event system. [S8-M11, S8-M12]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-8, IC-1 to IC-9. Feeds M1-M6, M8-M9, N1-N2, N9.
- comprehension_report.md: CO-1 to CO-11. Feeds M2, M4-M10, N3.
- critic_report.md: DC-1 to DC-15, ST-1 to ST-10, SY-1 to SY-13. Feeds B1-B2, M1-M10, N1-N8.
- intermediate_zoom_out_report.md: S8-B1, S8-B2, S8-M1 to S8-M13, S8-N1 to S8-N7. Confirms all Stage 5 findings open.
