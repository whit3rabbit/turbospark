# Stage 8: Intermediate Zoom Out report (OC-11)

Target: docs/designs/opencode2-parity/OC-11-formatter-pass.md (unchanged since Stage 5). Reviewer egm-zoom-out completed. It read the Stage 5, 6, 7 reports and did not re-run code searches; the reuse targets (LoginShellPath, ProcessExecutor, PathContainment, ShellOutputFormatting) are not re-verified.

Summary: a small-medium spec. All Stage 5 findings (ZO-1 to ZO-8, IC-1 to IC-9) remain open because nothing was edited. Stages 6 and 7 confirmed the main clusters: OC-10 seam dependency, permission ladder and safety claim, PATH resolution, what the model sees after a rewrite, registry scope vs the Open Question, undefined pass log. Stage 7 adds many asks that would bloat the spec; keep the ambiguity-removal fixes and reject the rest.

## Blockers

S8-B1 (lines 36-76, 102-115; confirms DC-1/ST-1). Missing Alternatives Considered. The real unweighed choice is smallest registry vs built-in plus user rows. Fix: three or four bullets only (leave to the model or explicit tool; shared OC-10 seam vs bespoke hook; swift-format only vs user rows only; turn-end vs per-edit), one-line reason each. Do not add the speculative options ST-1 lists (LSP formatting, bundled formatter, save-watch).

S8-B2 (lines 117-128; confirms DC-2/ST-2). Missing Risks and Mitigations. Fix: a compact table of about five rows (stale model view, config-file code execution, write race, hang, root escape), each pointing at a fix already being made.

## Major: keep (ambiguity removal)

S8-M1 (18-22, 134-141; ZO-1, ZO-2, IC-6, DC-8, ST-5, SY-5, ST-10). Add a Dependencies line: the pass registers on OC-10's post-edit seam, runs before diagnostics, reuses LoginShellPath and ProcessExecutor; fix INDEX so OC-11 appears once, after OC-10. Do not add an ordered-subscriber event system.
S8-M2 (121-124; ZO-3, IC-5, DC-3, ST-6, SY-4, CO-9). State: the pass is app-initiated outside the ladder; user rows live in the app archive, never the repo; the toggle is the trust grant and formatter config is trusted input. Reject per-command prompts, same-turn config-write refusal and config-change detection; record as known risk.
S8-M3 (38-41, 70-73; ZO-4, IC-8, ST-4, SY-1, DC-9). Resolve now: v1 ships swift-format only with a fixed command, plus user rows as examples; drop gofmt and clang-format; one match key (extension). Reject precedence and shebang rules.
S8-M4 (48-56, 99-100; ZO-5, IC-1, CO-6, DC-6). State that the tool result carries post-format content, the edit tool's recorded hash is refreshed, OC-10 runs afterwards, and undo captures the pre-format write. Reject diff return and turn-end batching.
S8-M5 (42-44, 49-50; IC-3, DC-4, CO-2, SY-2). A PATH value is captured once through LoginShellPath, cached, used to resolve an absolute executable; the run is an argument vector with no shell; in-project relative executables are not allowed in v1. Reject retry-on-miss and invalidation machinery.
S8-M6 (48-53, 99-100; IC-4, DC-5). Format a temp copy, replace on exit 0 only, skip with a failure note if the file hash changed during the run. Reject stdin piping and watcher handling.
S8-M7 (49, 112, 119-120; DC-7, SY-3). Give a timeout (for example 10 s), kill on expiry, cap error-tail output, cancel on turn stop. Reject concurrency bounds and signal-policy detail.
S8-M8 (44, 88-93; CO-7, DC-11, SY-8, IC-8). Drop the pass log; a missing formatter is a silent skip.
S8-M9 (55-56; CO-5, IC-2, DC-10). Define "written this turn" as paths returned by successful edit-tool calls; require PathContainment with symlink resolution. Reject the dirty-set and exclude list; formatting a file six times is cheap.
S8-M10 (74-76, 134-141; CO-4, SY-11). Name the triggering tools in one list; mark the Open Question resolved as write and edit tools only.

## Major: reject (scope bloat)

S8-M11 (DC-3 clauses, IC-5): per-command approval, same-turn config detection, one-time per-row confirmation.
S8-M12 (IC-1 to IC-4, DC-5, DC-7, DC-10, DC-11): diff return, batching, dirty-set coalescing, retry-on-miss, stdin piping, watcher handling, concurrency bounds, exclude lists.
S8-M13 (DC-11, CO-11, SY-13): per-safety-property test demands. Add one FormatterRunnerTests line (timeout, failure keeps original, root bound).

## Nits

S8-N1 (26-31; ZO-6, SY-7). Drop the ShellOutputFormatting citation; note TerminalCommandClassifier already lists the formatters.
S8-N2 (51-54; ZO-8, SY-9, DC-6). Note only on failure, capped and ANSI-stripped via ShellOutputFormatting.stripANSI; success silent.
S8-N3 (28, 31, 134-146; ST-3). Fix relative links with a ../../../ prefix.
S8-N4 (125-128, 146-151, 155-157; ZO-7, CO-10, DC-12). Trim localization, accessibility and mutation text to one-line references to INDEX.md.
S8-N5 (19-21, 65, 70-73; SY-6, SY-10, SY-12, ST-9). Remove "guaranteed", "Lean:" to "Recommendation:", rephrase "format-on-agent-prompt".
S8-N6 (144-145; DC-15). One line: a missing persistence key decodes to off.
S8-N7 (104-115, 134-136, 3-8). Optional polish; reject circuit breaker (IC-7), diagram redraw, commit pinning, evidence sentences; fix registry vs persistence ownership only.

Bottom line: after the keep list the spec stays small-medium with one built-in row, one dependency, one safety paragraph and one risk table.
