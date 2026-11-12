# Stage 5: Initial Zoom Out report (OC-11)

Target: docs/designs/opencode2-parity/OC-11-formatter-pass.md (full file, 164 lines)
Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom Out (ZO)

Context check passes. The design is small and mostly sound. Findings concern sequencing, reuse, and altitude.

ZO-1 (Major) Lines 134-141. The spec does not reuse the existing seam or helpers. INDEX.md says OC-10 lands the post-edit seam first and OC-11 reuses it, but OC-11 proposes its own hook in AppModel+Tools.swift and never names OC-10. A `LoginShellPath.swift` (Tools/MCP) and `ProcessExecutor.swift` (Tools/Core) exist but are not named ("existing hardened process path", "login-shell PATH"). Risk: two post-edit hooks and a second PATH resolver. Fix: name OC-10's seam, LoginShellPath and ProcessExecutor as reuse targets and state the dependency in the Overview.

ZO-2 (Major) Lines 134-139. INDEX.md lists OC-11 in sequencing step 3 ("after OC-10's seam") and again in step 7 with OC-10, so step 3 puts OC-11 before its dependency. Fix: add "Depends on OC-10 post-edit seam" and fix the INDEX so OC-11 appears once, after OC-10.

ZO-3 (Major) Lines 18-19, 45-47, 119-124. Thin permission and safety story for a feature that runs binaries. User-configured rows with explicit commands could live in a project file, so a cloned repo could plant a command that auto-runs after every edit. INDEX.md requires agent-facing drafts to state their ladder position in AppToolPermissionEngine; this draft does not. Fix: state that user-defined formatter commands are stored at user level, not in the repo, state the ladder position, and say the toggle is the trust grant.

ZO-4 (Major) Lines 38-41, 68-72. v1 scope is larger than the evidence supports and an Open Question is left unresolved. Three built-in rows plus a user-configurable registry (templates, extension matching, per-project records) is large for a small-medium draft, and user rows are a second configuration surface to design and localize. Fix: ship the smallest registry (swift-format only, or user rows only) and resolve the open question now.

ZO-5 (Major) Lines 49-53, 55-56. Data-flow semantics are underspecified. The file is rewritten after the tool reports success. The spec does not say how this interacts with the model's view of the file (the next edit may anchor on stale content), with checkpoint or undo stores, or with OC-10 diagnostics running on pre-format content. "Files the agent wrote in that turn" has no mechanism. Fix: add a paragraph stating the tool result carries post-format content, undo and checkpoints capture the pre-format write, and OC-10 runs after the formatter.

ZO-6 (Nit) Lines 26-31. ShellOutputFormatting is output shaping and unrelated. TerminalCommandClassifier's buildCommands already lists prettier, rustfmt, ruff and black, so a model-run formatter already auto-approves under `.auto`; that relevant fact is missing. Fix: drop the ShellOutputFormatting citation and note the buildCommands fact.

ZO-7 (Nit) Lines 125-128, 146-151. Localization and accessibility NFRs repeat program constraints from INDEX.md, and the mutation-check detail on line 150 belongs in a test plan. Fix: trim to a one-line reference.

ZO-8 (Nit) Lines 51-54. A "formatted by X" note on every successful edit adds transcript noise and localization cost; silent skip is already the model for CUJ 2. Fix: note only on failure, log success in the pass log.

## Inventive Catalyst (IC)

No Blocker. The scope and root-problem framing (lines 16-34) are sound. The contradictions concern the formatter changing bytes the model believes it wrote. (The reviewer read only lines 1-136 of OPENCODE2_COMPARISON.md, which omits the Section 8 finding.)

IC-1 (Major) Lines 18-22, 48-53, 99-100. Guaranteed style versus the model's picture of the file. Showing post-format content in the transcript (line 53) is a display fix; the model's next edit may use old_string text the formatter changed or hit a staleness guard. The spec never says whether formatted content or a diff reaches the model. Fix: return the formatter diff or post-format region in the tool result and refresh the recorded file hash; alternatively batch formatting at turn end with one "files reformatted" summary.

IC-2 (Major) Lines 55-56 vs 48-49. "Files the agent wrote in that turn" implies per-turn tracking but the pass runs per edit-tool success, so a file edited six times is formatted six times. Fix: keep a turn-scoped dirty-file set and coalesce to one pass per file, with a flush-or-drop rule on cancel.

IC-3 (Major) Lines 42-44 vs 49-50. Resolving on the "login-shell PATH" conflicts with "no shell" unless a cached snapshot is used, and project-local tools (node_modules/.bin/prettier, a venv's ruff) are invisible to a login-shell PATH so user rows will often skip silently. Fix: resolve lazily once per project open via the existing shell-environment path, cache absolute paths, allow absolute or project-relative commands in a row, and add retry-on-miss.

IC-4 (Major) Lines 48-52, 99. In-place rewrite and failure handling. A formatter that crashes or times out mid-write can truncate, and the user may have saved the same file externally in between. Fix: pipe through stdin/stdout where supported, or format a temp copy, and swap atomically only if the file hash still equals what the edit tool wrote; otherwise skip and report.

IC-5 (Major) Lines 121-124, 20-22, 42. The "toggle plus write permission cover the act" claim is thin: formatters read repo-controlled config (prettier plugins can execute project code), and the ladder position is missing. Fix: state the pass is app-initiated outside the ladder, store user rows in the app archive never the repo, consider one-time per-row confirmation, and document formatter config as trusted input.

IC-6 (Major) Lines 140-141 (and INDEX.md). The OC-10 post-edit seam is not mentioned; two independent hooks recreate coupling and ordering matters (diagnostics should run on formatted files). Fix: define a post-edit event with ordered subscribers (formatter, then diagnostics) and depend on it; fix the INDEX sequencing.

IC-7 (Nit) Lines 51-52, 99-100. A broken formatter pays its full time bound and appends an error tail on every write. Fix: add a per-formatter circuit breaker and mark syntax-error failures distinctly.

IC-8 (Nit) Lines 38-41 vs 70-72. Goals commit the three v1 rows while the Open Question asks whether to include them; the time bound has no value; the "pass log" (line 44) is not created by any planned file. Fix: decide, state the bound, define the log.

IC-9 (Nit) Lines 130-152. Undo and checkpoint interaction is unspecified (existence of such a mechanism not verified by the reviewer). Fix: state which content the record keeps.
