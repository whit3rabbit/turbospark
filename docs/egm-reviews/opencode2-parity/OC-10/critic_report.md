# Stage 7: Critic report (OC-10)

Target: docs/designs/opencode2-parity/OC-10-lsp-diagnostics-symbols.md (lines 1-204). Round 1 only; convergence not assessed. Reviewers egm-design (DC), egm-structural (ST), egm-style (SY), all completed. Standards: feature-spec-standard for DC and ST.

## Design critic (DC)

DC-1 Blocker. Whole doc. No Alternatives Considered. Unweighed: shell out to build or lint after edits; reuse the MCP stdio engine (`Tools/MCP/McpClientEngine.swift`); a Rust-side LSP crate; letting users wire an LSP MCP server; swiftc diagnostics for Swift only. Open Questions (76-90) are not a substitute.

DC-2 Blocker. Whole doc (139-150 only has scattered bullets). No Risks and Mitigations. Missing: server crash or hang with no restart or backoff policy and no timeout values; stale or out-of-order diagnostics with no version or settled signal; memory and CPU contention with local inference, no cap or idle eviction ("idle servers cost memory only" at 142 is unbacked); no token cap on summary or symbol results; repo-supplied config launching arbitrary binaries and servers running build scripts or proc macros; `file` arguments not confined to the project root.

DC-3 Major. 132-135. Reuse claim unverified. `ProcessExecutor` is a one-shot executor; long-lived stdio patterns are `McpClientEngine.swift`, `REPLWorkerProcess.swift`, `CodemodeWorkerSupervisor.swift`, `BackgroundShellManager.swift`. "The shutdown coordinator" is not a found name. Name the real ones.

DC-4 Major. 78-82, 99-101, 141-142. CUJ 1 "before the model plans its next step" needs a synchronous wait but the policy is open; late-result mechanism undefined (174); first edit with a cold, indexing server will miss a 2 to 5 s budget; no success metric. Pick a default and number.

DC-5 Major. INDEX line 83. The post-edit seam OC-11 reuses is never defined as an interface; formatter versus diagnostics ordering, concurrent multi-file edits, and edits by shell commands are unaddressed.

DC-6 Major. 87-90, 171-178. Ladder position is open though INDEX requires it be stated; `workspace_symbols` may return paths outside the project; relation of `ProjectPermissionsSectionView` toggle to the new settings view and where server config lives (project file versus app store) is unspecified.

DC-7 Major. 53-57. Tool count wrong ("Two" with three listed). `project_diagnostics` for never-opened files returns mostly empty because servers publish only for opened files; name is misleading.

DC-8 Major. 156-190, INDEX line 40. No phased delivery for a 12-file plan; document sync (full versus incremental, UTF-16 positions, encoding), multi-root, and unbuilt-project behavior for sourcekit-lsp (feature must not trigger a build) unspecified.

DC-9 Nit. 3-9, 36. AppFileSymbol link resolves wrong; use `../../../swift/...`.
DC-10 Nit. 25-27. "30+ servers" and "fifth" need source and date.
DC-11 Nit. 144-146. "tool-process hardening" undefined; state the server environment (PATH, HOME, secrets).
DC-12 Nit. 147-148. Only wrapper strings are localized; server text is not. Name the ten localization gates in Verification.
DC-13 Nit. 183-189. Add tests for crash recovery, stale-version drop, token cap.
DC-14 Nit. Open Questions have leans but no owner or decision gate.

## Structural (ST)

Heading hierarchy is clean, fence closed, finding references at 3-4 exist, "fifth" matches comparison line 499.

ST-1 Blocker. Missing Alternatives Considered (same as DC-1; adds shell-out, existing Swift LSP package, Rust engine, fire-and-forget as rejected options).
ST-2 Blocker. Missing Risks and Mitigations (same as DC-2; adds UTF-16 versus UTF-8 position mismatch).
ST-3 Major. 53 vs 54-57, 22-23, 165. Tool count contradiction.
ST-4 Major. 36. Broken relative link.
ST-5 Major. 117-137. No numbered data flow for the post-edit path (write, didOpen/didChange/didSave, wait, summary, late delivery); late delivery appears only at 82 and 175-176.
ST-6 Major. 87-90 vs 171-173. Plan files 6 and 8 depend on an unresolved open question; resolve or mark conditional.
ST-7 Nit. 199-200. `swift test` belongs under Tests, not Manual.
ST-8 Nit. 126-127 vs 165. Diagram names (`EditToolHook`, `DiagnosticsTool`, `SymbolsTools`) do not match plan files.
ST-9 Nit. 183-189. No tests for permission category, settings view, or sourcekit-lsp discovery.
ST-10 Nit. 62, 132-135. Undefined terms without links (login-shell PATH path, `ProcessExecutor` conventions, shutdown coordinator).
ST-11 Nit. No Related line; INDEX says OC-10 lands the post-edit seam before OC-11.

## Style (SY)

SY-1 Major. 24-26, 38-39. Attribution wrong: "clearest agent-quality gap" is the comparison report's verdict, not OpenCode's; the report says OpenCode's LSP runtime is mid-rebuild with a TODO in file-mutation.ts. Calibrate.
SY-2 Major. 53 vs 22-23, 54-57, 165. Tool count.
SY-3 Major. 149-150. "nothing else is user-visible" contradicts the transcript summary (99-101), permission toggle (177-178), and late-results note (80-82).
SY-4 Major. 78-82, 87-90. "lean" and "likely answer" with no number, owner, or measurement; "big" undefined. State a budget (for example 2 s) and what would change it.
SY-5 Major. 94-101. CUJ 1 not testable: "immediately" versus bounded wait; "diagnostics line count" ambiguous; "first errors" has no N; timing claim has no mechanism; expiry behavior unstated.
SY-6 Major. 141-145. Idle-cost claim unverified (indexing uses CPU); hardening unnamed.
SY-7 Major. 132-137. "bounded", "time out", "clean" have no values.
SY-8 Nit. 36. Link path.
SY-9 Nit. 26. "The TurboSpark shape is" is filler; "This design adds".
SY-10 Nit. 49, 61. "covering server" jargon; "the existing login-shell PATH path" vague.
SY-11 Nit. 67. "full position lifecycle" unclear.
SY-12 Nit. 82-85. "seed rows for common toolchain binaries" vague.
SY-13 Nit. 196-200. Move `swift test` to Tests and drop the reason clause.
SY-14 Nit. 189. Split the mutation-check sentence.
