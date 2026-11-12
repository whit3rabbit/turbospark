# Stage 7: Critic report (ZP-17 error boundaries)

Round 1 only (no remediation, so no re-audit). Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY), all completed. Standard: feature-spec-standard for DC and ST. Code check by DC: AppShutdownCoordinator exists at State/AppModel+Persistence.swift:607, hooked at App/TurboSparkApp.swift:72 (applicationWillTerminate). RolloutStore.swift, OfficeRenderModel.swift, PluginPageWebView.swift do not exist.

## Design critic (DC)

DC-1 Major. Whole file. No Risks & Mitigations section. Unaddressed: false positives (SIGKILL, Force Quit, logout, power loss, debugger stop), second instance seeing a live marker (pid liveness never stated), crash loop on restore, marker write failure and stale markers across upgrades, ZP-03/06/08 coupling.
DC-2 Major. Lines 116-131. Rejections are "by author" with no link or concrete trade-off; Alternative A is a strawman. Missing alternatives: macOS window and state restoration, MetricKit or DiagnosticReports for detection, signal handler (rejected on async-signal-safety grounds), narrow isolation (XPC for untrusted office parsing; WKWebView already out of process).
DC-3 Major. Lines 39-43, 78-79, 100-104, 111-114, 158-164. Target files do not exist and ZP-17 precedes ZP-03/08 in INDEX. Typed Results do not stop traps, overflow, deep-nesting stack exhaustion or OOM, so "the process does not crash" (CUJ 2) is unsupportable. No evidence or inventory for "known high-risk" surfaces.
DC-4 Nit. Lines 47-48, 71. "Exactly as on quit" versus "where the flush allowed it"; flush cadence not cited.
DC-5 Nit. Lines 56-61, 135, 149. Open Question is decided in its own text; marker location unspecified; the restore snapshot holds chat ID while detection is pre-unlock; restore must run post-unlock. Vault unlock on relaunch is normal behavior, not a defect fixed here.
DC-6 Nit. Lines 44-46, 106-109, 168-177. Lint has no tool, scope, CI hook or test.
DC-7 Nit. Line 5. Pin the ZCode link to the audited commit a60ba6a7.
DC-8 Nit. Lines 29-31, 133-140, 178-184. No localization or accessibility requirement, no design-check or localization gates in verification, no success metric; SIGKILL manual step is the only true abnormal-exit test.

## Structural (ST)

ST-1 Blocker. Lines 90-141. No Risks & Mitigations / Risks & Limits section; the standard requires one (failure modes, mitigations).
ST-2 Major. Lines 124, 131. "Rejected by author" with no link; Alternative B restates its cons.
ST-3 Major. Lines 116-131. Only two alternatives, one of them do-nothing; marker-in-vault, OS crash reports, periodic snapshot and lint-only options are absent.
ST-4 Major. Lines 56-61. Decision recorded as an open question; section precedes the architecture it references.
ST-5 Major. Lines 32-48, 63-88, 90-114. Goals name three surfaces but CUJs cover two (no plugin page CUJ); no CUJ for lint or banner dismissal; no measurable success criteria.
ST-6 Major. Lines 92-109. Architecture is a text diagram only; marker contents, snapshot contents and restore sequence across vault unlock are not laid out.
ST-7 Nit. Lines 7-8, 158-164. Target item 4 bundles three files, omits Type / Scope, and uses bare ZP IDs without links.
ST-8 Nit. Lines 174-176. Mutation checks mis-nested in a Proposed Changes bullet; belongs in Verification Plan.
ST-9 Nit. Lines 106-109, 178-184. Lint pass has no target file, baseline file or verification entry.
ST-10 Nit. Lines 3-5. External ZCode link on main is a moving target; pin the commit.

## Style (SY)

SY-1 Major. Lines 16-20, 53. "An honest translation" and "instead of pretending" are meta-commentary. Fix: state the facts.
SY-2 Major. Lines 24-25, 29-30. "Uncaught throw in a view body" is wrong for Swift (bodies do not throw); real classes are force unwraps, preconditions, array bounds. Vault unlock on relaunch is cited as a symptom without evidence.
SY-3 Major. Lines 28-30. Claims about existing code have no file citation (StreamRecoveryEventNoticeView).
SY-4 Major. Lines 47-48 vs 70-72. "Exactly" contradicts "where the flush allowed it". State best-effort.
SY-5 Major. Lines 39-43. "Known" high-risk unsupported; "typed errors rather than exceptions" is vague; say Result values.
SY-6 Major. Lines 56-61. Not a real open question; move the decision to the plan, keep only the privacy confirmation with an owner.
SY-7 Nit. Lines 111, 121-122, 124, 131. Intensifiers ("deliberately", "far") and bare "Rejected by author".
SY-8 Nit. Lines 120, 128. "True isolation", "zero cost" are hyperbole.
SY-9 Nit. Lines 135-138. "Reads only state already persisted" conflicts with the new snapshot; state "one write at launch, one delete at quit".
SY-10 Nit. Lines 139-141. "So future agents do not re-derive" is rhetorical.
SY-11 Nit. Lines 158-164. "Pins their error contract" is jargon; no ZP-03/08/06 mapping.
SY-12 Nit. Lines 170-176. "Mutation checks" unexplained; "assert the contract test fails" ambiguous.
SY-13 Nit. Lines 3-5. Verify the AppErrorBoundary name matches ErrorBoundary.tsx.
SY-14 Nit. Lines 44-46. Lint overclaims without a named mechanism.
SY-15 Nit. Line 157. Localization noted; no further action.
