# Stage 5: Initial zoom out report (ZP-20)

Target: docs/designs/zcode-parity/ZP-20-bundled-content-plugins.md (full file, 170 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed. Persona files read: agents/zoom_out.md, agents/inventive_catalyst.md.

## Zoom-out findings (ZO)

ZO-1 Major. Lines 45-46, 103, 143-150. The design depends on a bundled-skill root that does not exist in code (no `bundled-skills` or `BuiltinSkills` reference under swift/). ZP-07 is itself a Draft. ZP-20 does not say it is blocked on ZP-07, and the SkillManager change is a one-line "discovery of the bundled root entries". "Unremovable, disable-able" has no named seam in SkillManager or its enable/disable store.
ZO-2 Major. Lines 143-154. Path and ownership conflicts with ZP-07: ZP-07 says `Plugins/BuiltinSkills/plugin-creator/` in its plan and `Resources/bundled-skills/` in its file list; test file names differ (`BundledGuideSkillsTests.swift` vs `BundledSkillsTests.swift`); INDEX lines 87-88 name ZP-05 seeding as the delivery mechanism. Resources must also be declared in Package.swift or the Xcode project, which neither draft states.
ZO-3 Major. Lines 130-132, 152-158. The drift test is a string-exists check. It cannot verify pane names or settings semantics. Reuse existing path constants (for example the known skill-root lists in SkillManager.swift) or derive from one manifest. Open question (a)/(b) is left unowned.
ZO-4 Major. Lines 41-43, 77-79, 133-135. The visualize contract (1 MB cap, fragment-free, "save as") is not checked against ArtifactWebView or the artifact panel; no size cap was found in Artifact*.swift. No tool or ladder position is named for writing the artifact (ArtifactRegistrar, ArtifactWorktreeTools exist). INDEX requires stating agent-facing tool positions.
ZO-5 Major. Lines 18-19, 36-37, and the 9.14 mapping. Visualize belongs to finding 9.9 (Partial), not 9.14; INDEX lists only 9.14. The title says "plugins" but ships skills only.
ZO-6 Nit. Line 36-37. Six skills for untested content; no listing-cost statement (SkillManager+Budget.swift exists); a single guide file is not considered as an alternative.
ZO-7 Nit. Lines 60-61. Open question is "lean (a), author decides"; resolve it or mark it an implementation blocker.
ZO-8 Nit. Lines 65-86. CUJ 1 assumes MCP logs exist at a known path readable by file tools; CUJ 3 names shortcuts and profiles, which are not in the guide's goals.
ZO-9 Nit. Lines 136-137, 160-166. Localization of listing strings vs the INDEX ten-gate, 21-language constraint; no localization step in verification.
ZO-10 Nit. Lines 157-158. Mutation-check bullet is test-level detail.
ZO-11 Nit. Lines 3-8. ZCode links not re-verified (INDEX says to re-check before acting).

## Inventive catalyst findings (IC)

IC-1 Major. Lines 56-61, 130-132, 152-158. Accepting hand-written drift is a false compromise. Make app path and pane constants the source of truth, render a manifest, resolve placeholders at discovery; or let diagnose skills read live state through existing tools. Liability: templating step and late failure on missing placeholder.
IC-2 Major. Lines 41-45, 73-79, 133-135. The visualize contract is enforced only as prose, and the test checks that the words are present. Put a validator in the artifact write path (size, external URL scan, fragment check) returning a tool error the model can fix. Liability: heuristic scan, new code not in the file list, not a security boundary.
IC-3 Major. Lines 88-109, 150-151. Shared discovery is left implicit. Define one bundled-content provider contract in ZP-07, ZP-20 declares content only; decide whether diagnose-* are flat peers or progressive-disclosure sub-files.
IC-4 Nit. Lines 56-61, 111-126. Alternative A is a false dichotomy; one canonical reference could feed both a docs page and a thin skill.
IC-5 Nit. Lines 45-46, 103. If a diagnose skill is disabled, the agent silently gives generic help. Keep the guide index always-on or have the agent say a skill is off.
IC-6 Observation. Lines 21-22. No per-finding 9.14 entry was found in ZP-90 as read; check the claim that the excluded thirds are recorded there.
