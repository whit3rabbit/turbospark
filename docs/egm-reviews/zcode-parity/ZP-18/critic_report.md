# Stage 7: Critic report (ZP-18)

Target: docs/designs/zcode-parity/ZP-18-mobile-device-control.md (full file). One round only, no remediation, so convergence not assessed. Reviewers: egm-design (DC) and egm-structural (ST) against feature-spec-standard, egm-style (SY) with no standard. All completed.

## Design critic (DC)

Blocker
- DC-1. Lines 123-139. Both rejections are "Rejected by author" with no link or trade-off. Alternative A ignores structured output, errors, image return; B is a scope choice, not an alternative. Missing: native Swift in-process tool, `xcrun mcpbridge` or user-installed mobile MCP, idb now, user-installed marketplace plugin. Rewrite with concrete trade-offs.
- DC-2. Whole document. No Risks and Limits section. Unaddressed: shell injection via paths, bundle IDs, deep links; screenshots and logs as prompt-injection channel; destructive ops (`simctl erase`, `adb uninstall`, `adb shell`); simulator RAM against local model inference; orphaned simulators, emulators, adb at app quit (AppShutdownCoordinator); build timeouts vs Bash timeouts, license prompts; multiple devices or Xcode versions.
- DC-3. Lines 49-50, 66-69, 115-117. Screenshot handoff asserted. Only tool-result images (AppToolTypes.swift 199-225, 292-294) and user attachments exist; no path turns a Bash-written PNG into an image. If none, CUJ 1 needs new engine work, contradicting lines 52 and 120. Vision-less model behavior and image size unbounded.

Major
- DC-4. Lines 35-52, 166-180. Skills-only deliverable with text-clause tests; no measurable acceptance criteria for the hardware run.
- DC-5. Lines 54-59. Non-Goal self-contradictory; "accessibility-driven shell tools" unnamed; CUJ 1 login verification not achievable on iOS; asymmetry belongs in Problem and Goals.
- DC-6. Lines 147-148. N unspecified; no screenshot resolution, count cap, or asset-store accumulation limit.
- DC-7. Lines 37-39, 103, 156-165. Delivery undecided; paths `Resources/bundled-plugins/` match neither ZP-07/ZP-20 (`bundled-skills/`) nor an existing directory (`TurboSparkAppResources` is the real target); SkillManager has no bundled discovery; INDEX orders ZP-18 (step 3) before ZP-05 (step 4).
- DC-8. Lines 143-146, 51. "Medium risk" has no ladder position (INDEX 121-122); unclear whether each call prompts, making a 10-step loop unusable; anti-circumvention rule uncited.
- DC-9. Lines 25-33, 126. ZCode "wrap the same shell commands" claim uncited; 9.5 only grepped "simulator" on the Swift side.
- DC-10. Line 62. Windows non-goal is noise; the real constraint is Apple Silicon (x86 AVDs).

Nit
- DC-11. Lines 66-72. Settled item in Open Questions; idb duplicates the Non-Goal.
- DC-12. Lines 40-44. "launch" twice; "xcodeproj/workspaces" should read xcworkspace.
- DC-13. Lines 5-7, 20-21. No relative links to ZP-05, ZP-07; Swift-side provenance not carried (INDEX 128).
- DC-14. Line 105. `agents/ios-dev.md` not in Goals, files, tests.
- DC-15. Lines 149-150. Localization vague; ten parity gates (INDEX 115-116) not named.
- DC-16. Lines 174-180. No `make design-check`, verification doc, or CI-safe fake `xcrun` shim; hardware run is not release evidence.
- DC-17. INDEX size "medium" inconsistent with scope.

## Structural critic (ST)

Major
- ST-1. Lines 35-152. Missing Risks & Mitigations section (mandatory); failure modes, context and scalability limits, mitigations absent.
- ST-2. Lines 123-139. Rejection rationales are "Rejected by author"; B is a weak straw man; other designs absent. No maintainer link, so only depth is the gap.
- ST-3. Lines 64-72. Settled item under Open Questions; only idb is open.
- ST-4. Lines 54-59, 40-46. iOS non-goal wording unclear; asymmetry should be stated once; "launch" repeated.
- ST-5. Lines 100-118 vs 154-172. `agents/ios-dev.md` in diagram but no Create entry; delivery mechanism undecided so file plan cannot be validated; "catalog" strings have no file in Target Files.

Nit
- ST-6. Lines 160-165, 171. Items 2-4 lack Type/Scope and Proposed Changes; "Mutation checks" bullet is a different field kind.
- ST-7. Line 100. "Technical Plan (Architecture)" vs standard "Proposed Architecture".
- ST-8. Ordering. Open Questions precedes CUJs and architecture.
- ST-9. Lines 9-11, 184. Metadata format; consistent with INDEX, no action.
- ST-10. Line 57. "ZCode marks it optional" uncited.

## Style critic (SY)

Major
- SY-1. Lines 37-39, 156-165. Delivery decision hidden in a Goal; "day one" vague. Pick one or record as Open Question.
- SY-2. Lines 49, 66-69. Handoff uncited; `adb exec-out screencap -p` writes to stdout, redirect missing, so "lands as files" is inaccurate as written.
- SY-3. Lines 64-69. Not a question; "No platform gap identified" uncalibrated.
- SY-4. Lines 56-59. Three clauses joined by semicolons; name remaining commands; state asymmetry in one sentence.
- SY-5. Lines 105-106 vs 156-166. `agents/ios-dev.md` "optional" gives no decision rule.
- SY-6. Lines 143-148. "Medium risk", "classifier's anti-circumvention rule", "shell route-arounds" undefined; "PNG files" is a format not a requirement; N unspecified; "permission ladder" vs `AppToolPermissionEngine` naming.

Nit
- SY-7. Lines 27-30. "shell advice but no loop" subjective and unevidenced; "preflight" undefined.
- SY-8. Line 32. "no-bundled-runtime constraint" undefined; use INDEX wording.
- SY-9. Lines 43-44. Duplicate "launch".
- SY-10. Lines 60-62. Windows non-goal redundant.
- SY-11. Lines 93, 98. "preflight honesty", "failing opaquely" loose.
- SY-12. Lines 120-121. Repeats lines 51-52; drop.
- SY-13. Lines 131-139. "Rejected by author" restates the label; "more common loop" unsupported; pros and cons do not address each other.
- SY-14. Lines 128-130. "duplicates what skills already express" vague.
- SY-15. Lines 154-171. Items 2-4 path only; "Mutation checks:" is a fragment.
- SY-16. Lines 176-180. Scheme unnamed; "described correctly" has no pass criterion; CUJ 3 has no verification step.
