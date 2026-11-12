# Consolidated review report: ZP-18 mobile device control

Target: docs/designs/zcode-parity/ZP-18-mobile-device-control.md (full file, 184 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed (egm-design needed three spawn retries after "Concurrent subagent limit reached", then ran).
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 8, Nit 13.

## Blocker

B1. Lines 49-50, 66-69, 116-118 (also 52, 120). Screenshot handoff is asserted and called "settled" with no evidence; Bash returns text and only the Read tool (FileReadWriteTools.swift:64, :99) and tool-result images (AppToolTypes.swift:199-225) emit images. `adb exec-out screencap -p` writes to stdout, so "lands as files" lacks a redirect. Fix: name the real path (likely the skill tells the agent to Read the PNG, and where it is written); if none exists, list the change and drop "no new engine". [ZO-1, CO-2, DC-3, SY-2, S8-B1]
B2. Lines 35-152 (missing section). No Risks and Limits section, required by the standard. Fix: add a short one: destructive verbs (`simctl erase`, `adb uninstall`, `adb shell`) stay prompted, screenshots and logs are untrusted content, skill cleans up devices it booted, builds and boots have timeouts, skill quotes its arguments. [DC-2, ST-1, S8-M5]

## Major

M1. Lines 37-39, 103, 156-165. Delivery mechanism undecided yet paths hard-coded: `Resources/bundled-plugins/` does not exist (real target `TurboSparkAppResources`), ZP-07 and ZP-20 use `bundled-skills/`, SkillManager has no bundled discovery, INDEX orders ZP-18 (step 3) before ZP-05 (step 4). Fix: pick one shared root, say which draft owns it, record SkillManager change as a new discovery root. [ZO-2, ZO-3, CO-1, DC-7, ST-5, IC-2, SY-1, S8-M1]
M2. Lines 54-59, 76-84. iOS interaction limit is muddled ("accessibility-driven shell tools" unnamed) and contradicts CUJ 1 login verification. Fix: one sentence that iOS v1 supports launch, openurl, screenshot; give CUJ 1 a navigation precondition; state the Android asymmetry once. [ZO-5, CO-5, DC-5, ST-4, SY-4, IC-1, S8-M2]
M3. Lines 49-50, 82-83. No behavior for a model without vision in a local-first app. Fix: offer screenshots to vision models only, fall back to logs and simctl state. [ZO-4, CO-4, IC-4, S8-M3]
M4. Lines 143-146, 51. "Medium risk", "classifier's anti-circumvention rule", "permission ladder" undefined; no ladder position for xcrun, adb, emulator, gradle (INDEX 121-122); the rule is guidance at AppModel+AgentMode.swift:55, not a classifier rule. Fix: state current ladder behavior, cite the guidance, use `AppToolPermissionEngine` naming. [ZO-9, CO-3, DC-8, IC-3, SY-6, S8-M4]
M5. Lines 123-139. Alternatives rejected as "Rejected by author" with no trade-off; B is a scope choice. Fix: one concrete reason each, say B is a scope choice; do not add four more alternatives. [DC-1, ST-2, SY-13, SY-14, ZO-13, S8-M6]
M6. Lines 166-180. Verification proves prose contains phrases, not behavior; no pass criteria for the hardware run, scheme unnamed, CUJ 3 untested. Fix: write pass criteria and record the hardware run as evidence. [DC-4, ZO-11, CO-6, IC-7, SY-16, S8]
M7. Lines 147-148. "Tail N lines" unspecified and only a skill instruction. Fix: cite the Bash output cap. [DC-6, ZO-10, SY-6, CO-10, S8-M5]
M8. Lines 25-33, 126. ZCode "wrap the same shell commands" and "shell advice but no loop" are uncited; 9.5 only grepped "simulator". Fix: cite a file or soften. [DC-9, SY-7]

## Nit

N1. Lines 64-72. Settled screenshot item sits in Open Questions; idb duplicates the Non-Goal; delivery mechanism is the real open question. Fix: move and replace. [ZO-7, CO-11, ST-3, SY-3, DC-11]
N2. Lines 105-106. `agents/ios-dev.md` is in the diagram, not in Goals, files or tests. Fix: remove or list as non-goal. [ZO-8, CO-8, DC-14, ST-5, SY-5]
N3. Lines 40-44. "launch" listed twice; "xcodeproj/workspaces" should read xcworkspace; stop/uninstall scope unclear. [CO-7, DC-12, SY-9, ST-4]
N4. Line 62. Windows non-goal is noise for a macOS-only app; Apple Silicon emulator image availability is the real question. Fix: replace or drop. [ZO-12, DC-10, SY-10]
N5. Lines 40-44, 54-62. ZCode's SwiftUI scaffolding from 9.5 dropped silently. Fix: add to Non-Goals. [ZO-6]
N6. Lines 3-7, 37-38. No relative links to ZP-05 and ZP-07; Swift-side provenance not carried (INDEX 128). [DC-13]
N7. Lines 120-121. Repeats lines 51-52. Fix: drop. [SY-12]
N8. Lines 85-98. CUJ 3 lacks iOS case (no Xcode, no runtime, no signing); CUJ 2 lacks no-AVD and boot-failure case. [IC-6, CO-12]
N9. Lines 100, 64, 154-172. Heading "Technical Plan (Architecture)" vs "Proposed Architecture"; Open Questions precedes CUJs; Target Files items 2-4 lack Type/Scope; INDEX size "medium". Low value. [ST-6, ST-7, ST-8, ST-9, DC-17]
N10. Lines 31-33, 93, 98, 126-130. Undefined terms: "no-bundled-runtime constraint", MCP, "preflight honesty", "failing opaquely"; "ships them" reads as present fact. Fix: use INDEX wording. [CO-9, CO-10, SY-8, SY-11, ST-10]
N11. Lines 149-150. "Plugin listing strings in the catalog" vague; name the file or say no UI strings are added; ten parity gates. [DC-15, CO-10]
N12. Lines 166-172. "Mutation checks" fragment; reword as action and state expected failure. [SY-15, CO-6]
N13. Lines 134-139. Alternative B claim "iOS is more common" unsupported. [SY-13]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-13, IC-1 to IC-7. Feeds B1, M1-M7, N1-N8.
- comprehension_report.md: CO-1 to CO-12. Feeds B1, M1-M4, M6-M7, N1-N3, N8, N10-N12.
- critic_report.md: DC-1 to DC-17, ST-1 to ST-10, SY-1 to SY-16. Feeds all Blockers and Majors, N1-N13.
- intermediate_zoom_out_report.md: S8-B1, S8-M1 to S8-M6, S8-N. Confirms all earlier findings open.
- Severity calibration: DC-1 and DC-2 (reviewer Blocker) kept as Major M5 and Blocker B2 respectively. S8 downgraded DC-1 to Major (adopted) and DC-2 to Major; B2 stays Blocker because the standard mandates the section and ZP-01 treated the same gap as Blocker. DC-3 Blocker adopted.
- Recommended rejections (scope creep per S8): IC-1 idb or XCUITest detection, IC-3 allowlist and classifier test, IC-4 downscale and caps and capability preflight, DC-6 asset-store limit, DC-2 AppShutdownCoordinator code, DC-16 and IC-7 shim and smoke script, four extra alternatives from DC-1.
