# Stage 5: Initial zoom out report (ZP-18)

Target: docs/designs/zcode-parity/ZP-18-mobile-device-control.md (full file, 184 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed. No Blocker raised.

## Zoom-out (ZO)

Major
- ZO-1. Lines 5-7, 17-23, 66-69, 116-118. Screenshot handoff to "the existing image attachment path" is unnamed and declared settled with no evidence. Bash returns text. The file Read tool has an image payload case (Tools/File/FileReadWriteTools.swift:64, :99); AppToolMediaPolicy and AppToolMediaHistoryAdapter exist under Tools/Browser. Name the real path (likely the skill telling the agent to Read the PNG) or list the change needed.
- ZO-2. Lines 156-165. Target path `Sources/TurboSparkApp/Resources/` does not exist; bundled resources live under `Sources/TurboSparkAppResources/Resources/`. SkillManager.swift has no bundled/builtin discovery, so item 5 is a new discovery root, not a tweak.
- ZO-3. Lines 37-39, 101-103, 156-165. Delivery mechanism left open ("ZP-05 or bundled-skill root, matching ZP-07") yet paths commit to `bundled-plugins/` with plugin.json, while ZP-07 uses `bundled-skills/`. INDEX sequencing puts ZP-18 before ZP-05. Adopt one shared root or state the dependency.
- ZO-4. Lines 49-50, 78-84. No behavior for a text-only model; computer-use tools are vision-gated (ZCODE_COMPARISON 9.6). State fallback (logs and accessibility evidence) or offer only to vision models.
- ZO-5. Lines 54-59. iOS interaction non-goal is muddled: "accessibility-driven shell tools" are unnamed. State plainly: launch, openurl, screenshot only; or point at computer-use driving Simulator.app.

Nit
- ZO-6. Lines 40-44. Goals read as a command checklist; ZCode's SwiftUI scaffolding feature from 9.5 is dropped silently. Add to Non-Goals.
- ZO-7. Lines 66-72. "No open question, settled" entry belongs in Overview or Architecture, and is unevidenced.
- ZO-8. Line 106. `agents/ios-dev.md` is not in Goals or the file list; existing agents are built in (AgentManager+BuiltIns.swift). Remove or list as non-goal.
- ZO-9. Lines 143-146. "Anti-circumvention rule applies verbatim" is guidance text in AppModel+AgentMode.swift:55, not a classifier rule; "medium risk" has no ladder position (INDEX lines 121-122).
- ZO-10. Lines 147-148. "Tail N lines" is a skill instruction, not an enforced bound; cite the Bash output cap.
- ZO-11. Lines 171-180. Text-clause mutation test is weak evidence; manual hardware run is the real gate. CUJ 2 and 3 have no test beyond text checks.
- ZO-12. Line 62. Windows-hosted emulation is redundant for a macOS-only app; Apple silicon emulator image availability is the real question.
- ZO-13. Lines 125-132. "Runtime constraint" is not linked; "Rejected by author" is not a rationale.

## Inventive catalyst (IC)

Major
- IC-1. Lines 56-59, 70-72, 123-139. False compromise: iOS has no tap/type yet CUJ 1 verifies a login screen. Make idb (or XCUITest via xcodebuild test) an optional, preflight-detected tool, with deep links and launch arguments as fallback; reword CUJ 1 with a navigation precondition.
- IC-2. Lines 37-39, 103, 156-165, 106. Same delivery inconsistency as ZO-3, plus no discovery path for the agent file. Pick the root shared by ZP-07, ZP-09, ZP-20; keep plugin.json only with ZP-05.
- IC-3. Lines 51, 143-146. Dozens of Bash calls per loop invite blanket allow rules; ladder classification of xcrun, adb, emulator, gradle unverified (adb shell, simctl erase, gradle scripts are dangerous). Ship a scoped prefix allowlist for read-only verbs, keep destructive verbs prompted, add a classifier test.
- IC-4. Lines 49-50, 68-69, 85-91, 147. Full-resolution PNG per step is large; non-vision model fallback missing. Downscale at capture (`sips`) or use JPEG, cap screenshots per turn, add a model-capability preflight and a text-only fallback (`uiautomator dump`, simctl state, logs). Verify the attachment seam first.

Nit
- IC-5. Lines 40-48, 85-91. Booted devices persist; no cleanup, boot timeout, or hung-build handling. Add a cleanup step for devices the skill booted itself, with timeouts.
- IC-6. Lines 93-98. CUJ 3 covers only Android preflight; add iOS (no Xcode, no runtime, no signing).
- IC-7. Lines 174-180, 166-172. Verification is string-contains on prose; record the hardware run as an evidence artifact and add a tool-guarded shell smoke script.
