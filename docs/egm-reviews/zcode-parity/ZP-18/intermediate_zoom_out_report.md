# Stage 8: Intermediate zoom out report (ZP-18)

Target: docs/designs/zcode-parity/ZP-18-mobile-device-control.md (full file). Reviewer: egm-zoom-out (completed). The document was not edited, so every Stage 5 to 7 finding is still open. The core design (two markdown skills, no new engine, no MCP server, reuse of Bash and the permission ladder) is sound and minimal. Several asked-for mitigations would turn a skills-only change into a Swift project and should be rejected.

## Open findings that are warranted (S8)

- S8-B1 (Blocker). Lines 49-50, 66-69, 116-118. Screenshot handoff asserted, not shown (ZO-1, CO-2, DC-3, SY-2). Bash returns text; the file Read tool has an image payload case, so the skill may only need to tell the agent to Read the PNG. Name the path. If none exists, list the needed change and drop "no new engine". Fix the `adb exec-out screencap -p` redirect (stdout, not a file). Do not add downscaling or per-turn caps.
- S8-M1. Lines 37-39, 103, 156-165. Delivery root undecided while paths read as fixed (ZO-2, ZO-3, CO-1, DC-7, IC-2, SY-1). `Resources/bundled-plugins/` does not exist; real resources live under `TurboSparkAppResources`; ZP-07 and ZP-20 use `bundled-skills/`; INDEX orders ZP-18 before ZP-05. Pick one shared root, say which draft owns it, note SkillManager change is a new discovery root.
- S8-M2. Lines 54-59, 76-84. iOS interaction limit contradicts CUJ 1 (ZO-5, CO-5, DC-5, SY-4, IC-1). State in one sentence that iOS v1 supports launch, openurl, screenshot; give CUJ 1 a navigation precondition. Reject idb or XCUITest detection (IC-1).
- S8-M3. Lines 49-50, 82-83. Vision dependency has no fallback (ZO-4, CO-4, IC-4). One or two sentences: offer screenshots to vision models only, fall back to logs and simctl state. No capability-preflight mechanism.
- S8-M4. Lines 143-146, 51. Ladder position and anti-circumvention citation (ZO-9, CO-3, DC-8, IC-3, SY-6). Documentation fix: say what the ladder does for xcrun, adb, emulator, gradle; cite AppModel+AgentMode.swift:55. Downgraded from Blocker. No allowlist.
- S8-M5. Whole document. Missing Risks and Limits section (ST-1, DC-2). Keep short: destructive verbs stay prompted, screenshots and logs are untrusted, skill cleans up devices it booted, builds and boots need timeouts, log bound is the Bash output cap, skill quotes its arguments.
- S8-M6. Lines 123-139. Alternatives thin (DC-1, ST-2, SY-13). Downgraded from Blocker: MCP is rejected with a real reason. Replace "Rejected by author" with one concrete reason each; say B is a scope choice. Do not add four more alternatives.
- S8-N. Open Questions cleanup, `agents/ios-dev.md`, duplicate "launch", Windows non-goal, SwiftUI scaffolding non-goal, ZP-05/07 links, lines 120-121 repeat, CUJ 3 iOS case.

## Reject or defer as scope creep

IC-1 (idb detection), IC-3 (allowlist and classifier test), IC-4 parts (sips downscale, screenshot cap, capability preflight), DC-6 asset-store limit, DC-2 AppShutdownCoordinator code, DC-16 and IC-7 (fake xcrun shim, smoke script, design-check), DC-17 (INDEX size, outside this document), four extra alternatives from DC-1. Heading renames, reordering, field-format polish (ST-6 to ST-10, SY-13 to SY-15) are low value; batch or skip.

Recalibration: only DC-3 is a true Blocker. DC-1 and DC-2 are Majors needing short concrete text. Proportionate verification fix: pass criteria for the hardware run (SY-16, DC-4) recorded as evidence.
