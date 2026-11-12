# Stage 6: Comprehension report (ZP-18)

Target: docs/designs/zcode-parity/ZP-18-mobile-device-control.md (lines 1-184). Reviewer: egm-comprehension (completed). No document standard. A paraphrase of every section was possible, so no Blocker.

## What this proposes

The draft proposes two plugins that ship inside the app, `ios-simulator` and `android-emulator`. Each contains a SKILL.md, a markdown instruction file that tells the agent how to run a mobile dev loop. The loop is: check the tooling is installed, boot a simulator or emulator, build, install, launch, take a screenshot, and read logs. For iOS the agent uses the Xcode command-line tools (`xcrun simctl`, `xcodebuild`). For Android it uses `adb`, `emulator`, `avdmanager` and `gradle`, all user-installed. No new code is written to execute anything. The agent runs the commands through the existing Bash tool, so the existing permission checks apply. Screenshots are PNG files that the existing image attachment path passes to vision-capable models. The work is Swift resource files, one change to `SkillManager.swift` so it finds the plugin skills, and a content test. iOS tap/swipe/type is explicitly left out of v1.

## How the existing system works, per this document

- The agent can already run local shell commands and use an in-app browser, but it has no structured mobile loop.
- There is a Bash tool gated by a "permission ladder", which rates commands by risk. Simulator and emulator management is called "medium risk". A "classifier" has an "anti-circumvention rule" that forbids bypassing approvals.
- There is an "image attachment path" that can read image files into the conversation.
- There are skills, discovered by `SkillManager.swift`.
- There is a "ZP-05 builtin seeding mechanism" and a "bundled-skill root" (ZP-07 is said to have chosen between them), both ways of shipping built-in plugins. The draft does not say which one exists today.
- There is a "no-bundled-runtime constraint" that rules out a node-based MCP server.

## Findings (CO)

Major
- CO-1. Lines 37-39, 103, 156-165. Delivery mechanism unresolved but the plan reads as fixed. ZP-05 and ZP-07 are not linked, ZP-07's choice is not stated, and INDEX sequences ZP-18 (step 3) before ZP-05 (step 4) while calling ZP-05 the delivery mechanism. State the default path.
- CO-2. Lines 49, 66-69, 116-118. Screenshot handoff asserted, not explained: how a PNG written by Bash becomes an attachment, where it is written, how it is cleaned up. CUJ 1 depends on it. "Settled" has no code reference.
- CO-3. Lines 143-146. "Bash ladder", "permission ladder", "medium risk", "classifier's anti-circumvention rule applies verbatim" are undefined and unlinked; INDEX names `AppToolPermissionEngine` but the draft does not. Unclear whether the risk level is a property of these commands or an assertion.
- CO-4. Lines 49-50, 82-83. Vision assumption unstated; no behavior for a model without vision in a local-first app.
- CO-5. Lines 47-48, 56-59. iOS interaction limit vague: no named accessibility-driven shell tool; asymmetry with Android `adb shell input` not summarized; "ZCode marks it optional" is opaque without the source.
- CO-6. Lines 164-180. Test plan has unexplained terms: the mutation check is a one-line aside, asserted wording is unspecified, "activate" and `make swift-lib` undefined. Content tests prove phrases exist, not behavior.

Nit
- CO-7. Lines 40-44. "launch" then "launch/terminate" repeated; stop/uninstall scope and meaning of "runtime availability" unclear.
- CO-8. Lines 102-109, 154-173. `agents/ios-dev.md` "optional" has no plan; plugin layout (manifest vs skills) unexplained.
- CO-9. Lines 31-33, 126-130. "No-bundled-runtime constraint" defined only in INDEX; MCP undefined; why a Swift MCP server would break the constraint is not shown.
- CO-10. Lines 20-21, 147-150. "TurboSpark ships them" reads as present fact; "tail N" unspecified; "the catalog" unspecified.
- CO-11. Lines 64-72. Open Questions holds a settled item and omits the real open choice (delivery mechanism).
- CO-12. Lines 85-98. Android CUJs thin: no AVD or system image, or boot failure case; CUJ 3 lacks the iOS (no Xcode) case.
