# ZP-18: Mobile device control (iOS simulator, Android emulator)

Finding 9.5 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's
[ios-simulator plugin](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/ios-simulator-plugin/skills/ios-dev/SKILL.md)
and
[android-emulator plugin](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/android-emulator-plugin/README.md).

* Author: whit3rabbit
* Status: Implemented (see Implementation Notes)
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Let the agent drive mobile development loops: build, install, launch,
screenshot, and interact with apps in the iOS Simulator, and the same
loop against the Android emulator when the user has the SDK installed.
ZCode ships both as official plugins; TurboSpark ships them as bundled
plugins whose skills instruct the agent to drive the platforms' own
shell tools through the existing Bash tool, with screenshot handoff
through the existing image attachment path.

## The Problem

The agent's device reach is the local shell and the in-app browser.
An iOS developer asking "run my app in the simulator and check the login
screen" gets shell advice but no loop: no structured preflight, no
build-and-run, no screenshot capture into the conversation, no UI
automation. ZCode's plugins package this loop; the capability is pure
shell-tool orchestration (xcrun simctl, gradle), so it fits TurboSpark's
no-bundled-runtime constraint.

## Goals

* Two bundled plugins: `ios-simulator` and `android-emulator`, delivered
  through the ZP-05 builtin seeding mechanism (or the bundled-skill root
  day one, matching ZP-07's choice).
* The iOS plugin provides a skill covering: preflight (Xcode and
  runtime availability), simulator discovery and boot, project discovery
  (xcodeproj/workspaces), build and install (`xcodebuild` with
  destination), launch, screenshot into the chat, deep link open,
  launch/terminate, and log capture.
* The Android plugin provides a skill covering: SDK/avdmanager/emulator
  preflight (all user-installed), avd lifecycle, gradle build and
  install, screenshot, logcat capture, and basic UI automation via
  `adb shell input`.
* Screenshots flow into the conversation as image attachments so vision
  models can see the screen.
* Everything runs through the ordinary Bash tool and permission ladder;
  the plugins add no new execution engine.

## Non-Goals

* No idb-based UI automation for iOS in v1 (ZCode marks it optional);
  tap/swipe/type via simctl is not available, so iOS UI interaction is
  limited to what `simctl` and accessibility-driven shell tools offer;
  the skill documents this limit.
* No Android SDK or emulator bundling; the plugin requires
  user-installed tooling and says so in its preflight.
* No Windows-hosted Android emulation.

## Open Questions

* **Screenshot path for iOS**: `xcrun simctl io screenshot` writes PNG
  directly; no open question, recorded here as settled. Android:
  `adb exec-out screencap -p`. Both land as files the attachment path
  can read. No platform gap identified.
* **UI automation depth**: whether to adopt Facebook idb (external
  dependency) later for iOS tap/swipe/type. Deferred; note in the skill
  as future work.

## Critical User Journeys

### CUJ 1: iOS build-run-screenshot loop

* **Given**: a workspace containing an Xcode project.
* **When**: the user asks the agent to run and verify the login screen.
* **Then**: the agent preflights, boots a simulator, builds with
  xcodebuild, installs, launches, screenshots, and the screenshot
  appears in the conversation; the vision-capable model describes what
  it sees and reports back.

### CUJ 2: Android smoke

* **Given**: the user has Android Studio's SDK installed.
* **When**: the agent is asked to smoke-test a debug build.
* **Then**: the agent creates or reuses an avd, builds with gradle,
  installs via adb, launches, screenshots, and pulls logcat; failures
  surface with the captured log excerpt.

### CUJ 3: preflight honesty

* **Given**: no Android SDK installed.
* **When**: the android plugin's skill runs its preflight.
* **Then**: the agent reports the missing tooling with install pointers
  instead of attempting and failing opaquely.

## Technical Plan (Architecture)

```text
Bundled plugins (ZP-05 seeding or bundled-skill root)
  ios-simulator/
    SKILL.md (loop instructions, preflight, limits)
    agents/ios-dev.md (optional specialist agent)
  android-emulator/
    SKILL.md (loop instructions, preflight, limits)
        |
        v
Existing Bash tool + permission ladder
  xcrun simctl / xcodebuild        (macOS built-in)
  adb / emulator / gradle           (user-installed SDK)
        |
        v
Existing image attachment path
  screenshots reach vision models as attachments
```

No new execution engine, no MCP server required. Skills carry the
orchestration knowledge; the tools are the ones that already exist.

## Alternatives Considered

### Alternative A: native MCP servers like ZCode's

* **Pros**: structured tool schemas.
* **Cons**: ZCode's servers wrap the same shell commands; a bundled
  node-based MCP server would violate the runtime constraint, and a
  native Swift MCP server duplicates what skills already express.
* **Reason for Rejection**: Rejected by author: skills plus existing
  tools meet the capability without new machinery.

### Alternative B: port only the Android plugin

* **Pros**: adb automation is richer than simctl.
* **Cons**: iOS simulator is the more common loop on a macOS-first app.
* **Reason for Rejection**: Rejected by author: ship both, document the
  iOS interaction limit.

## Non-Functional Requirements

* **Security**: all commands flow through the Bash ladder; simulator
  and emulator management are medium risk; the skills never instruct
  bypassing approvals or shell route-arounds (the classifier's
  anti-circumvention rule applies verbatim).
* **Performance**: screenshots are PNG files; log captures are bounded
  (tail N lines) to protect context.
* **Localization**: plugin listing strings in the catalog; skill content
  is agent-facing English.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-plugins/ios-simulator/SKILL.md` (Create)
   * **Type / Scope**: skill content.
   * **Proposed Changes**: preflight, discovery, build, install, launch,
     screenshot, deep link, logs, documented iOS interaction limits.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-plugins/ios-simulator/.claude-plugin/plugin.json` (Create)
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-plugins/android-emulator/SKILL.md` (Create)
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-plugins/android-emulator/.claude-plugin/plugin.json` (Create)
5. `swift/TurboSparkApp/Sources/TurboSparkApp/State/SkillManager.swift` (Modify)
   * **Type / Scope**: discovery.
   * **Proposed Changes**: discover the bundled plugin skills.
6. `swift/TurboSparkApp/Tests/TurboSparkAppTests/BundledPluginsTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: manifests parse, skills discover and
     activate, skill text contains preflight-before-act and
     no-route-around clauses.
   * Mutation checks: remove the preflight clause and assert the
     content test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the bundled plugin tests.
* **Manual (hardware)**: run the iOS loop end to end on a machine with
  Xcode; run the Android loop if an SDK is present; verify screenshots
  reach a vision-capable model and are described correctly.

## Implementation Notes

Shipped as two bundled skills, `ios-simulator` and `android-emulator`, under
`swift/TurboSparkApp/Sources/TurboSparkAppResources/BundledSkills/`, the root
built in ZP-07 and used by ZP-20. This is the delivery the INDEX review chose:
ZP-05 is a later re-delivery path, not a prerequisite. No `SkillManager`
change was needed, and no `plugin.json`, no `agents/ios-dev.md` and no
`bundled-plugins/` folder exist (review M1, N2). Tests:
`BundledDeviceControlSkillsTests`. No UI strings were added, so no
localization change (N11).

Deviations from the draft, each a review finding:

* **Screenshot handoff (B1).** The draft's claim was false: Bash returns text,
  `read_file` reads text, and only the in-app browser and Computer Use produce
  image-bearing tool results. Rather than add an image path to `read_file`
  (a new execution surface, out of scope for a skill), the skills say the PNG
  is not visible to the model, show it to the user with `send_user_file`, and
  let a vision model look at the Simulator or Emulator window through
  `computer_request_access` and `computer_screenshot` only when the user has
  turned Computer Use on and approves. Without vision or Computer Use the
  skill verifies from text evidence and says it did not see the screen
  (M3). A `read_file` image path is deferred follow-up work. Android
  screenshots use `screencap` then `adb pull`, because
  `exec-out screencap -p` needs a shell redirect (a prompted shell program).
* **Ladder position (M4), stated and enforced.** `xcrun`, `xcodebuild`,
  `adb`, `emulator`, `avdmanager`, `sdkmanager`, `open` and `./gradlew` are
  not on `TerminalCommandClassifier`'s allowlist, so
  `ToolRiskClassifier.assessRisk` scores them `.high` and they ask under Ask
  and Auto; Agent mode's judge decides, per
  [PERMISSION_GATE.md](../../PERMISSION_GATE.md). The skills never ask for a
  mode change and carry the don't-route-around guidance that
  `AppModel.blockMessage` already gives the model. `gradle` and `gradlew` were
  later removed from the build allowlist (Gradle runs the workspace's build
  scripts), so they ask like the other device commands. The skill still
  confines them to `assembleDebug` and installs through `adb`.
  `testEveryDeviceCommandIsHighRiskAndNeverAutoApprovable` extracts every
  fenced command from both skills and checks it against the real classifier.
* **Risks and Limits (B2).** The skills fix their scope in text, and a test
  scans fenced commands for forbidden verbs: simulators and emulators only
  (no `devicectl`, a non-`emulator-` serial needs the user to name it), debug
  builds only, `CODE_SIGNING_ALLOWED=NO` and no `-allowProvisioningUpdates`
  (no Apple account use), no credential typing, no `adb tcpip`, `connect`,
  `pair`, `forward`, `reverse` or `root` (no network exposure), no erase,
  uninstall, `pm clear` or delete. The skill boots and stops only what it
  started. Screenshots, UI dumps and logs are declared untrusted content.
* **Bounded output (M7).** The Bash tool keeps 30,000 characters (head and
  tail) and a 120 s default timeout, 600 s maximum. The skills use
  `logcat -d -t 200`, `log show --last 2m` with a predicate, background runs
  for builds, and name the cap.
* **iOS interaction limit (M2).** v1 on iOS is launch, open URL, screenshot.
  `simctl` cannot tap, swipe or type, so CUJ 1 reaches the login screen by
  launch arguments, a deep link, or a user-prepared Simulator. Android adds
  `adb shell input` and `uiautomator dump`. idb, XCUITest and the
  SwiftUI scaffolding ZCode ships stay non-goals.
* **Preflight (CUJ 3).** Both skills run a host check first and stop with
  install pointers (Xcode, `xcode-select`, runtime, no AVD, arm64-v8a image
  on Apple Silicon) instead of attempting a build; a test pins the order.
* **Evidence not gathered.** The hardware loop (Xcode build, boot, launch,
  screenshot; Android emulator) was not run end to end in this change. The
  tests prove discovery, ladder scoring and clause presence, not that a
  model follows the loop. Pass criteria for a manual run: preflight output is
  reported, the app launches (exit status 0 from `simctl launch` or
  `monkey`), a PNG exists at the stated path and is offered with
  `send_user_file`, the log excerpt is under 200 lines, and the agent states
  whether it saw the screen.

--------------------------------------------------------------------------------

*Drafted with EGM*
