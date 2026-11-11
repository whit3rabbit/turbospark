# ZP-18: Mobile device control (iOS simulator, Android emulator)

Finding 9.5 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's
[ios-simulator plugin](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/ios-simulator-plugin/skills/ios-dev/SKILL.md)
and
[android-emulator plugin](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/android-emulator-plugin/README.md).

* Author: whit3rabbit
* Status: Draft
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

--------------------------------------------------------------------------------

*Drafted with EGM*
