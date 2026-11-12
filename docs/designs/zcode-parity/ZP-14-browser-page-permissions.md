# ZP-14: Browser page permission governance

Finding 6.9 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[embedded-browser permission governance spec](https://github.com/zai-org/ZCode/blob/main/docs/browser-use/2026-10-08-embedded-browser-permission-governance-spec.md),
added after a CNVD finding.

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11
* Depends on: `BrowserOrigin`, the browser log, the per-profile store.
  Related: ZP-15 (browser recording) also touches `WebKitBrowserEngine`;
  recording of capture permissions interacts with media capture here
  (needs author confirmation).

--------------------------------------------------------------------------------

## Overview

Add default-deny governance for permission classes inside the embedded
browser (see the enforcement table in Technical Plan): requests the
delegate can intercept get a confirmation UI, persistent per-site
records, a settings page, and audit log entries. Today the engine
implements no `WKUIDelegate` permission method, so unhandled WebKit
requests fall to default behavior with no user consent surface.

Terms: an agent-owned tab is a tab under agent control
(`AppBrowserAutomationCoordinator` holds an ownership token); user
takeover clears the token via `userTookOver(tabID:)`. "Vault" means the
per-profile store named in the Store section.

## The Problem

Baseline, per `scripts/make-app-bundle.sh`: the app bundle declares a
microphone usage string but no camera or location string, so a page's
microphone request can reach the system TCC prompt with no app-level
policy or record, and camera and location requests are expected to fail
without a prompt. The `WKUIDelegate` extension in
`WebKitBrowserEngine.swift` implements no permission method. Other
classes have no delegate hook (see the enforcement table). Per-class
behavior is to be confirmed with a test page before implementation
(needs author confirmation). ZCode built its governance after a real
vulnerability report (CNVD; source citation needs author confirmation).

## Goals

* Camera and microphone requests (`WKUIDelegate` media capture) default
  to deny at the delegate. Geolocation gets the same treatment through
  the macOS 27 geolocation hook, behind an availability guard
  (deployment target is macOS 14). Every other permission class is
  listed in the enforcement table below as not interceptable by the app.
* Only the classes in the enforcement table are classified. Any class
  the delegate delivers that is not listed is denied. JavaScript dialogs
  are not permission classes and keep their existing handling.
* Sensitive requests raise a four-state confirmation bubble: allow once,
  allow for site, deny once, deny for site, with a per-request timeout
  defaulting to deny. Countdown is 30 seconds (needs author
  confirmation). Deny once and timeout write an audit entry and no
  record. Allow for site and deny for site write a record.
* Persistent per-site allow/deny records power a settings "site
  permissions" page with search and per-record delete.
* Every decision writes an audit log entry (decision, origin, class,
  timestamp) viewable in the pane's info surface.

## Non-Goals

* No change to agent-tab navigation policy or egress isolation (the
  existing per-origin navigation policy stays authoritative). Agent tabs
  do gain a permission deny rule (Decision in Technical Plan); that rule
  does not affect which origins they may navigate to.
* No page permission changes for `ArtifactWebView` (already deny-by-
  default with an explicit content network grant).
* No syncing of records across profiles.
* No screen-share or device source pickers; WKWebView exposes no hook
  for them.
* No change to the system TCC prompt, the file chooser, or downloads. A
  microphone grant at the OS level still applies; this design adds an
  app-level deny in front of it.

## Open Questions

* **Record storage scope**: per profile. Criterion: a profile is the
  existing unit of isolation for sensitive state, so a grant in one
  profile must not apply in another (needs author confirmation).

## Critical User Journeys

### CUJ 1: camera request denied by default

* **Given**: a page in the embedded browser requests camera access.
* **When**: the request hits the delegate.
* **Then**: in an agent-owned tab: denied silently and audited. In a
  user-taken-over tab: the confirmation bubble appears with a countdown;
  no answer before timeout means deny; the chosen state persists per
  site and is visible in the settings page.

### CUJ 2: site permission management

* **Given**: the user granted camera to one site earlier.
* **When**: they open settings, search the site, and delete the record.
* **Then**: the next visit asks again; the deletion is audited.

## Technical Plan (Architecture)

Enforcement table (right-hand column wording needs author
confirmation):

| Class | Enforcement | Default |
| --- | --- | --- |
| Camera, microphone | `WKUIDelegate` media capture, macOS 12+ | deny |
| Geolocation | `WKUIDelegate` geolocation, macOS 27 only | deny; on macOS 14-26 rely on missing `NSLocationUsageDescription` |
| Notifications, clipboard-read, screen share, MIDI/HID/USB/serial/bluetooth | no WKUIDelegate hook | not governed by this design; behavior to be recorded per the baseline test |

```text
WebKitBrowserEngine (WKUIDelegate extension, permission methods added)
  webView(_:requestMediaCapturePermissionFor:initiatingFrame:
          decisionHandler:)      -> SitePermissionPolicy
  webView(_:requestDevicePermissionFor:...) -> policy
  JavaScript dialogs are not permission classes (keep current handling)
       |
       v
SitePermissionPolicy
  1. ownership check: agent-owned -> deny, no bubble, no lookup
  2. classification table (enforcement table above)
  3. lookup: per-site persistent record -> cached decision
  4. miss: raise PermissionBubble (four states, countdown,
     default deny on timeout)
       |
       v
SitePermissionStore
  records keyed (profile, origin, class) in the vault
  settings page binding; audit log via the existing
  browser log store
```

The delegate completes decisions synchronously from records and
asynchronously from bubbles, always calling the WebKit decision handler
exactly once.

**Decision**: agent-owned tabs hard-deny every sensitive class with no
bubble and no record lookup. Stored allow records are ignored in
agent-owned tabs. Ownership is read when the request arrives. If
ownership changes while a bubble is pending, the bubble is dismissed and
the request is denied. The decision handler is called exactly once on
every path.

Edge cases, all resolving to deny: the tab navigates or closes while a
request is pending; the tab moves to background; the web content
process terminates (`webViewWebContentProcessDidTerminate`); a second
request for the same origin and class arrives while one is pending
(coalesce to the first answer); the countdown fires. The timeout is a
named constant `sitePermissionPromptTimeout`, value 30 seconds (needs
author confirmation).

Requirements-to-design mapping:

| Goal | Component |
| --- | --- |
| Goal 1 | `WebKitBrowserEngine` delegate plus `SitePermissionPolicy` |
| Goal 3 | pending-dialog model plus bubble |
| Goal 4 | `SitePermissionStore` and settings pane |
| Goal 5 | browser log event |

`SitePermissionStore` is separate from `BrowserPermissionRuleStore`
(navigation allowlist) and shares only the `BrowserOrigin` type (or the
opposite, per the Alternative C decision; needs author decision).

## Alternatives Considered

Trade-offs below are inferred from the review findings (needs author
confirmation).

### Alternative A: do nothing and rely on WebKit and TCC

* **Pros**: no work.
* **Cons**: no app record, no audit, microphone requests reach the
  system prompt from agent tabs.
* **Reason for Rejection**: Rejected: fails the agent-tab requirement.

### Alternative B: hard-deny everything now, ship no UI

* **Pros**: smallest surface, closes the agent-tab exposure
  immediately, no bubble stack.
* **Cons**: breaks user-driven conferencing and capture.
* **Reason for Rejection**: Deferred option: ship this first and add the
  bubble later.

### Alternative C: extend `BrowserPermissionRuleStore` and `BrowserPermissionApprovalCoordinator`

* **Pros**: one origin type, one store, existing approval flow, instead
  of new `SitePermissionPolicy` and `SitePermissionStore`.
* **Cons**: the store currently holds only navigation origins; it needs
  a permission-class dimension.

### Alternative D: session-only grants, no persistence

* **Pros**: no stored records, no retention question.
* **Cons**: repeated prompts.

### Selected design

State which of the alternatives above is chosen and why. Needs author
decision.

## Risks and Mitigations

All mitigations are deny-by-default (needs author confirmation).

* **Prompt flood**: one pending bubble per tab, further requests deny.
* **Spoofed bubble**: the origin line is rendered by the app from
  `BrowserOrigin.canonicalString`, never from page text.
* **Redirect mid-prompt**: navigation dismisses the bubble and denies.
* **Cross-origin iframe**: keyed by the requesting frame origin, both
  origins shown.
* **Decision-handler leak on tab close**: a single-shot wrapper calls
  deny on deinit.
* **Stale allow records**: deletable in settings, no automatic expiry in
  v1.
* **Unreadable or corrupt record store**: treat every request as a miss
  and deny in agent tabs, prompt in user tabs; quarantine the file per
  `AppJSONStore` behavior.

## Non-Functional Requirements

* **Security**: default deny; countdown default deny; records are keyed
  by `BrowserOrigin.canonicalString` (exact scheme, host, effective
  port); no registrable-domain widening, so `www.example.com` and
  `example.com` are separate records. A request from a subframe is keyed
  by the requesting frame's origin from `WKFrameInfo.securityOrigin`,
  not the top-level page, and the bubble shows both when they differ.
  Audit entries go to the existing browser log through a new event type
  `BrowserPermissionDiagnosticEvent` carrying class, decision (allow
  once, allow site, deny, auto-deny), canonical origin, and timestamp.
  No page content or URL path is logged. The log keeps its existing cap;
  records in the store are capped at 500 (needs author confirmation).
  Record deletion writes an audit entry, so "append-only" means entries
  are never edited, not that records are permanent. The audit viewer is
  one surface only: the settings pane.
* **Privacy**: records live in the profile vault; no records leave the
  machine.
* **Accessibility and localization**: bubble states are announced;
  strings in the catalog for 21 languages; DESIGN.md compliant bubble.

## Detailed Implementation Plan

### Target Files to Create/Modify (build order: 2, 1, 3, 4, 5, 6)

1. `swift/TurboSparkApp/Sources/TurboSparkApp/BrowserAutomation/SitePermissionPolicy.swift` (Create; placement next to the rule store needs author confirmation)
   * **Type / Scope**: classification and decision logic.
   * **Proposed Changes**: class table, record lookup, agent-owned hard
     deny branch, countdown bubble orchestration, single decision
     handler call guarantee.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/BrowserAutomation/SitePermissionStore.swift` (Create; placement next to the rule store needs author confirmation)
   * **Type / Scope**: persistence.
   * **Proposed Changes**: per-profile records in the vault, keyed by
     `BrowserOrigin.canonicalString` (exact scheme, host, effective
     port), settings bindings, audit entries into the browser log
     through a new `BrowserPermissionDiagnosticEvent` type. Persistence
     mechanism (profile database vs `AppJSONStore`) needs author
     decision.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/BrowserAutomation/WebKitBrowserEngine.swift` (Modify)
   * **Type / Scope**: delegate.
   * **Proposed Changes**: add the permission methods to the
     `WKUIDelegate` extension, routing through SitePermissionPolicy;
     never call decision handlers twice.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Browser/SitePermissionBubbleView.swift` (Create)
   * **Type / Scope**: confirmation UI.
   * **Proposed Changes**: extend the pending-dialog request model with
     a permission kind and reuse `BrowserDialogApprovalView`'s resolve
     path instead of a new bubble stack. Four-state bubble with
     countdown fill, Reduce Motion compliant transitions.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/Browser/BrowserSettingsPaneView.swift` (Modify), with bindings in `BrowserSettingsViewModel.swift` (Modify)
   * **Type / Scope**: settings.
   * **Proposed Changes**: site permissions section with search,
     per-record delete, and the audit log viewer entry.
6. `swift/TurboSparkApp/Tests/TurboSparkAppTests/SitePermissionTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: classification table, record lookup hit and
     miss, timeout-default-deny, agent-owned hard deny, single decision
     call, records for www and apex stay distinct (default-port and
     trailing-dot hosts canonicalize to one record), record delete.
     Also: an injected clock for the timeout; store round trip and the
     corrupt-file path; an audit event emitted on allow, deny, timeout,
     and record delete; `ArtifactWebView` unchanged. Seam: delegate calls
     go through a single-shot decision wrapper that tests can drive
     without WebKit.
   * **Mutation checks**: swap a sensitive class to harmless and assert
     the classification test fails; allow double decision-handler calls
     and assert the guard test fails; remove the agent-tab deny, change
     the timeout default to allow, and change the unknown-class default
     to allow, each asserted to fail a test.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the permission tests.
* **Static Analysis and Linting**: `make design-check` for bubble and
  settings UI.
* **Localization**: catalog additions in 21 languages.
* **Manual**: against local test pages requesting camera, geolocation,
  and notifications, exercise allow-once, allow-site, deny-site, and
  timeout paths in both tab ownership modes.
* **Exit criterion**: all of the above pass plus the manual run. Local
  green is not release evidence; follow `.claude/docs/verification.md`.

--------------------------------------------------------------------------------

*Drafted with EGM*
