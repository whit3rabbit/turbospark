# ZP-14: Browser page permission governance

Finding 6.9 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[embedded-browser permission governance spec](https://github.com/zai-org/ZCode/blob/main/docs/browser-use/2026-10-08-embedded-browser-permission-governance-spec.md),
added after a CNVD finding.

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Add default-deny governance for device and sensitive permissions inside
the embedded browser: camera, microphone, geolocation, notifications,
clipboard-read, screen share, and device access requests get a
confirmation UI, persistent per-site records, a settings page, and audit
log entries. Today the engine's `WKUIDelegate` permission handling is
explicitly empty, so unhandled WebKit requests fall to default behavior
with no user consent surface.

## The Problem

WebKit's default behavior for unhandled permission requests is a silent
or system-level decision with no app-level record. A page the agent
navigates could request camera or geolocation and get WebKit's built-in
prompting (or silence, depending on the request class), with no per-site
policy, no way to review what was granted, and no audit trail. ZCode
built this after a real vulnerability report (CNVD); shipping a browser
without page-permission governance repeats the exposure.

## Goals

* All sensitive permission requests (media capture, geolocation,
  notifications, clipboard-read, screen share, MIDI/HID/USB/serial/
  bluetooth device classes) default to deny at the WebKit delegate level.
* Harmless classes (for example simple alerts, basic focus) pass
  silently.
* Sensitive requests raise a four-state confirmation bubble: allow once,
  allow for site, deny once, deny for site, with a per-request timeout
  defaulting to deny.
* Screen share shows a source picker; device requests show a device
  picker where WebKit exposes choices.
* Persistent per-site allow/deny records power a settings "site
  permissions" page with search and per-record delete.
* Every decision writes an audit log entry (decision, origin, class,
  timestamp) viewable in the pane's info surface.

## Non-Goals

* No governance change for agent-owned tabs' navigation (the existing
  per-origin navigation policy and egress isolation stay authoritative).
* No page permission changes for `ArtifactWebView` (already deny-by-
  default with an explicit content network grant).
* No syncing of records across profiles.

## Open Questions

* **Record storage scope**: per profile (records live in the profile
  vault like other sensitive state) vs app-wide. Lean per profile,
  consistent with the vault posture.
* **Agent-owned tabs**: should the agent's tabs be allowed to trigger
  permission bubbles at all, or should sensitive classes be hard-denied
  with no bubble when the tab is agent-owned? ZCode's governance is
  user-centric; the strongest posture for agent tabs is hard deny. Lean
  hard deny for agent-owned tabs, bubbles for user takeover. Decide at
  review; the existing "user takeover before direct input" state makes
  this boundary detectable.

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

### CUJ 3: screen share source picker

* **Given**: a conferencing page requests screen share in a
  user-taken-over tab.
* **When**: the user allows once.
* **Then**: the source picker lists windows and screens; the chosen
  source is used for this grant only.

## Technical Plan (Architecture)

```text
WebKitBrowserEngine (WKUIDelegate, filled in)
  webView(_:requestMediaCapturePermissionFor:initiatingFrame:
          decisionHandler:)      -> SitePermissionPolicy
  webView(_:requestDevicePermissionFor:...) -> policy
  runJavaScriptAlert/Confirm (harmless: keep current dialogs)
       |
       v
SitePermissionPolicy
  classification table (sensitive vs harmless classes)
  lookup: per-site persistent record -> cached decision
  miss: raise PermissionBubble (four states, countdown,
        default deny on timeout); agent-owned: deny
       |
       v
SitePermissionStore
  records keyed (profile, origin, class) in the vault
  settings page binding; audit log via the existing
  browser log store
```

Classification: sensitive classes default-deny; harmless pass through.
The delegate completes decisions synchronously from records and
asynchronously from bubbles, always calling the WebKit decision handler
exactly once.

## Alternatives Considered

### Alternative A: rely on WebKit's per-site ITP-style prompts

* **Pros**: zero work.
* **Cons**: no app records, no audit, inconsistent classes, and agent
  tabs would still reach system prompts; this is the current exposure.
* **Reason for Rejection**: Rejected by author: governance is the
  feature.

### Alternative B: blanket deny all sensitive permissions, no UI

* **Pros**: smallest surface.
* **Cons**: breaks legitimate conferencing/capture use in user-driven
  browsing with no recourse; ZCode chose governance over blanket deny
  for the same reason.
* **Reason for Rejection**: Rejected by author: keep user-driven use
  possible.

## Non-Functional Requirements

* **Security**: default deny; countdown default deny; records are
  origin-canonicalized (scheme plus registrable domain per the existing
  origin rules used by the agent allowlist); the audit log is
  append-only within the session.
* **Privacy**: records live in the profile vault; no records leave the
  machine.
* **Accessibility and localization**: bubble states are announced;
  strings in the catalog for 21 languages; DESIGN.md compliant bubble.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Browser/SitePermissionPolicy.swift` (Create)
   * **Type / Scope**: classification and decision logic.
   * **Proposed Changes**: class table, record lookup, agent-owned hard
     deny branch, countdown bubble orchestration, single decision
     handler call guarantee.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Browser/SitePermissionStore.swift` (Create)
   * **Type / Scope**: persistence.
   * **Proposed Changes**: per-profile records in the vault, origin
     canonicalization, settings bindings, audit entries into the
     browser log store.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/BrowserAutomation/WebKitBrowserEngine.swift` (Modify)
   * **Type / Scope**: delegate.
   * **Proposed Changes**: fill the currently empty WKUIDelegate
     permission conformances, routing through SitePermissionPolicy;
     never call decision handlers twice.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Browser/SitePermissionBubbleView.swift` (Create)
   * **Type / Scope**: confirmation UI.
   * **Proposed Changes**: four-state bubble with countdown fill, source
     and device pickers, Reduce Motion compliant transitions.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/App/SettingsPanes.swift` (Modify; basename, Browser settings pane owner)
   * **Type / Scope**: settings.
   * **Proposed Changes**: site permissions section with search,
     per-record delete, and the audit log viewer entry.
6. `swift/TurboSparkApp/Tests/TurboSparkAppTests/SitePermissionTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: classification table, record lookup hit and
     miss, timeout-default-deny, agent-owned hard deny, single decision
     call, canonicalization equivalence (www vs apex), record delete.
   * Mutation checks: swap a sensitive class to harmless and assert the
     classification test fails; allow double decision-handler calls and
     assert the guard test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the permission tests.
* **Static Analysis and Linting**: `make design-check` for bubble and
  settings UI.
* **Localization**: catalog additions in 21 languages.
* **Manual**: against local test pages requesting camera, geolocation,
  and notifications, exercise allow-once, allow-site, deny-site, and
  timeout paths in both tab ownership modes.

--------------------------------------------------------------------------------

*Drafted with EGM*
