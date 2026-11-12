# Goldfish Annotated Review Report

Status: resolved

**Target:** `docs/designs/zcode-parity/ZP-14-browser-page-permissions.md` (full file, 204 lines)
**Findings source:** `docs/egm-reviews/zcode-parity/ZP-14/consolidated_review_report.md`

Triage only. The target was not edited. Items marked "needs author confirmation" contain text I had to invent.

## Summary

| Category | Count | Items |
| --- | --- | --- |
| PROPOSE FIX | 22 | B1-B5, M1-M9, N2, N4, N5, N6, N7, N8, N9, N10 |
| CONTEST | 1 | N1 |
| ESCALATE | 1 | N11 |
| DECLINE | 2 | N3, N12 (N12 carries one small optional Non-Goals edit) |

Total 26 consolidated items (5 Blocker, 9 Major, 12 Nit; the report numbers findings Issue 1-26 below in that order, with N12 as Issue 26). No Blocker or Major was contested. No `.kiro/` specs were involved.

## Verified-false or partly false reviewer claims

1. **B2 (partly false):** "no camera, microphone, location usage strings or entitlements found." `scripts/make-app-bundle.sh` line 324 declares `NSMicrophoneUsageDescription` (also `NSAudioCaptureUsageDescription` and `NSScreenCaptureUsageDescription` at lines 326-328). Camera and location strings are indeed absent. So microphone requests can reach the system TCC prompt today. The exposure is not near zero for microphone.
2. **N1 (false):** the doc's "21 languages" is correct. `swift/docs/SWIFT_LOCALIZATION.md` line 37 says the catalog carries 21 languages. "Ten" in `AGENTS.md` and `swift/AGENTS.md` counts parity gates, not languages. Reviewers conflated the two.
3. **N5 (partly false):** the ladder position is already stated. `docs/designs/zcode-parity/INDEX.md` line 98 places ZP-14 in rung 2 ("medium, app-local"). The dependency and ZP-15 overlap notes are still absent.
4. **N2 (partly false):** reviewers say alert/confirm/prompt delegates exist. They do, but in the class body (`WebKitBrowserEngine.swift` lines 866, 884, 908, plus `createWebViewWith` at 1021), not in the `extension WebKitBrowserEngine: WKUIDelegate` at line 1492, which is empty. The doc's "permission handling is explicitly empty" is accurate. Only the wording "filled in" and "currently empty conformances" is loose.

Reviewer claims verified TRUE: `BrowserOrigin` is not registrable-domain keyed (B3); `App/SettingsPanes.swift` does not exist (M6); `BrowserAutomationDiagnosticEvent` is a closed shape (M2).

## Facts read from the repo (used below)

* `BrowserOrigin` (`BrowserAutomation/BrowserPermissionRuleStore.swift` lines 4-43): fields `scheme`, `host`, `effectivePort`; `canonicalString` is `scheme://host[:port]` with the default port omitted. http and https only. Trailing dot on host is stripped. No public-suffix logic. So the doc's "scheme plus registrable domain" is wrong.
* WKUIDelegate permission hooks, from the installed SDK header `WKUIDelegate.h`: `requestMediaCapturePermissionForOrigin` (macOS 12+, camera and microphone); `requestDeviceOrientationAndMotionPermissionForOrigin` (marked unavailable on macOS); `requestGeolocationPermissionForOrigin` (macOS 27.0 only, deployment target is `.macOS(.v14)` in `Package.swift` line 19, so it needs an availability guard). There is no `requestDevicePermissionFor`, and no hook for notifications, clipboard-read, screen share, MIDI, HID, USB, serial, or bluetooth. The doc's delegate signature at lines 101-103 is not a real WKUIDelegate method name in this form (the real selector is `webView(_:decideMediaCapturePermissionsFor:initiatedBy:type:)`).
* Existing pieces: `BrowserPermissionRuleStore`, `BrowserPermissionApprovalCoordinator`, `Browser/BrowserDialogApprovalView.swift`, `BrowserAutomationLogger.swift`. User takeover is `AppBrowserAutomationCoordinator.userTookOver(tabID:)` (line 104), which clears `ownershipToken`.
* Settings files: `Browser/BrowserSettingsPaneView.swift`, `BrowserSettingsViewModel.swift`, `BrowserSettings.swift`. `App/` holds `AppSettingsView.swift`, `ToolsSettingsPaneView.swift`, etc., but no `SettingsPanes.swift`.

## Prioritized Findings & AI Suggestions

### [Issue 1] [`consolidated B1`] - Severity: Blocker
*   **Affected Scope:** [Lines 16-18, 36-38, 44-45, 89-95, 100-104](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L16-L104) (Overview, Goals, CUJ 3, Architecture)
*   **Discrepancy Description:** Goals promise classes WKWebView does not expose. Only media capture has a delegate hook (plus geolocation on macOS 27). No screen-share source picker exists; CUJ 3 is unbuildable.
*   **Rationale:** Design promises more than the platform allows.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Reviewer is right, verified against `WKUIDelegate.h`. Security: the safe reading is that unhooked classes are not governed by the app at all, so the doc must say who decides them (WebKit or the OS) and not claim default deny where the app has no control. For classes the app cannot intercept, the only conservative levers are to not request the OS entitlement or usage string (camera and location have none today) and to disable the feature in the configuration where WebKit offers a switch.
*   ***Proposed Edit:***
    *   Location: replace Goals bullet 1 (lines 36-38) and bullet 4 (lines 44-45); delete CUJ 3 (lines 89-95); add table after line 123.
    *   Change: Goal 1 becomes: "Camera and microphone requests (`WKUIDelegate` media capture) default to deny at the delegate. Geolocation gets the same treatment through the macOS 27 geolocation hook, behind an availability guard (deployment target is macOS 14). Every other class is listed in the enforcement table below as not interceptable by the app." Goal 4 is removed. CUJ 3 is removed and a Non-Goal is added: "No screen-share or device source pickers; WKWebView exposes no hook for them." Add this table (needs author confirmation for the right-hand column wording):

        | Class | Enforcement | Default |
        | --- | --- | --- |
        | Camera, microphone | `WKUIDelegate` media capture, macOS 12+ | deny |
        | Geolocation | `WKUIDelegate` geolocation, macOS 27 only | deny; on macOS 14-26 rely on missing `NSLocationUsageDescription` |
        | Notifications, clipboard-read, screen share, MIDI/HID/USB/serial/bluetooth | no WKUIDelegate hook | not governed by this design; behavior to be recorded per B2 |

### [Issue 2] [`consolidated B2`] - Severity: Blocker
*   **Affected Scope:** [Lines 20-32, 120-123, 169-172](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L20-L172) (Overview, The Problem)
*   **Discrepancy Description:** Baseline behavior is unverified and stated inconsistently; exposure may be near zero.
*   **Rationale:** The problem statement cannot justify the work without a measured baseline.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Mostly right, but the evidence claim is partly wrong (see verified-false item 1): `NSMicrophoneUsageDescription` is present in `scripts/make-app-bundle.sh` line 324, while camera and location strings are not. I did not read entitlements, so I did not verify the sandbox state. Security-relevant: the microphone path can reach a system prompt with no app-level record, so the problem is real for that class.
*   ***Proposed Edit:***
    *   Location: replace "The Problem" first two sentences (lines 26-28).
    *   Change: "Baseline, per `scripts/make-app-bundle.sh`: the app bundle declares a microphone usage string but no camera or location string, so a page's microphone request can reach the system TCC prompt with no app-level policy or record, and camera and location requests are expected to fail without a prompt. The `WKUIDelegate` extension in `WebKitBrowserEngine.swift` implements no permission method. Other classes have no delegate hook (see the enforcement table). Per-class behavior is to be confirmed with a test page before implementation (needs author confirmation)."

### [Issue 3] [`consolidated B3`] - Severity: Blocker
*   **Affected Scope:** [Lines 146-149, 185-186](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L146-L186) (Non-Functional Requirements, Tests)
*   **Discrepancy Description:** Origin key "scheme plus registrable domain per the existing origin rules" is false; `BrowserOrigin` is exact scheme, host, effective port.
*   **Rationale:** Registrable-domain keying needs a public suffix list and widens grants to sibling subdomains and shared-hosting tenants.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified. `BrowserOrigin` has `scheme`, `host`, `effectivePort`, and `canonicalString` (`BrowserPermissionRuleStore.swift` lines 4-43). Security: widening to a registrable domain would let a grant made for one subdomain apply to unrelated tenants, so exact-origin keying is the conservative choice.
*   ***Proposed Edit:***
    *   Location: line 147 and test list line 185-186; Store description line 115.
    *   Change: "records are keyed by `BrowserOrigin.canonicalString` (exact scheme, host, effective port); no registrable-domain widening, so `www.example.com` and `example.com` are separate records. A request from a subframe is keyed by the requesting frame's origin from `WKFrameInfo.securityOrigin`, not the top-level page, and the bubble shows both when they differ." In tests replace "canonicalization equivalence (www vs apex)" with "records for www and apex stay distinct; default-port and trailing-dot hosts canonicalize to one record."

### [Issue 4] [`consolidated B4`] - Severity: Blocker
*   **Affected Scope:** [Lines 64-70 vs 74-81, 107-112, 161-163, 185](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L64-L185) (Open Questions, CUJ 1, Architecture)
*   **Discrepancy Description:** Agent-owned policy is an open question yet hard-coded everywhere; record lookup precedes the agent deny, so a takeover "allow for site" could grant an agent tab; ownership flip with a pending bubble unspecified.
*   **Rationale:** Contradictory and unsafe ordering.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid, and security-relevant (permissions, approvals). The doc itself leans hard deny, so closing the question matches the doc. Ownership flips at `AppBrowserAutomationCoordinator.userTookOver`, so the boundary is detectable as the doc says.
*   ***Proposed Edit:***
    *   Location: delete Open Question "Agent-owned tabs" (lines 64-70); edit Architecture lines 107-112.
    *   Change: Add a Decision under Technical Plan: "Agent-owned tabs hard-deny every sensitive class with no bubble and no record lookup. Stored allow records are ignored in agent-owned tabs. Ownership is read when the request arrives. If ownership changes while a bubble is pending, the bubble is dismissed and the request is denied. The decision handler is called exactly once on every path." Reorder the diagram: ownership check first, then record lookup, then bubble.

### [Issue 5] [`consolidated B5`] - Severity: Blocker
*   **Affected Scope:** [Lines 125-142](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L125-L142) (Alternatives Considered)
*   **Discrepancy Description:** Alternatives are "Rejected by author", circular, "ITP-style" misnomer; missing real alternatives. There is also no Risks section (see M3).
*   **Rationale:** Alternatives must carry checkable trade-offs.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Existing types worth weighing are verified: `BrowserPermissionRuleStore`, `BrowserPermissionApprovalCoordinator`, `BrowserDialogApprovalView`.
*   ***Proposed Edit:***
    *   Location: replace lines 125-142.
    *   Change (needs author confirmation, trade-offs are my inference from the findings):
        * "Alternative A: do nothing and rely on WebKit and TCC. Pros: no work. Cons: no app record, no audit, microphone requests reach the system prompt from agent tabs. Rejected: fails the agent-tab requirement."
        * "Alternative B: hard-deny everything now, ship no UI. Pros: smallest surface, closes the agent-tab exposure immediately, no bubble stack. Cons: breaks user-driven conferencing and capture. Deferred option: ship this first and add the bubble later."
        * "Alternative C: extend `BrowserPermissionRuleStore` and `BrowserPermissionApprovalCoordinator` instead of new `SitePermissionPolicy` and `SitePermissionStore`. Pros: one origin type, one store, existing approval flow. Cons: the store currently holds only navigation origins; it needs a permission-class dimension."
        * "Alternative D: session-only grants, no persistence. Pros: no stored records, no retention question. Cons: repeated prompts."
        * Selected design: state which of the above is chosen and why. Needs author decision.

### [Issue 6] [`consolidated M1`] - Severity: Major
*   **Affected Scope:** [Lines 107-112, 174-177](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L107-L177) (Architecture, file 4)
*   **Discrepancy Description:** Parallel bubble stack beside the existing pending-dialog pipeline; edge cases unspecified.
*   **Rationale:** Duplicates `BrowserDialogApprovalView` machinery; navigation, close, concurrent requests undefined.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. `BrowserDialogApprovalView.swift` already resolves a pending request by id through a `resolve` closure. Security: every edge case defaults to deny.
*   ***Proposed Edit:***
    *   Location: after Technical Plan prose (line 123); file 4 (line 174).
    *   Change: "Extend the pending-dialog request model with a permission kind and reuse `BrowserDialogApprovalView`'s resolve path instead of a new bubble stack. Edge cases, all resolving to deny: tab navigates or closes while pending; tab moves to background; web content process terminates (`webViewWebContentProcessDidTerminate`, line 932); a second request for the same origin and class while one is pending (coalesce to the first answer); countdown fires. Timeout is a named constant `sitePermissionPromptTimeout`, value 30 seconds (needs author confirmation)."

### [Issue 7] [`consolidated M2`] - Severity: Major
*   **Affected Scope:** [Lines 49, 114-118, 146-151, 164-168, 181](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L49-L181) (Goals, Architecture, NFR, file 2)
*   **Discrepancy Description:** Audit and record storage undefined; "vault" names no mechanism; logger has a closed event shape; "append-only within the session" conflicts with CUJ 2.
*   **Rationale:** Storage and audit path are not buildable as written.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified: `BrowserAutomationDiagnosticEvent` has fixed fields `actionType`, `canonicalOrigin`, `outcome`, `durationMilliseconds` and a decoder that rejects non-canonical origins (`BrowserAutomationLogger.swift` lines 24-60). A permission decision (class, decision, source) does not fit. Which persistence type the "vault" means I did not confirm, so that part is a question for the author.
*   ***Proposed Edit:***
    *   Location: NFR Security bullet (line 148-149) and Store description (line 164-168).
    *   Change: "Audit entries go to the existing browser log through a new event type `BrowserPermissionDiagnosticEvent` carrying class, decision (allow once, allow site, deny, auto-deny), canonical origin, and timestamp. No page content or URL path is logged. The log keeps its existing cap; records in the store are capped at 500 (needs author confirmation). Record deletion writes an audit entry, so 'append-only' means entries are never edited, not that records are permanent. The audit viewer is one surface only: the settings pane." Name the persistence mechanism: needs author decision (profile database vs `AppJSONStore`).

### [Issue 8] [`consolidated M3`] - Severity: Major
*   **Affected Scope:** Missing section (after line 142)
*   **Discrepancy Description:** No Risks and Mitigations section.
*   **Rationale:** Spoofing, prompt flood, redirect mid-prompt, cross-origin iframe, handler leak, stale records, corrupt vault.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Security-relevant; mitigations are all deny-by-default.
*   ***Proposed Edit:***
    *   Location: new `## Risks and Mitigations` after Alternatives.
    *   Change (needs author confirmation):
        * Prompt flood: one pending bubble per tab, further requests deny.
        * Spoofed bubble: the origin line is rendered by the app from `BrowserOrigin.canonicalString`, never from page text.
        * Redirect mid-prompt: navigation dismisses the bubble and denies.
        * Cross-origin iframe: keyed by the requesting frame origin, both origins shown.
        * Decision-handler leak on tab close: a single-shot wrapper calls deny on deinit.
        * Stale allow records: deletable in settings; no automatic expiry in v1.
        * Unreadable or corrupt record store: treat every request as a miss and deny in agent tabs, prompt in user tabs; quarantine the file per `AppJSONStore` behavior.

### [Issue 9] [`consolidated M4`] - Severity: Major
*   **Affected Scope:** [Lines 39-40](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L39-L40) (Goals)
*   **Discrepancy Description:** Harmless set open-ended; alerts and focus are not permission classes; default for unlisted classes unstated.
*   **Rationale:** Classification boundary undefined.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. JS alert/confirm/prompt are handled separately in the engine (lines 866-908). Security: unknown classes should deny.
*   ***Proposed Edit:***
    *   Location: Goal 2 (lines 39-40); Architecture line 104.
    *   Change: "Goal 2: Only the classes in the enforcement table are classified. Any class the delegate delivers that is not listed is denied. JavaScript dialogs are not permission classes and keep their existing handling." Delete 'harmless classes' wording.

### [Issue 10] [`consolidated M5`] - Severity: Major
*   **Affected Scope:** [Lines 182-200](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L182-L200) (Tests, Verification Plan)
*   **Discrepancy Description:** Verification outruns the design: missing tests for countdown, bubble, settings, audit, store round trip; no clock or delegate seam; no exit criterion; mutation checks omit agent deny, timeout deny, unknown-class default.
*   **Rationale:** Test coverage does not map to CUJs.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; `AGENTS.md` requires mutation-checked tests for subtle guards and points to `.claude/docs/verification.md` (exists).
*   ***Proposed Edit:***
    *   Location: item 6 (lines 182-189) and Verification Plan (lines 191-200).
    *   Change: Add tests: injected clock for the timeout; store round trip and corrupt-file path; audit event emitted on allow, deny, timeout, and record delete; `ArtifactWebView` unchanged. Add mutation checks: remove the agent-tab deny, change the timeout default to allow, change the unknown-class default to allow, each asserted to fail a test. Add to verification: "Exit criterion: all of the above pass plus the manual run. Local green is not release evidence; follow `.claude/docs/verification.md`." Add a seam: delegate calls go through a single-shot decision wrapper that tests can drive without WebKit.

### [Issue 11] [`consolidated M6`] - Severity: Major
*   **Affected Scope:** [Line 178](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L178) (file 5)
*   **Discrepancy Description:** `App/SettingsPanes.swift` does not exist.
*   **Rationale:** Wrong target file.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified. `App/` has no such file; `Browser/` has `BrowserSettingsPaneView.swift`, `BrowserSettingsViewModel.swift`, `BrowserSettings.swift`.
*   ***Proposed Edit:***
    *   Location: item 5, line 178.
    *   Change: "`swift/TurboSparkApp/Sources/TurboSparkApp/Browser/BrowserSettingsPaneView.swift` (Modify), with bindings in `BrowserSettingsViewModel.swift` (Modify)." Drop the parenthetical.

### [Issue 12] [`consolidated M7`] - Severity: Major
*   **Affected Scope:** [Lines 1-6, 28-31, 126-142](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L1-L142) (header, The Problem, Alternatives)
*   **Discrepancy Description:** Uncited and undefined background (CNVD, "repeats the exposure", "ZCode chose governance for the same reason", ITP, takeover state, vault, agent-owned tab).
*   **Rationale:** A fresh reader cannot check the claims.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid for human readers too. The ZCode spec link at lines 4-6 is the only citation; I could not verify its contents from the repo.
*   ***Proposed Edit:***
    *   Location: The Problem (lines 28-31); add a short Terms paragraph after Overview.
    *   Change: Remove "ZCode chose governance over blanket deny for the same reason" unless the author cites the ZCode spec section; keep the spec link. Add: "Terms: an agent-owned tab is a tab under agent control (`AppBrowserAutomationCoordinator` holds an ownership token); user takeover clears the token via `userTookOver(tabID:)`. 'Vault' means the per-profile store named in the Store section." Drop "ITP-style". Whether to keep the CNVD reference: needs author confirmation of the source.

### [Issue 13] [`consolidated M8`] - Severity: Major
*   **Affected Scope:** [Lines 41-43, 78-81](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L41-L81) (Goals, CUJ 1)
*   **Discrepancy Description:** Countdown has no duration; four-state set exceeds the threat; "deny once" audit and record behavior unstated.
*   **Rationale:** Under-specified UI contract.
*   ***AI Suggestion:*** PROPOSE FIX (countdown and deny-once parts only)
*   ***Justification:*** The missing duration and deny-once behavior are real gaps. Whether to cut to three states is a product call, so I leave it to the author. Security: deny once must not persist, and timeout must not persist either (a timeout is not a user choice).
*   ***Proposed Edit:***
    *   Location: Goal 3 (lines 41-43).
    *   Change: Append: "Countdown is 30 seconds (needs author confirmation). Deny once and timeout write an audit entry and no record. Allow for site and deny for site write a record." Optional (author decision): collapse to allow once, allow for site, deny, with deny-for-site available from settings.

### [Issue 14] [`consolidated M9`] - Severity: Major
*   **Affected Scope:** [Lines 34-49 vs 97-118](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L34-L118) (Goals, Architecture)
*   **Discrepancy Description:** No requirements-to-design mapping; relation of new Policy and Store to the existing rule store and coordinator unclear.
*   **Rationale:** Reader cannot trace a Goal to a component.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Follows from the Issue 5 decision.
*   ***Proposed Edit:***
    *   Location: after the diagram (line 123).
    *   Change: Add a 4-row table: Goal 1 -> `WebKitBrowserEngine` delegate plus `SitePermissionPolicy`; Goal 3 -> pending-dialog model plus bubble; Goal 5 -> `SitePermissionStore` and settings pane; Goal 6 -> browser log event. Add one sentence: "`SitePermissionStore` is separate from `BrowserPermissionRuleStore` (navigation allowlist) and shares only the `BrowserOrigin` type" or the opposite, per the Alternative C decision (needs author decision).

### [Issue 15] [`consolidated N1`] - Severity: Nit
*   **Affected Scope:** [Lines 153, 197](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L153) (NFR, Verification)
*   **Discrepancy Description:** "21 languages" vs "ten localization parity gates".
*   **Rationale:** Apparent inconsistency.
*   ***AI Suggestion:*** CONTEST
*   ***Justification:*** `swift/docs/SWIFT_LOCALIZATION.md` line 37: "`Localizable.xcstrings` carries 1,265 keys in 21 languages". `AGENTS.md` line 140 and `swift/AGENTS.md` line 62 say "ten parity gates", which are test gates, not languages. The two numbers measure different things; the doc is correct.
*   ***Proposed Edit:***
    *   Location: none.
    *   Change: Optional wording for clarity: "strings added to the catalog in all 21 languages, passing the ten parity gates."

### [Issue 16] [`consolidated N2`] - Severity: Nit
*   **Affected Scope:** [Lines 21, 100-105](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L21-L105) (Overview, Architecture)
*   **Discrepancy Description:** "Explicitly empty" and "filled in" overstate.
*   **Rationale:** Alert/confirm/prompt delegates already exist.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified: the `WKUIDelegate` extension is empty (`WebKitBrowserEngine.swift` line 1492), but JS dialog methods live in the class body (lines 866-908). The doc's claim is about permission handling and is accurate; the wording "filled in" and line 171 "currently empty conformances" should say "add permission methods".
*   ***Proposed Edit:***
    *   Location: line 20-21 and line 171-172.
    *   Change: "the engine implements no `WKUIDelegate` permission method" and "add the permission methods to the `WKUIDelegate` extension".

### [Issue 17] [`consolidated N3`] - Severity: Nit
*   **Affected Scope:** [Lines 159-175](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L159-L175)
*   **Discrepancy Description:** Policy and store could be one type.
*   **Rationale:** Simplicity.
*   ***AI Suggestion:*** DECLINE
*   ***Justification:*** Nit, and the split follows from Alternative C, which the author has to decide first. Recorded, not fixed.
*   ***Proposed Edit:***
    *   Location: none.
    *   Change: none.

### [Issue 18] [`consolidated N4`] - Severity: Nit
*   **Affected Scope:** [Lines 52-56](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L52-L56) (Non-Goals)
*   **Discrepancy Description:** "Navigation unchanged" reads contradictory with new agent-tab permission deny.
*   **Rationale:** Scope ambiguity.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Cheap clarity fix.
*   ***Proposed Edit:***
    *   Location: Non-Goal 1, line 53.
    *   Change: "No change to agent-tab navigation policy or egress isolation. Agent tabs do gain a permission deny rule (Decision in Technical Plan); that rule does not affect which origins they may navigate to."

### [Issue 19] [`consolidated N5`] - Severity: Nit
*   **Affected Scope:** Header (lines 1-10)
*   **Discrepancy Description:** No ladder position; dependencies and ZP-15 overlap not in INDEX.
*   **Rationale:** Readers cannot place this doc.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Partly wrong: `INDEX.md` line 98 already places ZP-14 in rung 2. Dependencies are missing.
*   ***Proposed Edit:***
    *   Location: after line 10.
    *   Change: "* Depends on: `BrowserOrigin`, the browser log, the per-profile store. Related: ZP-15 (browser recording) also touches `WebKitBrowserEngine`; recording of capture permissions interacts with media capture here (needs author confirmation)."

### [Issue 20] [`consolidated N6`] - Severity: Nit
*   **Affected Scope:** [Lines 99-120](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L99-L120)
*   **Discrepancy Description:** Diagram mixes pseudocode; classification table never shown; line 120 repeats lines 107-109.
*   **Rationale:** Redundancy.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Covered by the Issue 1 table. Delete line 120's first sentence.
*   ***Proposed Edit:***
    *   Location: line 120.
    *   Change: delete "Classification: sensitive classes default-deny; harmless pass through." Keep the second sentence about the single decision-handler call.

### [Issue 21] [`consolidated N7`] - Severity: Nit
*   **Affected Scope:** [Lines 155-189](../../../designs/zcode-parity/ZP-14-browser-page-permissions.md#L155-L189)
*   **Discrepancy Description:** File list lacks ordering; item 5 parenthetical garbled; item 6 label style inconsistent.
*   **Rationale:** Formatting.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Item 5 is replaced in Issue 11. Rest is cosmetic.
*   ***Proposed Edit:***
    *   Location: items 5 and 6.
    *   Change: item 5 per Issue 11; item 6 "Mutation checks" as a top-level sub-bullet matching other items. Add "(build order: 2, 1, 3, 4, 5, 6)" after the heading.

### [Issue 22] [`consolidated N8`] - Severity: Nit
*   **Affected Scope:** Lines 100, 159-181
*   **Discrepancy Description:** `Browser/` vs `BrowserAutomation/` placement unconfirmed; no full paths.
*   **Rationale:** Placement not justified.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Both directories exist (`Browser/` holds pane and settings UI, `BrowserAutomation/` holds the engine, rule store, coordinator). Policy and store sit with the rule store in `BrowserAutomation/`, UI in `Browser/`.
*   ***Proposed Edit:***
    *   Location: items 1 and 2 (lines 159, 164).
    *   Change: move `SitePermissionPolicy.swift` and `SitePermissionStore.swift` to `BrowserAutomation/`, keep `SitePermissionBubbleView.swift` in `Browser/`. Needs author confirmation.

### [Issue 23] [`consolidated N9`] - Severity: Nit
*   **Affected Scope:** Lines 20-21, 38; 29-30
*   **Discrepancy Description:** "device and sensitive permissions", "device access requests", "device classes" used interchangeably; rhetorical three-item list.
*   **Rationale:** Term drift.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Resolved by Issue 1's single class table; use "permission class" throughout.
*   ***Proposed Edit:***
    *   Location: lines 16-20, 36-38.
    *   Change: replace with "permission class" and refer to the table. Trim the list at lines 29-30 to the two classes the baseline paragraph names.

### [Issue 24] [`consolidated N10`] - Severity: Nit
*   **Affected Scope:** Lines 62-69
*   **Discrepancy Description:** "Lean" used as a decision rule.
*   **Rationale:** No criterion.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Record-scope question remains; Issue 4 closes the agent-tab one.
*   ***Proposed Edit:***
    *   Location: Open Question 1, lines 61-63.
    *   Change: "Record storage scope: per profile. Criterion: a profile is the existing unit of isolation for sensitive state, so a grant in one profile must not apply in another." Needs author confirmation.

### [Issue 25] [`consolidated N11`] - Severity: Nit
*   **Affected Scope:** Lines 152, 176
*   **Discrepancy Description:** "Bubble states are announced" lacks a mechanism; "DESIGN.md compliant" names no rule.
*   **Rationale:** Unverifiable accessibility claim.
*   ***AI Suggestion:*** ESCALATE
*   ***Justification:*** The correct mechanism depends on `DESIGN.md` and `docs/design-system/` rules that govern accessibility announcements. I did not read them in this pass. The author, who will implement the bubble, should name the specific rules. Not fixed here.
*   ***Proposed Edit:***
    *   Location: n/a
    *   Change: n/a (author to name the DESIGN.md sections and the announcement API).

### [Issue 26] [`consolidated N12`] - Severity: Nit
*   **Affected Scope:** Lines 41-45, 52-57
*   **Discrepancy Description:** No measurable targets; Non-Goals omit TCC prompt, file chooser, downloads, policy export; no rollout or compatibility note.
*   **Rationale:** Scope completeness.
*   ***AI Suggestion:*** DECLINE (measurable targets, policy export, rollout); PROPOSE FIX for the Non-Goals list
*   ***Justification:*** A design this size does not need performance targets or policy export. The system TCC prompt and file chooser omission is a real scope boundary: the microphone TCC prompt remains an OS-level decision this design does not replace. Security: say so explicitly.
*   ***Proposed Edit:***
    *   Location: Non-Goals, after line 57.
    *   Change: "* No change to the system TCC prompt, the file chooser, or downloads. A microphone grant at the OS level still applies; this design adds an app-level deny in front of it."

## Applied

* Applied (22): B1-B5, M1-M9, N2, N4, N5, N6, N7, N8, N9, N10. M8 applied for the countdown and deny-once parts only; N12's Non-Goals edit applied.
* Unapplied: N1 (CONTEST), N11 (ESCALATE), N3 (DECLINE), N12 measurable-targets part (DECLINE), M8 three-state collapse (author decision).
* Notes: Goal numbers shifted after the picker goal was removed, so the requirements table uses Goals 1, 3, 4, 5. Several "needs author confirmation" and "needs author decision" markers remain in the doc.
