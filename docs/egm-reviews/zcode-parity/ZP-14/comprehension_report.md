# Stage 6 comprehension report: ZP-14 browser page permissions

Reviewer: egm-comprehension (CO). Read the document alone. Completed.

## What this proposes

Add a default-deny permission layer to the app's embedded browser. When a web page asks for a sensitive capability (camera, microphone, location, notifications, clipboard read, screen share, or device classes like USB or Bluetooth), the app denies it by default. A user-taken-over tab shows a confirmation bubble with four choices (allow once, allow for site, deny once, deny for site) and a countdown that defaults to deny. Agent-owned tabs hard-deny silently, though this is only a "lean" in the Open Questions. Decisions are saved per site in the profile vault, are manageable in a settings page, and are written to an audit log. Three new Swift files and edits to two existing ones implement it, with unit tests.

## How the existing system works, per this document

The browser engine (`WebKitBrowserEngine`) has a `WKUIDelegate` whose permission handling is empty, so WebKit's own default applies, with no app-level consent, record, or audit. There is an existing per-origin navigation policy with egress isolation for agent tabs, an agent allowlist with origin rules (scheme plus registrable domain), a "user takeover before direct input" state, a profile vault, a "browser log store", and a Browser settings pane. `ArtifactWebView` is already deny-by-default. The document does not say how any of these work beyond one-line mentions.

## Findings

CO-1 Blocker. Lines 64-70 vs 74-81, 107-111, 161-163, 199. The agent-tab policy is undecided ("Decide at review") but every later section treats it as settled. Alternative A's con ("agent tabs would still reach system prompts") never says what the chosen design does. Fix: record the decision and remove the "lean" wording.

CO-2 Major. Lines 41-43, 79, 110, 69. Bubbles appear only in user-taken-over tabs but anchoring and delivery are unstated; "user takeover before direct input" is never defined; effect of takeover ending mid-request unknown. Fix: define the state and the mid-request outcome.

CO-3 Major. Lines 37-38, 101-105, 28. No classification table. No mapping from clipboard-read, notifications, MIDI/HID/serial to a WebKit callback; only media-capture and device-permission are named. "Silence, depending on the request class" never specified. Fix: add the table with the interception mechanism per class.

CO-4 Major. Lines 61-63, 111-113, 147-149, 186. Records keyed by (profile, origin, class) but origin described as "scheme plus registrable domain", which grants all subdomains; tests assert www/apex equivalence; "existing origin rules" not defined or linked. Fix: define origin vs site; use exact origin.

CO-5 Major. Lines 28, 126-133, 56, 150, 6, 30. "ITP-style prompts", "CNVD", and ZCode Finding 6.9 are not explained, so the parity target cannot be judged. Fix: one-line definitions and a summary of what ZCode does.

CO-6 Major. Lines 89-95, 44-45. Meaning of "allow for site" for screen share is unexplained; "device picker where WebKit exposes choices" is vague; no behavior when WebKit exposes none. Fix: state the four-state mapping to the picker flow.

CO-7 Major. Lines 119-123, 148-149. Concurrency not covered: simultaneous requests, repeated request while bubble open, tab close or navigation mid-bubble, app quit. "Append-only within the session" ambiguous on persistence; CUJ 2 says deletions are audited but no retention or clearing. Fix: add a short edge-case list and state retention.

CO-8 Major. Lines 18, 49-50, 181. "Deny once" audit and record behavior unstated; "pane's info surface" undefined and conflicts with the Settings viewer in item 5. Fix: pick one surface; state whether deny once creates a record.

CO-9 Nit. Lines 61, 112, 150, 117, 49, 153, 193-195. Undefined project terms: vault, browser log store, agent-owned tab, pane, DESIGN.md compliant, make targets. Fix: define or link on first use.

CO-10 Nit. Lines 152-153, 197, 178. "21 languages" assumes known catalog; countdown length unstated; microphone only implied under "media capture"; "basename" wording confusing. Fix: state values and full paths.

CO-11 Nit. Lines 52-56. Non-goal "agent-tab navigation policy unchanged" reads as contradicting the new agent-tab deny behavior. Fix: say explicitly that navigation is unchanged and permissions are new.
