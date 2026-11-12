# Stage 5 zoom-out report: ZP-14 browser page permissions

Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed. Target: full file (204 lines). Line numbers refer to ZP-14.

## Zoom-out (ZO)

ZO-1 Blocker. Lines 146-149, 185-186. Origin key "scheme plus registrable domain per the existing origin rules" is wrong: `BrowserOrigin` in `BrowserAutomation/BrowserPermissionRuleStore.swift` is exact scheme + host + effective port, no registrable-domain logic. Registrable domain needs a public suffix list (new dependency) and one grant would cover sibling subdomains such as `*.github.io`. Fix: key by `BrowserOrigin.canonicalString`; drop the www/apex test.

ZO-2 Blocker. Lines 36-38, 44-45, 89-95, 100-104. Design promises classes WebKit does not expose. `requestDevicePermissionFor` is unconfirmed. WKUIDelegate has `requestMediaCapturePermissionFor` and `requestDeviceOrientationAndMotionPermissionFor` only. No hooks for geolocation (CoreLocation/entitlement), notifications, clipboard-read, WebUSB/HID/Serial/Bluetooth. Screen share has no app-supplied source picker, so CUJ 3 is not buildable. No camera, microphone or location usage strings or entitlements were found, so the OS may already block capture; real current exposure is not stated. Fix: split Goals into delegate-enforced (media capture, orientation) and other mechanisms; defer pickers.

ZO-3 Major. Lines 107-112 vs 64-70, 78-80. Policy flow looks up a record before the agent-owned deny, so an "allow for site" record made during takeover could grant an agent-owned tab. Fix: check ownership first; resolve or mark provisional the open question.

ZO-4 Major. Line 178. `App/SettingsPanes.swift` does not exist. Browser settings live in `Browser/BrowserSettingsPaneView.swift`, `BrowserSettingsViewModel.swift`, `BrowserSettings.swift`. Fix: retarget item 5.

ZO-5 Major. Lines 49, 117-118, 148-149, 165-168, 181. `BrowserAutomationLogger` has a closed event shape (`actionType`, `canonicalOrigin`, `outcome`, `durationMilliseconds`) with no permission class or decision. "Append-only within the session" conflicts with persistence. Viewing surface differs (line 49 info surface vs line 181 Settings). Fix: add a case or a separate sink; pick one surface; state persistence.

ZO-6 Major. Lines 107-112, 174-177. The engine already has a pending-dialog pipeline (`handleJavaScriptDialog`, `resolvePendingDialog`, `cancelPendingDialog`, `dismissPendingDialogs`, wired in `BrowserPaneView.swift` near line 468). Design builds a parallel bubble stack. Unspecified: navigation or tab close with bubble open, simultaneous requests, subframe origin, web content process crash. Fix: extend the pending-dialog model with a permission kind.

ZO-7 Major. Lines 114-118, 164-168. "Records in the vault" names no mechanism (SQLCipher `ProfileDatabase` vs `AppJSONStore`) and no migration. Fix: pick one.

ZO-8 Nit. Lines 39-40. Alerts and focus are not permission classes; alert delegates already exist at about lines 864-930 of `WebKitBrowserEngine.swift`.

ZO-9 Nit. Line 21 and lines 100-105. The extension at line 1492 is empty but alert/confirm/prompt delegates exist elsewhere; diagram mislabels them as "filled in".

ZO-10 Nit. Lines 41-43. Four states plus countdown is more than the threat needs; timeout length unspecified. Minimal version: deny default, allow once, allow for site, revoke in settings.

ZO-11 Nit. Lines 159-175. Policy and store could be one type for a first cut; single-handler guarantee is implementation detail.

ZO-12 Nit. Lines 182-190. No seam for delegate testing; timeout test needs an injected clock.

ZO-13 Nit. Lines 153, 197. "21 languages" vs INDEX "ten localization parity gates".

## Inventive catalyst (IC)

IC-1 Major. Lines 16-17, 36-37, 41-46, 100-105, 174-177. Copies ZCode's Chromium-shaped surface. Same API-gap finding as ZO-2. Fix: per-class "how enforced" column (delegate, absent API, OS layer); one-day spike listing what WKWebView calls into; drop screen-share and device-picker CUJs.

IC-2 Major. Lines 20-22, 25-32. Default behavior misdescribed ("built-in prompting or silence" unverified). With no delegate macOS 12+ shows a system prompt, and entitlements may be absent so exposure may be near zero. Fix: measure baseline per class (entitlements, usage strings, observed behavior) in the Problem section.

IC-3 Major. Lines 64-70, 78-81, 107-112, 159-168. Agent-owned hard deny is assumed everywhere yet listed as open. Unhandled: ownership flipping back to agent with a live bubble or record. Fix: state the invariant: persisted allows apply only in user-owned tabs, ownership re-evaluated at decision time, pending bubbles auto-deny on flip.

IC-4 Major. Lines 61-63, 113-118, 146-151, 164-168. Vault vs log store undefined; session-only audit contradicts CUJ 2 and the viewer; registrable-domain key unsafe. Fix: key by full origin; persist audit in profile DB with cap; name the table.

IC-5 Nit. Whole doc. No statement that it adds no agent-facing tool (ladder position); ZP-15 and ZP-19 ordering not in INDEX.

IC-6 Nit. Lines 41-43, 78-81, 174-177. Timeout constant, background-tab behavior (deny immediately), bubble anchoring unstated.

IC-7 Nit. Lines 120-123, 187-189. Enforce exactly-once by a single-shot wrapper type so the mutation test needs no WKWebView.

IC-8 Nit. Lines 135-142. Alternative B thin; a middle option is hard-deny now and defer bubble, store, settings. State why records and settings are needed on day one.

IC-9 Nit. Lines 100, 159-181. Confirm `Browser/` directory placement versus `BrowserAutomation/`; settings file cited by basename.
