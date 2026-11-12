# Stage 8 intermediate zoom-out report: ZP-14 browser page permissions

Reviewer: egm-zoom-out (S8). Completed. Verdict: not ready. The draft has no additions or mitigations since Stages 5-7: every cited line number still matches and all prior Blocker and Major findings remain open. Line numbers refer to ZP-14.

Code checks redone: `App/SettingsPanes.swift` does not exist (`Browser/BrowserSettingsPaneView.swift` does). `BrowserPermissionRuleStore.swift` and `Browser/BrowserDialogApprovalView.swift` exist and are not reused. No registrable-domain or public-suffix logic exists in browser code, so the "existing origin rules" claim at lines 146-149 is false.

## Open Blockers

S8-B1. Lines 16-17, 36-38, 44-45, 89-95, 100-104. Goals promise classes WebKit does not expose. Only media capture (and orientation/motion) has a delegate hook. `requestDevicePermissionFor` unconfirmed. No hook for geolocation, notifications, clipboard-read, device classes; no app screen-share picker. Fix: per-class "how enforced" column; cut Goals and CUJ 3; spike a baseline (entitlements, usage strings, OS behavior); lines 20-22 unverified.

S8-B2. Lines 146-149, 185-186. Origin key wrong; use `BrowserOrigin.canonicalString`; drop www/apex test.

S8-B3. Lines 64-70 vs CUJ 1, 107-112, 161-163. Agent-owned policy open yet hard-coded; record lookup precedes the agent deny. Fix: decide; check ownership first; ignore allow records for agent tabs; auto-deny pending bubbles on ownership flip.

S8-B4. Lines 125-142. Alternatives circular; missing: extend `BrowserPermissionRuleStore` and `BrowserPermissionApprovalCoordinator`, reuse `BrowserDialogApprovalView`, hard-deny now and defer UI, session-only grants.

## Open Majors

S8-M5. Lines 107-112, 174-177. Parallel dialog pipeline; extend existing pending-dialog flow (`handleJavaScriptDialog`, `resolvePendingDialog`, `cancelPendingDialog`, `dismissPendingDialogs`). Edge cases unspecified: navigation or close mid-bubble, simultaneous requests, subframe origin, process crash.

S8-M6. Lines 49, 114-118, 146-151, 164-168, 181. Storage and audit undefined (ProfileDatabase vs AppJSONStore; `BrowserAutomationLogger` closed event shape); session-only audit conflicts with CUJ 2; two viewing surfaces; no retention or cap.

S8-M7. Missing Risks and Mitigations section; unknown-class default; closed harmless list (lines 39-40; alerts and focus are not permission classes).

S8-M8. Line 178. Wrong settings file; retarget to `BrowserSettingsPaneView.swift`, `BrowserSettingsViewModel.swift`, `BrowserSettings.swift`.

S8-M9. Lines 182-200. Verification outruns design: no tests for countdown, audit, settings, vault round trip, `ArtifactWebView` regression; no test seam or injected clock; no exit criterion or scheme; add `.claude/docs/verification.md` and "local green is not release evidence".

S8-M10. Undefined terms and uncited claims: CNVD, "ITP-style prompts" (ITP is cookie tracking prevention), takeover state, agent-owned tab, vault, log store; countdown duration; "allow for site" for screen share; lines 5-6, 30-31 uncited.

## Nits

S8-N1. Lines 153, 197. "21 languages" vs INDEX "ten" gates.
S8-N2. Lines 100-105. Diagram says "filled in" but alert/confirm/prompt delegates already exist.
S8-N3. Lines 41-43. Four states plus countdown exceed the threat; minimal: deny default, allow once, allow for site, revoke in settings.
S8-N4. Lines 159-175. Policy and store could be one type; single-handler guarantee as a single-shot wrapper.
S8-N5. Lines 187-189. Mutation checks omit agent-owned deny, timeout deny, unknown-class default, vault-locked.
S8-N6. Lines 52-56. Clarify navigation unchanged, permissions new.
S8-N7. No statement of ladder position (adds no agent-facing tool); ZP-15 overlap unnoted.
S8-N8. INDEX line 98 "medium" sizing credible only after the B1 scope cut.

## Scope-adding asks to reject

Public suffix list or registrable-domain keying; screen-share and device pickers (ScreenCaptureKit, Screen Recording TCC); prompt rate limiting, policy export, app-wide sync, master toggle UI; a new state-machine framework; per-class record granularity beyond what the delegate enforces.

## Recommended minimal shape

1. Enforce only what the delegate exposes (media capture, orientation if confirmed); state the rest per class.
2. Hard-deny in agent-owned tabs; bubble (allow once, allow for site, deny) only in takeover tabs, extending the pending-dialog and approval-view path.
3. Extend `BrowserPermissionRuleStore` keyed by `BrowserOrigin.canonicalString`; one named store; settings list with delete in the existing Browser pane.
4. One persisted audit sink with a cap, viewed in one place.
