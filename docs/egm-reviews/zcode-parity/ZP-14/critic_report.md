# Stage 7 critic report: ZP-14 browser page permissions

Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY). All completed. One round only, no remediation, convergence not assessed. Standard: feature-spec-standard for design and structural. Line numbers refer to ZP-14.

## Design (DC)

DC-1 Blocker. Lines 16-18, 36-37, 101-102, 120-122. Several classes cannot be intercepted via WKUIDelegate: only media capture (and possibly orientation) has a callback. `requestDevicePermissionFor` unconfirmed. No hook for geolocation (CoreLocation), clipboard-read, notifications; MIDI/HID/USB/serial/bluetooth not exposed to WKWebView. Problem section's per-class baseline is unverified. Fix: add a capability matrix (class, WKWebView behavior, mechanism) and cut Goals to what is enforceable.

DC-2 Blocker. Lines 63-70, 77-81, 185. Agent-owned vs takeover is "open" yet hard-coded in CUJs, policy and tests. Ownership flip with a pending bubble or stored record unspecified; takeover handling already dismisses pending dialogs (`WebKitBrowserEngine.swift` lines 448-450). Fix: resolve now; ignore allow records for agent tabs; handler still called exactly once.

DC-3 Blocker. Lines 125-142. Alternatives fail the Alternatives Rule: "Rejected by author", circular reasoning; missing options: extend `BrowserPermissionRuleStore` and `BrowserPermissionApprovalCoordinator`, reuse `BrowserDialogApprovalView`, session-only grants, app-wide storage, master toggle. Fix: concrete trade-offs and the missing options.

DC-4 Major. Lines 22-23, 99-118. New `SitePermissionPolicy` and `SitePermissionStore` not reconciled with existing `BrowserPermissionRuleStore`, `BrowserPermissionOwner`, `BrowserOrigin`, `BrowserPermissionApprovalCoordinator`. Fix: add relation section; justify new types or fold in.

DC-5 Major. Lines 146-149, 186. Cited "existing origin rules" do not exist: `BrowserOrigin` is scheme, host, effective port. Registrable domain widens grants to siblings and shared-hosting tenants. Fix: key on exact origin; state top-level vs initiating-frame handling.

DC-6 Major. Lines 49, 147-149. Audit log schema, persistence, retention, redaction unspecified; "within the session" conflicts with CUJ 2; origins may leak via support bundles. Fix: define schema, retention cap, redaction, storage.

DC-7 Major. Lines 41-43, 99-122. Bubble lifecycle: tab destroyed or navigated mid-bubble, concurrent and duplicate requests, prompt spam, countdown duration and backgrounding, interaction with macOS TCC; no camera or microphone usage strings or entitlements found. Fix: add a state machine and TCC/entitlement requirements.

DC-8 Major. Lines 44-45, 89-95. Screen share source picker has no mechanism (WKWebView gives media type only; would need own ScreenCaptureKit picker and Screen Recording TCC). Fix: specify or move to follow-up.

DC-9 Major. Lines 59-70, 144-153. No Risks and Mitigations section; missing: UI spoofing, prompt fatigue, stale allow records, locked or corrupt vault (default deny?), record versioning, unknown-class default, limits. Fix: add the section.

DC-10 Nit. Line 98, INDEX line 98. Dependencies on vault, log store, rule store not listed.

DC-11 Nit. Lines 128-133. "ITP-style prompts" misnomer; ITP is cookie tracking prevention.

DC-12 Nit. Lines 178-181. `SettingsPanes.swift` likely wrong; Browser pane is `Browser/BrowserSettingsPaneView.swift` and `BrowserSettingsViewModel.swift`.

DC-13 Nit. Lines 152-153, 197. "21 languages" vs INDEX "ten localization parity gates".

DC-14 Nit. Lines 187-189. Mutation checks miss agent-owned deny, timeout deny, unknown-class default, vault-locked case.

DC-15 Nit. Lines 1-6, 5-6, 30-31. CNVD origin story and ZCode spec link unverified.

## Structural (ST)

No Blocker. Heading hierarchy valid; one code fence closed; links well-formed.

ST-1 Major. Lines 125-142. Alternatives: unlinked "Rejected by author", thin reasons; Alternative A is a straw man. Fix: concrete reasons.

ST-2 Major. Lines 125-142. Only two alternatives; contestable choices (agent hard deny vs bubble, per-profile vs app-wide, system TCC prompt, per-class records) have none.

ST-3 Major. Missing section. Risks and Mitigations absent: bubble spoofing, prompt flood, background-tab bubble, mid-prompt redirect, cross-origin iframe, handler leak on tab close, unbounded audit and record growth.

ST-4 Major. Lines 146-149, 186. Registrable-domain key conflicts with web permission norms; ambiguous with "origin-canonicalized". Audit "append-only within the session" conflicts with CUJ 2 and viewer.

ST-5 Major. Lines 59-70 vs 78-81, 110-111, 165-168. Open questions treated as decided elsewhere; storage scope pre-answered at line 150.

ST-6 Major. Lines 34-49 vs 97-118. No requirements-to-design mapping; Goals for notifications, clipboard, geolocation, device classes have no delegate method.

ST-7 Major. Lines 182-200. Tests and manual plan skip countdown, bubble UI, audit log, settings page, pickers; no exit criterion; no `ArtifactWebView` regression check.

ST-8 Nit. Lines 41-45, 52-57. No measurable targets (countdown value); Non-Goals omit TCC prompt, file chooser, downloads, policy export.

ST-9 Nit. Lines 4-10. No References section; no ZP-15 note (screen capture overlap); no CNVD source.

ST-10 Nit. Lines 3-6. CNVD claim unlinked.

ST-11 Nit. Lines 99-118. Diagram mixes pseudocode and `...`; classification table never shown.

ST-12 Nit. Lines 155-189. No ordering or dependencies; item 5 parenthetical garbled; item 6 label style inconsistent.

ST-13 Nit. Lines 152-153, 197. "21 languages" vs "ten" gates.

ST-14 Nit. Line 194. Verification omits `.claude/docs/verification.md` and "local green is not release evidence"; no rollout or flag or compatibility note.

## Style (SY)

SY-1 Blocker. Lines 16-22, 26-29, 120-123, 169-172. Existing behavior stated as fact in Overview but hedged in Problem; no per-class verified baseline; "silence" case not observed. Fix: state verified behavior per class or label unverified.

SY-2 Major. Lines 5-6, 30-31. "Added after a CNVD finding" and "repeats the exposure" uncited; "real" is filler.

SY-3 Major. Lines 39-40. "for example simple alerts, basic focus" leaves the harmless set open; state a closed list and the default for unlisted classes.

SY-4 Major. Lines 36-38, 100-104. Goals read as covered but only two callbacks named.

SY-5 Major. Lines 41-43 vs 78-81, 110-111, 64-70. Bubble vs hard deny contradiction; question open yet assumed.

SY-6 Major. Lines 46-48, 61-63, 146-151, 161-166. Storage scope open yet stated as fact; registrable-domain merging subdomains with no rationale.

SY-7 Major. Lines 117, 148-149. "Append-only within the session" and "existing browser log store" unnamed, no retention.

SY-8 Major. Lines 41-43, 77-81, 198-200. Countdown has no duration.

SY-9 Major. Lines 89-95, 44-45. Goals vs CUJ 3 on screen share picker timing; "allow for site" for screen share unstated; no fallback when no device choices.

SY-10 Major. Lines 127-133. "ITP-style prompts" inaccurate; "Rejected by author" circular.

SY-11 Major. Lines 135-142. "ZCode chose governance over blanket deny for the same reason" uncited.

SY-12 Major. Lines 152-153, 197. 21 languages vs INDEX ten gates.

SY-13 Major. Lines 182-200. Verification outruns plan: no tests for bubble UI, settings, audit, vault round trip, pickers; no scheme name for build.

SY-14 Major. Lines 53-58, 69-70. No cited type for takeover state; mid-request takeover behavior unstated.

SY-15 Nit. Lines 20-21, 38. "Device and sensitive permissions", "device access requests", "device classes" used interchangeably.

SY-16 Nit. Lines 29-30. Three-item rhetorical list repeats Overview.

SY-17 Nit. Lines 62-69. "Lean" stated as decision rule; give the criterion.

SY-18 Nit. Line 120. Classification sentence repeats lines 107-109.

SY-19 Nit. Lines 152, 176. "Bubble states are announced" lacks the mechanism; "DESIGN.md compliant" names no rule.

SY-20 Nit. Lines 8-10, 204. Header lacks owner for open questions (skip unless required).

SY-21 Nit. Lines 160-182. `Browser/` vs `BrowserAutomation/` folder split; confirm.
