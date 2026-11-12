# Consolidated review report: ZP-14 browser page permissions

Target: docs/designs/zcode-parity/ZP-14-browser-page-permissions.md (full file, 204 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed. The first attempt at the Stage 5 inventive catalyst hit the concurrent subagent limit and was retried once successfully.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 5, Major 9, Nit 12.

## Blocker

B1. Lines 16-18, 36-38, 44-45, 89-95, 100-104. Goals promise classes WKWebView does not expose: only media capture (and orientation/motion) has a delegate hook; `requestDevicePermissionFor` is unconfirmed; no hooks for geolocation, notifications, clipboard-read, device classes; no app screen-share source picker (CUJ 3 unbuildable). Fix: add a per-class "how enforced" table (delegate, absent API, OS layer), cut Goals and CUJ 3 to what is enforceable, defer pickers. [ZO-2, IC-1, CO-3, CO-6, DC-1, DC-8, ST-6, SY-4, SY-9, S8-B1]

B2. Lines 20-32, 120-123, 169-172. Baseline behavior unverified and stated inconsistently ("silence, depending on class"); no camera, microphone, location usage strings or entitlements found, so exposure may be near zero. Fix: measure and record current behavior per class (entitlements, usage strings, test page) or label unverified. [ZO-2, IC-2, DC-1, SY-1, S8-B1]

B3. Lines 146-149, 185-186. Origin key "scheme plus registrable domain per existing origin rules" is false: `BrowserOrigin` is exact scheme, host, effective port; registrable domain needs a public suffix list and widens grants to sibling subdomains and shared-hosting tenants. Fix: key by `BrowserOrigin.canonicalString`, drop the www/apex test, state frame-origin handling. [ZO-1, IC-4, CO-4, DC-5, ST-4, SY-6, S8-B2]

B4. Lines 64-70 vs 74-81, 107-112, 161-163, 185. Agent-owned policy is an open question yet hard-coded everywhere; record lookup precedes the agent deny, so a takeover "allow for site" could grant an agent tab; ownership flip with pending bubble unspecified. Fix: decide now, check ownership before lookup, ignore allow records for agent tabs, auto-deny pending bubbles on flip, call the handler exactly once. [ZO-3, IC-3, CO-1, DC-2, ST-5, SY-5, S8-B3]

B5. Lines 125-142. Alternatives fail the rule: "Rejected by author", circular, ITP misnomer; missing: extend `BrowserPermissionRuleStore` and `BrowserPermissionApprovalCoordinator`, reuse `BrowserDialogApprovalView`, hard-deny now and defer UI, session-only grants, app-wide storage. Fix: concrete trade-offs and the missing options. [DC-3, DC-4, ST-1, ST-2, SY-10, SY-11, IC-8, S8-B4]

## Major

M1. Lines 107-112, 174-177. Builds a parallel bubble stack beside the existing pending-dialog pipeline; edge cases unspecified (navigation or close mid-bubble, simultaneous or repeat requests, subframe origin, process crash, countdown value and backgrounding). Fix: extend the pending-dialog model with a permission kind; add a short edge-case list and a named timeout constant; background tab denies immediately. [ZO-6, DC-4, DC-7, CO-2, CO-7, SY-8, IC-6, S8-M5]

M2. Lines 49, 114-118, 146-151, 164-168, 181. Audit and record storage undefined: "vault" names no mechanism (ProfileDatabase vs AppJSONStore), `BrowserAutomationLogger` has a closed event shape, "append-only within the session" conflicts with CUJ 2 and the viewer, two viewing surfaces, no retention, cap, or redaction. Fix: name one store, add event case or separate sink, one viewing surface, cap and redaction. [ZO-5, ZO-7, IC-4, CO-7, CO-8, DC-6, SY-7, S8-M6]

M3. Missing section. No Risks and Mitigations: spoofing, prompt flood, background-tab bubble, redirect mid-prompt, cross-origin iframe, handler leak on tab close, stale allow records, locked or corrupt vault, unbounded growth. Fix: add a short section. [ST-3, DC-9, S8-M7]

M4. Lines 39-40. Harmless set is open-ended and alerts and focus are not permission classes; default for unlisted classes not stated. Fix: closed list; unknown class denies. [SY-3, ZO-8, S8-M7]

M5. Lines 7-10 of review context, 182-200. Verification outruns the design: no tests for countdown, bubble, settings, audit, vault round trip, pickers, `ArtifactWebView` regression; no injected clock or delegate seam; no exit criterion or scheme; mutation checks omit agent deny, timeout deny, unknown-class default; no `.claude/docs/verification.md` or "local green is not release evidence". Fix: add a check per CUJ and the missing mutations; single-shot handler wrapper. [ZO-12, IC-7, ST-7, SY-13, DC-14, ST-14, S8-M9]

M6. Line 178. `App/SettingsPanes.swift` does not exist. Fix: retarget to `Browser/BrowserSettingsPaneView.swift`, `BrowserSettingsViewModel.swift`, `BrowserSettings.swift`. [ZO-4, DC-12, S8-M8]

M7. Lines 1-6, 28-31, 126-142. Uncited and undefined background: CNVD origin, "repeats the exposure", "ZCode chose governance for the same reason", ITP, takeover state, vault, log store, agent-owned tab. Fix: cite or drop; define terms; link or cite the type exposing takeover. [SY-2, SY-10, SY-11, SY-14, CO-5, CO-9, ST-10, DC-15, S8-M10]

M8. Lines 41-43, 78-81. Countdown has no duration, and the state set (four states) exceeds the threat; "deny once" audit and record behavior unstated. Fix: name the value; consider allow once, allow for site, deny, revoke in settings. [SY-8, ZO-10, CO-8, S8-N3]

M9. Lines 34-49 vs 97-118. No requirements-to-design mapping and no clear relation between new Policy and Store types and existing rule store and coordinator. Fix: add a short mapping and relation section. [ST-6, DC-4]

## Nit

N1. Lines 153, 197. "21 languages" vs INDEX "ten localization parity gates". Fix: cite the enforced count. [ZO-13, DC-13, ST-13, SY-12, S8-N1]

N2. Lines 21, 100-105. "Explicitly empty" and "filled in" overstate: alert/confirm/prompt delegates exist; only the permission extension (line 1492) is empty. Fix: reword. [ZO-9, S8-N2]

N3. Lines 159-175. Policy and store could be one type; single-handler guarantee is implementation detail. [ZO-11, S8-N4]

N4. Lines 52-56. Navigation unchanged vs new agent-tab permission deny reads contradictory. Fix: say it explicitly. [CO-11, S8-N6]

N5. No ladder position statement; ZP-14 dependencies (vault, log store, rule store) and ZP-15 overlap not in INDEX. Fix: one line each. [IC-5, DC-10, ST-9, S8-N7]

N6. Lines 99-118, 120. Diagram mixes pseudocode; classification table never shown; line 120 repeats lines 107-109. Fix: add a sample table, drop the repeat. [ST-11, SY-18]

N7. Lines 155-189. File list lacks ordering; item 5 parenthetical garbled; item 6 label style inconsistent. [ST-12, CO-10]

N8. Lines 100, 159-181. `Browser/` vs `BrowserAutomation/` placement unconfirmed; no full paths. [IC-9, SY-21]

N9. Lines 20-21, 38. "Device and sensitive permissions", "device access requests", "device classes" used interchangeably; three-item rhetorical list at lines 29-30. [SY-15, SY-16]

N10. Lines 62-69. "Lean" used as a decision rule; give the criterion. [SY-17]

N11. Lines 152, 176. "Bubble states are announced" lacks mechanism; "DESIGN.md compliant" names no rule. [SY-19]

N12. Lines 41-45, 52-57. No measurable targets; Non-Goals omit system TCC prompt, file chooser, downloads, policy export; no rollout or compatibility note. [ST-8, ST-14]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-13 (zoom-out), IC-1 to IC-9 (inventive catalyst). Feeds B1-B5, M1, M2, M5, M6, M8, N1-N3, N5, N8.
- comprehension_report.md: CO-1 to CO-11. Feeds B1, B3, B4, M1, M2, M7, N1, N4, N7.
- critic_report.md: DC-1 to DC-15 (design), ST-1 to ST-14 (structural), SY-1 to SY-21 (style). Feeds all Blockers and most Majors and Nits.
- intermediate_zoom_out_report.md: S8-B1 to S8-B4, S8-M5 to S8-M10, S8-N1 to S8-N8. Confirms no remediation since Stage 5 and all Stage 5-7 Blockers and Majors open. Recommends rejecting scope-adding asks (public suffix keying, screen-share and device pickers, rate limiting, policy export, app-wide sync, state-machine framework) and a minimal shape: enforce only delegate-exposed classes, hard-deny in agent tabs, extend `BrowserPermissionRuleStore`, one audit sink.
- Merge notes: DC-1 and ZO-2 and IC-1 merged into B1; baseline-behavior portions split into B2. The Stage 5 and Stage 7 numbering restarts per reviewer; the DC Blocker list (DC-1 to DC-3) maps to B1, B4, B5.
