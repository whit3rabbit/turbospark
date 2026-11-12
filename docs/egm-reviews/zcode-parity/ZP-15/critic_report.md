# Critic report (Stage 7, round 1): ZP-15 browser video recording

Target: docs/designs/zcode-parity/ZP-15-browser-webm-recording.md (full file). One round only; convergence not assessed. Reviewers: egm-design (DC, feature-spec-standard), egm-structural (ST, feature-spec-standard), egm-style (SY). All completed.

## Design (DC)

DC-1 (Major) Lines 51-61, 99, 146-151. Central capture decision left open but plan hard-codes SCStream; a content-rect crop still captures overlays (popovers, sheets, other panes), breaking the page-only goal; hosting-window reparenting is uncosted. Fix: spike and record the decision, or state the exact SCContentFilter and occlusion failure behavior.

DC-2 (Major) Lines 28, 62-65, 136, 161-163. WebM vs MP4 unresolved and self-contradictory; INDEX line 42 still says WebM; format setting implies an undefined choice. Fix: decide MP4, rename draft, title and INDEX row, remove the format setting.

DC-3 (Major) Lines 51-61, 180-182. Screen Recording permission (TCC owner, denied and revoked paths, macOS recording nag) unaddressed. Fix: add a permission section plus denied-path CUJ and test.

DC-4 (Major) Lines 77-90, 108-109, 128-135. Privacy under-specified: login pages, 2FA and bank pages are captured; no retention, cleanup, or model/remote-provider (ZP-22)/export exposure rules; session approval conflicts with "ask always"; ladder position only implied. Fix: add a Risks section on sensitive pages, retention and export, approval semantics, and state the ladder position.

DC-5 (Nit) Lines 39, 108-109. 500 MB cap and duration cap unjustified (no bitrate, fps, Retina estimate, no duration default). Fix: state MB per minute, fps, resolution cap and duration default.

DC-6 (Nit) Lines 85-90. No mechanism for crash, force-quit, disk-full, tab close or lease end. Fix: specify fragmented MP4 or a recovery pass and stop triggers.

DC-7 (Nit) Lines 112-128. Alternatives thin; "by author" attribution; missing private WKWebView API and CDP-style screencast. Fix: restate reasons as technical trade-offs, add the missing alternatives.

DC-8 (Nit) Lines 3-5, 164-172. No occlusion/privacy-boundary test, no CPU or frame-drop budget; the ZCode recording spec (docs/browser-use/2026-08-12-browser-webview-video-recording-spec.md) is named but not linked. Fix: add the acceptance numbers and the link.

## Structural (ST)

Heading hierarchy clean; no skipped levels or broken tables.

ST-1 (Major) Lines 112-128. Rejection reasons are "Rejected by author" with no linked issue; Alternative A is called fallback-only yet shipped as fallback. Fix: rewrite Reason lines as concrete trade-offs.

ST-2 (Major) Lines 49-65 vs 92-173. Capture and container options live in Open Questions, not Alternatives, while the plan commits to SCStream and MP4. Fix: move options into Alternatives Considered; keep Open Questions for what is truly open.

ST-3 (Major) Missing section. No Risks and Mitigations section (required by the standard); failure modes scattered (permission revoked, navigation/resize/occlusion, quit or crash, disk full, hosting window failure, process ownership). Fix: add the section.

ST-4 (Major) Lines 1-3, 62-66, 114-128, 136, 146-173. Title and filename say WebM, plan outputs MP4, format field present. Fix: retitle or record the container decision; align filename and INDEX.

ST-5 (Nit) Lines 3-6, 22-28. ZCode recording spec cited by name only. Fix: add the link.

ST-6 (Major) Lines 77-83, 133, 156-160. CUJ 2 one-time approval conflicts with ask-gated always; "debuggable flow" vague; "opt-in per session" undefined. Fix: reconcile approval semantics and sharpen the Given.

ST-7 (Nit) Lines 30-40, 85-90, 109. Duration cap absent from Goals and CUJ 3, no default. Fix: add and state a default.

ST-8 (Nit) Lines 92-110. Diagram omits helper-app boundary and agent tool path; heading is "Technical Plan" not "Proposed Architecture". Fix: rename or note, show the boundary.

ST-9 (Nit) Lines 139-140. "21 languages" vs INDEX "ten localization parity gates". Fix: state a verified count or reference the gates.

ST-10 (Nit) Lines 8-10, 186. Metadata block matches INDEX pattern. Fix: none required; verify against the template.

## Style (SY)

SY-1 (Major) Lines 1, 3-5, 62-66, 146-151, 163. Title says WebM, body says MP4; format field undefined. Fix: resolve container, retitle.

SY-2 (Major) Lines 51-61, 62-66, 114-128. Option (b) is an open fallback at 59-61 and "rejected by author" at 120-121. Fix: mark capture mechanism decided with (b) as fallback in one place.

SY-3 (Major) Lines 18-20. Reversal of the earlier exclusion has no linked spec. Fix: link .kiro/specs/browser-automation and the exclusion line.

SY-4 (Major) Lines 54-58. "Solves it" and "in-repo precedent" assert feasibility without evidence. Fix: say "may address; unverified until prototyped" and cite the Computer Use file path.

SY-5 (Major) Lines 39-40, 108-109. Caps underspecified: no duration default, "named notice" and disk-space threshold undefined. Fix: give values and the string key.

SY-6 (Major) Lines 77-83, 134-135, 157-158. CUJ 2 vague and inconsistent; "stops at session end or tool call" ambiguous. Fix: state per-call vs per-session approval and "stops when the agent calls browser_stop_recording or the session ends".

SY-7 (Major) Lines 30-34, 163. "Opt-in per session" has no control or setting. Fix: add the setting or drop the phrase.

SY-8 (Major) Lines 134-135. Ladder position not named, required by INDEX program constraints. Fix: name it.

SY-9 (Nit) Lines 136-138. Hardware-encode and no-transcript-cost claims unmeasured. Fix: say expected, not verified, and add a target or drop.

SY-10 (Nit) Lines 89-90. "Nothing is truncated corrupt" ungrammatical. Fix: "the finalized file plays to the last written frame."

SY-11 (Nit) Lines 118-119. Subjective wording ("choppy", "invisible exactly when"). Fix: state the lower frame rate and timing drift.

SY-12 (Nit) Lines 5, 24-27. ZCode CI smoke claim unlinked. Fix: link the spec or smoke file.

SY-13 (Nit) Lines 12, 22-25. Filler phrases. Fix: tighten the Overview sentence.

SY-14 (Nit) Lines 94-106. Diagram shows primary path only. Fix: label it so or show the fallback.

SY-15 (Nit) Lines 155-160, 180-182. "Media capability pattern" and "hardware-gate rules" unlinked. Fix: link the file or rule.

SY-16 (Nit) Lines 8-10. Status "Draft" alongside decided-sounding text. Fix: align with SY-2.
