# Consolidated review report: OC-07 first-run onboarding and splash

Target: docs/designs/opencode2-parity/OC-07-onboarding-and-splash.md (full file, 176 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All eight reviewers completed; none failed. Unverifiable: .kiro/specs/desktop-shell-hardening/brief.md (absent in this checkout).
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 14, Nit 8.

## Blocker

B1. Whole doc (no section). No Alternatives Considered; the standard fails a single-path spec. Fix: add two to three lines each on overlay vs second window, app-level vs profile flag, state-ready vs timer, with technical rejection reasons. [DC-1, ST-1, S8 scope note]
B2. Whole doc (no section; see 62-66, 72-74, 89-94). No Risks & Mitigations (crash mid-flow, corrupt or deleted flag replaying, splash never dismissing, install failure or low disk, primer drift). Fix: add a short section. [DC-5, ST-2, IC-6]

## Major

M1. Lines 31-34, 149-151. ModelInstallGate is a per-install verdict function with no installed-model state; modifying it misplaces logic. Fix: drop item 5, read installed-model state from AppModel. [ZO-1, IC-3, S8-1]
M2. Lines 72-74, 104-120, 145-148. Flag scope and flow order against the vault gate unresolved; diagram omits the gate, drops the returning branch, and contradicts line 119. Fix: decide scope (note profile-scoped BrowserSettings.onboardingDone precedent), state order, redraw. [ZO-2, IC-2, CO-1, CO-2, DC-2, ST-6, SY-3, S8-2]
M3. Lines 38-45, 75-78, 124, 42. Splash on every launch with no measured need; 400 ms floor contradicts "no artificial delay" and Reduce Motion instant dismiss; no timeout or failure path; ready signal undefined. Fix: show only if slow (about 150 ms) or first run only, one floor rule, add a cap, name the signal. [ZO-4, IC-1, CO-7, CO-8, DC-4, SY-1, SY-10, S8-4]
M4. Lines 24-26, 50-52, 82, 119-120, 133-163. Speculative injectable step machine and absent-step test for an import wizard that cannot be verified; "exists" undefined. Fix: fixed steps now, add import when the wizard ships. [ZO-3, ZO-5, IC-4, CO-4, ST-4, SY-11, S8-3, S8-5]
M5. Lines 28-36, 44-45. Fresh-install detection undefined for upgrading users, reinstall, flag loss. Fix: one sentence: existing models or chats mean not fresh. [DC-3, IC-6, S8 scope note]
M6. Lines 46, 57-58, 65-66, 89-94. Skip, Skip-all, back, progress, and quit-mid-flow flag semantics undefined. Fix: one sentence defining each and when the flag is written. [CO-5, SY-4, IC-9]
M7. Lines 46-52, 62, 86-88. First-model step: embed vs navigate, "fits", install in progress or failure, offline policy unspecified. Fix: state step 2 navigates to the existing install hub and defers failure handling to it. [DC-6, CO-6, CO-10]
M8. Lines 3-9, 19-26, 63-64. Parity claim unbacked: OpenCode prefetches state and opens a draft tab (comparison line 376), has console sign-in, and an animated splash; welcome, primer, and import steps are TurboSpark design. Section 1 finding (line 85) uncited. Fix: one Overview paragraph on what is adopted and dropped; cite both findings. [CO-3, DC-7, SY-5, ZO-10, IC-10, ST-5, S8]
M9. Lines 53-56, 67-68. Permissions primer duplicates point-of-ask Allow buttons; "permission rules" and "Audio workspace" unlinked and undefined. Fix: fold the primer into welcome copy or link and define the terms. [IC-5, CO-9, SY-6]
M10. Lines 25, 31-34, 50, 85. swift/ links resolve against the wrong base; .kiro brief link dead (also in INDEX.md lines 53, 82). Fix: use ../../../swift/ paths; mark the brief "not yet in repo" in plain text. [ZO-6, IC-11, ST-3, ST-4, DC-10, DC-11, SY-14]
M11. Lines 160-163. Source-scan test is brittle, names no APIs, and misses transitive prompts. Fix: name the API list from RecordingPermissionsTests and add a runtime check with a recording fake. [DC-9, ZO-8, CO-15]
M12. Lines 126-127, 150-153. 21-language cost unsized; no string inventory; CUJ 2 empty-state string unplanned. Fix: list strings, sequence localization after the step list is final. [DC-8, ZO-7, IC-8]
M13. Lines 86-88 vs 96-100, 108-109. CUJ 1 says a relaunch shows the hero directly, but the splash always shows. Fix: "splash, then the last state". [SY-2]
M14. Lines 29-37, 102-120. Plan lacks flag store, step model type, readiness signal, and order versus ProfileUnlockView. Fix: short data-flow paragraph, name only (no store-format spec). [ST-7]

## Nit

N1. Lines 137-148. Flow files under App/ rather than with their feature. Fix: FirstRun/ folder. [ZO-9]
N2. Lines 154-159. Test and mutation-check detail in a strategic doc; term unexplained. Fix: trim or restate. [IC-7, CO-13]
N3. Lines 126, 144-146, 159, 168-172. DESIGN.md section 9, 400-line guidance, localization gates, make targets unlinked; line 144 hedged. Fix: link INDEX constraints and DESIGN.md. [CO-11, CO-14, ST-11, ST-12, SY-15, SY-16, DC-13]
N4. Lines 57-58, 93-95, 99-100. Empty-state prompt and "last state as today" not in the plan. Fix: name the existing view or add it. [DC-12, SY-12]
N5. Lines 1-13. Upstream links unpinned; INDEX cites commit 7ce35879b8. Fix: pin. [DC-14]
N6. Lines 70-77. Open Questions precede the plan that answers them; no owner. Fix: move after the plan, name owner. [ST-8, ST-10]
N7. Lines 19, 40, 63-64, 72, 122, 129-131, 45-68. Wording: "genuinely", "quiet pulse", "brand moment", redundancy, no splash budget, primary action unnamed, non-goal lacks baseline. Fix: tighten per source items. [SY-7, SY-8, SY-9, SY-13, SY-17, CO-16, CO-17]
N8. Line 152. Localizable.xcstrings and test paths not verified by one reviewer. Fix: verify before implementation. [ST-13]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-10, IC-1 to IC-11.
- comprehension_report.md: CO-1 to CO-17.
- critic_report.md: DC-1 to DC-14, ST-1 to ST-13, SY-1 to SY-17.
- intermediate_zoom_out_report.md: S8-1 to S8-6 plus scope notes. Confirms all Stage 5 Majors open. Recommends keeping Alternatives and Risks to a few lines, taking migration, Skip, parity, and diagram fixes as single sentences, and treating the rest as optional polish. Minimal edit set: delete item 5, drop import step, decide flag scope and vault order, conditional splash with one floor rule, optionally fold the primer.
- Merge notes: ST-9 (metadata placement) needs no change and is omitted. Severity of DC-9 and DC-8 kept at Major as rated by the design reviewer; the Stage 8 reviewer would treat them as optional.
