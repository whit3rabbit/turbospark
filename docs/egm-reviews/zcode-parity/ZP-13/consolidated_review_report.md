# Consolidated review report: ZP-13 dev-build icon badge

Target: docs/designs/zcode-parity/ZP-13-dev-build-icon-badge.md (full file, 152 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO and IC = zoom_out_report (zoom-out, inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 0, Major 9, Nit 7.

## Blocker

None.

## Major

M1. Lines 31-34, 48-53, 59-62, 128-131. Detection rule undecided and inconsistent (Goals mix (b) and (c), lean is (a)+(b)); marker change targets the Makefile, which only calls `scripts/make-app-bundle.sh`. Fix: resolve to one Info.plist key written by the script before codesign; drop `#if DEBUG` and install-location legs; modify the script, not the Makefile. [ZO-1, IC-2, CO-1, CO-2, DC-1, DC-2, ST-3, SY-1, S8-2]
M2. Lines 15-16, 78-84, 119-122. A second writer of `applicationIconImage` ignores the existing `AppDockIconRenderer` (user Dock icon presets); last writer wins. Fix: apply the badge as a post-step on the renderer's output and reuse its image source. [S8-1]
M3. Lines 59, 79, 115-127. Icon source is wrong ("bundle icon or Assets"; only a loose `AppIcon.icns` exists) and `swift run` has no bundle, so CUJ 1 hits the failure path. Fix: use the renderer's source or scope to bundles and drop `swift run` from CUJ 1. [ZO-2, CO-3, DC-4, S8-4]
M4. Lines 78-84, 108-113, 124-127. Off-main compositing contradicts the main-thread assignment; 100 ms budget and "when possible" unmeasurable; `App.init` is an unsafe hook. Fix: composite synchronously in `ForegroundAppDelegate.applicationDidFinishLaunching`; delete the budget. [ZO-4, ZO-6, IC-3, CO-5, DC-3, SY-2, ST-5, SY-11, S8-5]
M5. Lines 19-27, 57-62. A locally packaged bundle run from `dist/` (the verification case) is unbadged, undercutting the Problem; the state-confusion claim overreaches since a badge does not isolate bundle id or data. Fix: decide local-package behavior; add a non-goal that the badge is a visual aid only. [ZO-5, DC-2, DC-5, IC-4, CO-8, SY-6, S8-3]
M6. Lines 15, 31-33, 42-43, 81-82. Surface stated three ways (Dock and window icon, Dock icon, window representation); only the Dock icon is set. Fix: Dock icon only, stated once. [ZO-7, IC-5, CO-4, ST-4, SY-4, S8-7]
M7. Lines 108-113. No Risks section (required by the standard). Fix: two bullets only: plist key written before codesign, release shipping without the key is badged; note "DEV" is an image so localization gates do not apply. [ST-1, DC-6, S8-9]
M8. Lines 90-106. Rejections attributed to "the author"; "asset catalogs" is wrong; build-time stamping is the strongest missing alternative. Fix: state real trade-offs, fix wording, add build-time alternative C. [ST-2, DC-7, IC-1, SY-9, ZO-3, CO-10, S8-8]
M9. Lines 86-88, 112-113. "Full error handling" is vague and Core Graphics detail is over-specified. Fix: "on failure keep the current icon and log once", nil check; one-line ribbon description. [SY-3, ZO-4, S8-6]

## Nit

N1. Lines 133-148. Tests and verification: no release-gate assertion, mutation check mis-nested, "build the app scheme" has no command, corner sampling fragile. Fix: add one `verify-app-bundle.sh` key assertion, inject marker and configuration into the policy, use a synthetic source image, give the exact command. [ZO-8, IC-6, DC-8, DC-9, CO-9, ST-6, SY-10, SY-12, S8-10]
N2. Lines 64-69. CUJ 2 covers only a missing icon; Goals cover any failure. Fix: widen the Given; nil check suffices. [ST-8, IC-7]
N3. Lines 3-5. ZCode and "parity" undefined. Fix: one clause. [CO-7, S8-11]
N4. Lines 21-27, 34-37, 114-115. Filler and tautology: "matters for this repo specifically", "no detection cost beyond the check", unverified title claim. Fix: cut or verify. [SY-6, SY-7, SY-8, CO-6]
N5. Lines 15, 31, 44, 86-87. "Blue" vs "design system's accent blue" unnamed; "no red variants" wording. Fix: link DESIGN.md or the token. [SY-5, CO-4]
N6. Line 19. Heading "The Problem" vs "Problem Statement". Fix: rename if the template requires. [ST-7]
N7. Lines 73-84. Diagram thread order unclear. Fix: resolved by M4. [SY-11]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-8, IC-1 to IC-7. Feeds M1-M6, M8, M9, N1, N2.
- comprehension_report.md: CO-1 to CO-10. Feeds M1, M3-M6, M8, N1, N3-N5.
- critic_report.md: DC-1 to DC-9, ST-1 to ST-8, SY-1 to SY-12. Feeds all Majors and Nits.
- intermediate_zoom_out_report.md: S8-1 to S8-11. S8-1 (existing AppDockIconRenderer) is the only finding not raised earlier. Stage 8 recommends rejecting as scope creep: install-location check, `#if DEBUG` leg, distinct dev bundle id, full Risks section, caching and off-main budget, blank-output validation, NSDockTile alternative, design-check entry.
