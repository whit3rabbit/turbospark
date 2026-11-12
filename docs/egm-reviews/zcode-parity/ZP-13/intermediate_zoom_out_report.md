# Stage 8: Intermediate zoom-out report (ZP-13)

Target: full file, 152 lines. Inputs: zoom_out_report.md, comprehension_report.md, critic_report.md. No Blockers. Rule applied: keep asks that remove an ambiguity or fix a factual error; reject asks that add a mechanism, section or surface.

S8-1 (Major) Lines 15-16, 78-84, 119-122. New finding: `Theme/AppDockIconRenderer.swift` (`applyDockIcon`, lines ~20, 23, 109) already owns `applicationIconImage` for user-selectable Dock icon presets. The draft adds a second writer, so the last writer wins and the badge or the preset is lost. Make the badge a post-step on the renderer's output and reuse its image source. This also settles the icon-source questions (ZO-2, DC-4, CO-3).
S8-2 (Major) Lines 31-34, 46-53, 128-131. Keep: resolve to one signal, an Info.plist key written by `scripts/make-app-bundle.sh` before codesign (signing runs at lines ~445-475, after the plist). Drop the `#if DEBUG` leg and the install-location check. Replace the Makefile item. No case table.
S8-3 (Major) Lines 22-27, 57-62. Decide whether a locally packaged bundle run from `dist/` is badged; under the marker rule it is not, which undercuts the Problem. Either write the marker only in a release mode or narrow the Problem to debug runs. Add one non-goal: the badge does not isolate bundle id or data. Reject a distinct dev bundle id.
S8-4 (Major) Lines 59, 79. CUJ 1 vs `swift run`: use the renderer's image source or scope to bundles and drop `swift run`. No fallback sourcing mechanism.
S8-5 (Major) Lines 78-84, 108-113, 124-127. Composite synchronously on main in `ForegroundAppDelegate.applicationDidFinishLaunching`; delete "off main thread", "sub-100 ms", "when possible". Reject caching.
S8-6 (Major) Lines 86-88, 112-113. Cut Core Graphics detail to one line; "on failure keep the current icon and log once"; a nil check is enough. Reject blank-output validation (IC-7).
S8-7 (Nit) Lines 15, 31-33, 42-43. Delete "window icon" and "window representation"; Dock icon only; link DESIGN.md for the accent blue.
S8-8 (Nit) Lines 90-106. Keep one added alternative, build-time stamping (reject reason: no unbundled run support, needs image tooling without Python or Node). Reword "rejected by author" to trade-offs; fix "asset catalogs" wording. Reject NSDockTile and distinct-id alternatives.
S8-9 (Major, reduced) Lines 108-113. Reject a full Risks section; keep two bullets: plist key written before codesign, and a release shipping without the key would be badged. One clause that "DEV" is drawn into an image so localization gates do not apply.
S8-10 (Nit) Lines 133-148. Keep: policy takes marker and configuration as inputs; one `verify-app-bundle.sh` assertion that the key is present; synthetic source image in tests; restate mutation check as a requirement; exact build command. Reject: design-check entry, tolerance matrices, extra compositor test.
S8-11 (Nit) Lines 3-5, 19-27. Reject prose rework; one clause defining ZCode and parity; delete the two filler sentences.

Ledger: rejected as scope creep: install-location check, `#if DEBUG` leg, distinct dev bundle id, full Risks section, off-main compositing and budget and cache, blank-output validation, NSDockTile alternative, design-check entry. Kept: single plist key, delegate hook, icon source fix, build-time alternative, one verify assertion, compose with AppDockIconRenderer.
