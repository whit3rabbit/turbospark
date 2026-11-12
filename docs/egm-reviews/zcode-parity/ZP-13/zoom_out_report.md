# Stage 5: Zoom out and inventive catalyst report (ZP-13)

Target: docs/designs/zcode-parity/ZP-13-dev-build-icon-badge.md (full file, 152 lines)

## Zoom-out (egm-zoom-out)

ZO-1 (Major) Lines 48-52, 128-131. The marker goes in the wrong file: `make app-bundle` only calls `scripts/make-app-bundle.sh` (icon copy ~287-288, Info.plist ~302), so the Makefile change is a no-op. The script already writes Info.plist, so a release-only key can be the marker. Option (b) alone is root cause; (a) plus (b) makes two code paths. Resolve to (b), drop (a) and (c).
ZO-2 (Major) Lines 15-16, 59, 86-88. "bundle icon or Assets" is wrong: no asset catalog icon exists; the only source is `assets/icons/AppIcon.icns`. Under `swift run` there is no bundle, so CUJ 1 hits the CUJ 2 failure path or badges a generic icon. State the icon source for unbundled runs and fix CUJ 1.
ZO-3 (Nit) Lines 100-106. Alternative B cites "asset catalogs"; the repo uses a loose .icns. Reword.
ZO-4 (Nit) Lines 86-88, 110-113. Core Graphics detail and the sub-100 ms, off-main-thread budget are over-specified; `applicationIconImage` must be set on main. Composite synchronously; "full error handling" is just do/catch.
ZO-5 (Nit) Lines 22-27, 140-141. `verify-app-bundle.sh` runs on a packaged, never-badged bundle. A locally packaged bundle run from `dist/` is unbadged under the marker rule, which is the case that matters most. Decide whether it should badge.
ZO-6 (Nit) Lines 124-127. Name `ForegroundAppDelegate.applicationDidFinishLaunching` as the hook; `App.init()` runs before NSApp is guaranteed.
ZO-7 (Nit) Lines 15, 42-43. "Window icon" is undefined; Goals say Dock only. Cut or define.
ZO-8 (Nit) Lines 133-140. Policy function should take marker and configuration as inputs for deterministic tests; note the badge text is an image, so localization gates do not apply.

## Inventive catalyst (egm-inventive-catalyst)

IC-1 (Major) Lines 46-53, 71-84, 128-131. False compromise: runtime compositing is chosen to keep one icon source, at the cost of detection, compositor, fallback, marker and hook. Move the work to build time: `make-app-bundle.sh` (already copies the icns) produces a badged icon for dev bundles; release bundles are untouched by construction. Liability: `swift run` has no bundle, so keep a `#if DEBUG` runtime path or accept bundled-only; build tooling must avoid Python and Node.
IC-2 (Major) Lines 29-38, 48-53, 57-62. Goals mention an install-location check that the Open Question lean contradicts; the marker file has no other consumer. Use one provenance signal (an Info.plist key written by the packaging script before signing) and resolve the Open Question in the draft.
IC-3 (Major) Lines 78-84, 108-113. "Off main thread" plus "before first window display when possible" can show the unbadged icon at launch; 100 ms budget is unfounded. State the thread contract; composite synchronously or cache the result.
IC-4 (Nit) Lines 19-27. A badge masks, not prevents, shared bundle id and data-directory collisions. Name as an explicit non-goal or consider a distinct dev identifier.
IC-5 (Nit) Lines 15, 31-33, 40-44. Overview promises window icon; plan only sets the Dock icon.
IC-6 (Nit) Lines 133-148. No release-gate check; add an assertion to `verify-app-bundle.sh` that the release key is present. Corner-pixel sampling against the real icon is fragile; use a synthetic source.
IC-7 (Nit) Lines 64-69. Add blank or partially transparent output as a failure case; validate size before assignment.
