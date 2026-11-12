# Stage 7: Critic report (ZP-13)

One round, no remediation. Design and structural used feature-spec-standard; style had no separate standard.

## Design (egm-design)

DC-1 (Major) Lines 128-131. Marker goes in the wrong file. `Makefile` `app-bundle` only runs `scripts/make-app-bundle.sh` (icon copy 286-289, Info.plist heredoc ~291). Write the marker there, before the codesign pass (ordering warning at 278-280). An Info.plist key (for example `TurboSparkReleasePackaged`) avoids a loose file in the signed bundle.
DC-2 (Major) Lines 31-34, 48-53, 59-62. Detection rule undecided; Goals mix (b) and (c), lean is (a)+(b). `swift run` makes a bare executable that never reaches marker logic. A locally packaged build run from dist/ (the release-verification case) stays unbadged, undermining lines 22-27. Make the marker mean "official release artifact" or badge when not under /Applications.
DC-3 (Major) Lines 79-83, 110-112, 125-127. Off-main compositing vs main-thread NSApp assignment is contradictory; 100 ms has no measurement. Composite synchronously on main or specify the MainActor hop. `NSApp` can be nil in `App.init`; use `ForegroundAppDelegate.applicationDidFinishLaunching`.
DC-4 (Major) Lines 66, 79, 119-122. Icon source mis-described: only a loose `AppIcon.icns` exists, no asset catalog. In `swift run` the missing source would be the normal path and log every launch. Pick a source or scope to bundles and drop `swift run` from CUJ 1.
DC-5 (Major) Lines 19-27. Badge changes only the Dock image; shared bundle id means shared defaults, Keychain, app data and any single-instance lock, so "beside the installed release" may not run. State the badge is a visual aid only, or record the risk.
DC-6 (Major) Lines 108-113. No Risks section: Dock persistence after quit, undefined window surface, accent legibility in light and dark, localization status of the "DEV" glyph, accessibility, test seams.
DC-7 (Nit) Lines 90-106. Missing alternatives: build-time icon stamping in `make-app-bundle.sh`, `NSDockTile`, distinct dev bundle id. Alternative A "truncated" unsupported.
DC-8 (Nit) Lines 133-140. Corner sampling has no tolerance, size or scale; add a test that the release path never invokes the compositor.
DC-9 (Nit) Lines 142-148. No `make design-check` entry; check the marker survives codesign via `verify-app-bundle.sh`.

## Structural (egm-structural)

ST-1 (Major) Whole document, near 108-113. No Risks and Mitigations section; include marker breaking the signing seal, release shipping without marker getting badged, false badge on manual release run.
ST-2 (Major) Lines 97-98, 105-106. Rejections attributed to "the author"; restate as trade-offs.
ST-3 (Major) Lines 31-34, 46-53, 128-131. Open Question leaves Goals, Makefile item and tests unresolved; state who resolves it.
ST-4 (Major) Lines 15-16, 31-34, 42-43, 81-82. Surface ("Dock and window icon", "Dock icon", "window representation") stated three ways; only applicationIconImage is set.
ST-5 (Nit) Lines 79-82 vs 110-111. Threading hand-off undocumented.
ST-6 (Nit) Line 139. "Mutation checks" mis-nested as a sibling field.
ST-7 (Nit) Line 19. "The Problem" vs "Problem Statement".
ST-8 (Nit) Lines 64-69. CUJ 2 covers only a missing icon; Goals cover any failure.

## Style (egm-style)

SY-1 (Major) Lines 31-33, 48-53, 126-131. Detection rule stated inconsistently; state one rule, mark question resolved or neutral.
SY-2 (Major) Lines 110-111. "Sub-100 ms" and "when possible" are not measurable; no verification step.
SY-3 (Major) Lines 112-113. "Full error handling" is vague; name do/catch and nil-checked CG calls; traps are not catchable.
SY-4 (Major) Lines 15, 31-33, 43. Surface described three ways; overclaims the plan.
SY-5 (Nit) Lines 15, 31, 44, 86-87. "Blue" vs "accent blue"; "no red variants" implies others.
SY-6 (Nit) Lines 21-27. "Recreate state" unspecific; cut "matters for this repo specifically".
SY-7 (Nit) Lines 22-23. "Window title does not distinguish" unverified.
SY-8 (Nit) Lines 34-37, 114-115. "No detection cost beyond the check" is a tautology.
SY-9 (Nit) Lines 92-106. "Rejected by author" adds nothing; state the real reason.
SY-10 (Nit) Lines 139-140. Mutation check is a procedure fragment; state as a test requirement.
SY-11 (Nit) Lines 73-84, 126-127. Diagram thread order unclear.
SY-12 (Nit) Line 145. "Build the app scheme" lacks the exact command.
