# Stage 6: Comprehension report (ZP-13)

Target: full file. Reviewer: egm-comprehension (no context).

## Paraphrase

At startup, if the running TurboSparkApp is an unpackaged development build, the app draws a blue diagonal "DEV" ribbon over its icon with Core Graphics, off the main thread, and assigns it to NSApp.applicationIconImage so the Dock icon differs from an installed release. On failure it keeps the original icon and logs one line. Packaged releases skip the path. The change adds DevIconBadger.swift, a launch hook, an optional Makefile marker step, and unit tests.

## Findings

CO-1 (Major) Lines 31-33 vs 48-53. The Goal's "unpackaged" rule mixes "or" and "with" ambiguously, includes an install-location check that the lean excludes, and "/Applications-style" is undefined. The rule is a preference, not a decision, yet plan, tests and Makefile hinge on it.
CO-2 (Major) Lines 48-53, 128-131. The marker is unspecified: name, path in the bundle, who verifies it, whether it exists today. No case table for build type and location.
CO-3 (Major) Lines 59, 79, 115-127. Unclear whether `swift run` yields an .app bundle; icon source for an unbundled run ("bundle icon or Assets") is undefined.
CO-4 (Major) Lines 15, 43, 86-88. "Design system's accent blue" is not named or linked; "window icon" is never realized in the plan, which only sets applicationIconImage.
CO-5 (Major) Lines 78-83, 110-113. Off-main compositing vs main-thread NSApp assignment: which step runs where, and what happens if it finishes after window display. The 100 ms budget has no measurement method.
CO-6 (Nit) Lines 36-38, 82. "The check" is the unresolved rule; "Dock icon updates automatically" is asserted without explanation.
CO-7 (Nit) Lines 3-5. ZCode and "parity" are not introduced.
CO-8 (Nit) Lines 24-27. "Recreate state under test" is vague; a verified packaged bundle is not badged, so the link to the badge is unexplained.
CO-9 (Nit) Lines 139-140. Mutation check is terse (manual or automated?); "injected failure" mechanism unstated.
CO-10 (Nit) Lines 97-106. Rejection reasons restate the cons; no independent rationale for Alternative B.

No Blockers.
