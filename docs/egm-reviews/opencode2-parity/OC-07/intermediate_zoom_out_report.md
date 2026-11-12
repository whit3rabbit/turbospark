# Stage 8: Intermediate zoom out report (OC-07)

Reviewer: egm-zoom-out (S8), full file, given zoom_out_report.md. Completed. Target unedited since Stage 5. Verdict: no Blocker under the Zoom Out standard.

## Stage 5 findings still open

S8-1 Major, lines 31-34, 149-151 (ZO-1): ModelInstallGate is a per-install decision enum, not an installed-model registry. Fix: delete item 5, read installed-model state from AppModel.
S8-2 Major, lines 72-74, 104-120 (ZO-2): flag scope and order against the vault gate unresolved; diagram contradicts line 119. Fix: decide both, justify against BrowserSettings.onboardingDone (profile-scoped precedent).
S8-3 Major, lines 133-163 (ZO-3): injectable step machinery, absent-step rule and mutation check serve a nonexistent import step. Fix: three fixed steps now, add import when the wizard ships.
S8-4 Major, lines 38-45, 75-78, 124 (ZO-4): splash on every launch with no measured need; floor contradicts NFR. Fix: show only if state is not ready within about 150 ms, or first run only.
S8-5 Major, lines 24-26, 50-52, 82 (ZO-5): import step depends on an unverifiable spec. Fix: remove step 3 and the hand-off sentence, or cite the spec location.
S8-6 Nit (ZO-6 to ZO-10): swift link depth, string inventory, name scanned prompt APIs, FirstRun/ folder, restate parity.

## Later-stage asks that exceed Zoom Out scope

- DC-1, ST-1 (Alternatives), DC-5, ST-2 (Risks): required by the feature-spec standard; keep to a few lines each. Alternatives: two lines (overlay vs second window, app-level vs profile flag), not a five-row matrix.
- DC-3: take as one sentence (existing models or chats mean not fresh); no new CUJ or multi-profile rule.
- DC-6, CO-6: collapse to one decision, step 2 navigates to the existing install hub; defer failure behavior to the hub.
- DC-8, DC-9, ST-7, CO-1: add weight. Fix the diagram only to show the returning branch and the vault gate; name the readiness signal; skip store-format spec.
- CO-5, SY-4: one sentence (Skip advances one step; flag written on finish or skip-all; quit mid-flow leaves it unset).
- CO-3, DC-7, SY-5, ZO-10, ST-5, DC-14: one Overview paragraph (parity means sequence and gating; sign-in and animated outline dropped; four-step flow is a TurboSpark design).
- IC-5: simplest cut is folding the primer into welcome copy, leaving welcome plus first model.
- IC-8: keep (sequence localization after the step list is final).
- Polish (ST-8, ST-10, SY-7, SY-8, SY-13, SY-16, SY-17, CO-11 to CO-17, DC-13, ST-11 to ST-13): optional Nits.

## Minimal edit set closing all open Majors

1 delete item 5 and the gate claim; 2 drop the import step and its test rule; 3 decide flag scope, state order versus the vault gate, fix the diagram; 4 make the splash conditional and merge the floor and Reduce Motion statements into one rule; 5 optionally fold the primer into welcome; plus one migration sentence and one parity sentence.
