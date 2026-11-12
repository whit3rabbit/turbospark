# Stage 6: Comprehension report (OC-07)

Reviewer: egm-comprehension (CO), read the document alone. Completed.

## What this proposes

The document proposes two additions to the TurboSpark Swift app. First, a branded splash overlay inside the main window (wordmark plus a quiet pulse) that shows while app state loads and then dismisses. Second, a first-run flow for brand-new installs with up to four skippable steps: welcome, first model, import hand-off (only if an import wizard exists), and a permissions primer that explains what the Audio workspace will later ask for. Finishing or skipping the flow sets a persisted flag so it never replays, and then opens a new chat with the composer focused. The work is described as parity with OpenCode's onboarding and splash, and it adds three new Swift files, edits three existing ones, and adds two test files.

## How the existing system works, per this document

The app has no persisted first-launch flag, tour, or checklist. A new user sees a static welcome hero view and some model-install gating logic, and the main window appears directly with no branded loading state. There is a "vault unlock gate" and an "encrypted profile", so some data is encrypted and gated behind an unlock step. There are "permission rules" under which only the Audio surfaces' Allow buttons request mic and screen access. An "install hub entry" exists, and a desktop-shell-hardening import wizard is specced but may not exist yet. The document never says how the launch sequence is ordered today.

## Findings

CO-1 Major, lines 105-120, 148: diagram shows both paths through the splash but not the fresh/returning branch or where the vault gate sits; "vault gate composition" unexplained. Fix: redraw with the returning branch to main surface and the vault gate marked.
CO-2 Major, lines 72-74 vs 104-106, 145-148: flag location is open yet the plan assumes app-level; line 119 conflicts with flow running before unlock. Fix: resolve and record, or mark the plan provisional and state order versus the vault gate.
CO-3 Major, lines 3-9, Overview: does not say what OpenCode does or which parts are intentionally dropped. Fix: add 2-3 lines.
CO-4 Major, lines 48-52, 163-164: "import wizard exists" is not a checkable condition; tests say "import spec absent". Fix: define it concretely and use one term.
CO-5 Major, lines 46-47, 57-58, 66, 89-94: Skip, back, and progress semantics undefined for a variable step list; unclear when the flag is written. Fix: state what each Skip does and when the flag is written.
CO-6 Major, lines 86-88, 93-94: install in progress or failure behavior unspecified; unclear whether the empty-state prompt exists today. Fix: specify both.
CO-7 Major, lines 75-77, 124: ready signal undefined and not located in code; 400 ms floor contradicts "no artificial delay". Fix: define the signal and reconcile.
CO-8 Major, lines 42, 77: Reduce Motion "instant dismiss" conflicts with the 400 ms floor. Fix: state whether the floor is waived.
CO-9 Major, lines 51-56, 67-68: "permission rules" and "Audio workspace" undefined or unlinked. Fix: link and define in one sentence.
CO-10 Major, lines 31-34, 48-49, 149-151: "install hub entry", "install gating", "fits" undefined. Fix: define "fits" and describe the hub in one sentence.
CO-11 Nit, lines 126-127, 168: 21-language list and localization gates not linked. Fix: link.
CO-12 Nit, lines 25, 50: what the import wizard imports is not stated. Fix: one sentence.
CO-13 Nit, lines 135-148, 157-159: launch order for TurboSparkApp.swift not restated; mutation check unexplained. Fix: restate order; define the term.
CO-14 Nit, lines 145-146, 168-172: `make` targets, DESIGN.md section 9, "400-line guidance" unlinked. Fix: link.
CO-15 Nit, lines 160-163: reference test unlinked, "prompt APIs" unnamed. Fix: link and list.
CO-16 Nit, lines 129-131: primary action per step unnamed. Fix: name it.
CO-17 Nit, line 63: non-goal has no OpenCode baseline. Fix: tie to CO-3.
