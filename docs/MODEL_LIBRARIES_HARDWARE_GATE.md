# Model libraries: hardware and manual gate

This is the record for task 7.3 of the model-libraries spec. Everything below
is **NOT RUN** until someone runs it on the reference machine with a real
external drive and writes the result in the same row. A row with no result
claims nothing. Do not fill a row from a simulator, a fake volume, a render or
a unit test: those are already covered by the automated gates (Rust
integration tests with fake volumes, the Swift app suite with a fake library
service).

Reference machine and drive: Apple M4 Max, macOS 26.5.2, Samsung T7 (USB).
Run on AC power.

Use the app bundle that task 7.2 verified: `dist/TurboSpark.app`, version
0.3.2, ad-hoc signed, built with the pins in docs/RELEASE.md. Do not use a
debug build. Task 7.2 proved only that the bundle builds and stays alive for
ten seconds with an empty HOME. No person has opened it yet.

Keep nothing you care about on the drive under test. Use a copy of a small
model for the move and eject rows. For the streaming rows, use one model large
enough to stream: a mixture-of-experts install above physical memory minus
4 GiB.

How to record a row: result `PASS`, `FAIL` or `N/A`, the measured value where
one is asked for, the date, and one line on what you saw. A crash or hang gets
its own paragraph at the end: steps, what happened, the crash report path.

## A. Drive recognition and speed (requirements 4.1, 4.2, 5.1, 5.2, 5.6)

| # | Check | Result | Value | Notes |
|---|-------|--------|-------|-------|
| A1 | Plug in the T7 on a SuperSpeed (10 Gb/s) cable: Models and Storage lists the connection type and negotiated speed | NOT RUN | | |
| A2 | Same drive on a USB 2.0 cable: the slow-connection warning appears and suggests a 10 Gb/s or faster USB-C cable | NOT RUN | | |
| A3 | Time from mount to the library showing Connected (requirement 4.1: within 5 seconds) | NOT RUN | seconds: | |
| A4 | Rename the volume in Finder while the app runs: the library keeps its name, models keep their install ids, favorites and tags stay | NOT RUN | | |
| A5 | Test speed on the library: finishes in under 15 seconds, shows MB/s, and the load-time estimates use it | NOT RUN | seconds: / MB/s: | |
| A6 | No speed warning or estimate is shown for models on This Mac (requirement 5.7) | NOT RUN | | |

## B. Libraries, offers and installs (requirements 1.x, 2.x, 4.3, 4.4)

| # | Check | Result | Notes |
|---|-------|--------|-------|
| B1 | Add Library... opens the real folder picker, adds the drive, and the first access may show the macOS removable-volume prompt (not shown as an app error) | NOT RUN | |
| B2 | Remove the library, then connect the drive again: the add-library offer appears once, in Installed Models and in the Model libraries section, with one info toast | NOT RUN | |
| B3 | Not now: the offer is gone, does not return after relaunch, and the library can still be added from settings | NOT RUN | |
| B4 | Install a text model into the T7 library with the destination picker: it lists under that library and the free space shown is the T7's | NOT RUN | |
| B5 | Unplug the drive during an install: the install stops, one sentence says the drive was disconnected, no partial model is listed | NOT RUN | |
| B6 | Default library set to the T7, drive unplugged, start an install: the picker preselects This Mac and says why | NOT RUN | |
| B7 | A scan folder on the T7: unplug with settings open and the row flips to Not connected without reopening; plug in and it flips back with the model count | NOT RUN | |
| B8 | First launch after an upgrade from a build with a Move Store root on the T7: the store converts to a library, no model is copied, and the models appear | NOT RUN | |

## C. Not connected, loading and gating (requirements 3.x, 5.4, 5.5)

| # | Check | Result | Notes |
|---|-------|--------|-------|
| C1 | Unplug the drive (no model loaded): no alert, no banner, no toast at launch or refresh; its models read Not connected with the last-seen date | NOT RUN | |
| C2 | Not connected models are absent from the chat loader, composer control, inspector picker and server menu, and the row says "Connect <drive> to load" | NOT RUN | |
| C3 | A user with models only on the unplugged drive sees them as not connected, not the first-run setup screen | NOT RUN | |
| C4 | USB 2.0 cable with a dense model: the slow-load warning with an estimate and "Load anyway"; the model loads after Load anyway | NOT RUN | |
| C5 | USB 2.0 cable with a streaming model: the "generation speed will be limited" warning with Load anyway | NOT RUN | |
| C6 | From the menu-bar server window, choosing a warned model brings the main window forward and asks there | NOT RUN | |

## D. Eject and disconnect (requirements 6.1 to 6.4)

| # | Check | Result | Notes |
|---|-------|--------|-------|
| D1 | Finder eject while idle with a model from the drive loaded in chat: the model unloads, the eject completes, a message names the model | NOT RUN | |
| D2 | Finder eject while that model is generating: generation stops first, then unload and eject | NOT RUN | |
| D3 | Finder eject with the model attached to the local server, and with an audio model resident | NOT RUN | |
| D4 | Cable pull (no eject) with a dense model loaded: the session ends, a message advises ejecting first, the app does not crash, the library list and model records are intact on relaunch | NOT RUN | |
| D5 | Cable pull with a streaming model loaded: same as D4 | NOT RUN | |
| D6 | Quit and relaunch after D4 and D5: registry state, libraries and favorites intact | NOT RUN | |

Known not covered by the eject coordinator (do not expect these to pass, and
record what happens): image sessions, Humanize, server image and speech
models, an open still in flight, and an eject under a running move.

## E. Moves (requirements 7.x, 8.1)

| # | Check | Result | Notes |
|---|-------|--------|-------|
| E1 | Move a small text model from This Mac to the T7 with the Move sheet: required and available space shown, progress, then the row shows the T7; notes, tags, favorite and reasoning default stayed; a server favorite still resolves | NOT RUN | |
| E2 | Cancel a move in progress: the original is still in place and loads; the sheet says it was not changed | NOT RUN | |
| E3 | Escape and the window close gesture do nothing while a move runs | NOT RUN | |
| E4 | Move an image model and an audio model (a receipt-backed one) | NOT RUN | |
| E5 | The loaded model, a served model and a model in use by Humanize are refused with the reason; Load Chat Model, the menu-bar server window and a favorite cannot load a model that is mid-move | NOT RUN | |
| E6 | Unplug the destination during a move: the sheet names the drive, the original is kept and loads; a `.staging/move-*` folder may remain on the drive | NOT RUN | |
| E7 | Quit during a move: the original is kept; note the `.staging` leftover | NOT RUN | |
| E8 | Move back from the T7 to This Mac | NOT RUN | |

## F. Accessibility and appearance (DESIGN.md)

These were never run on a device. Rendered offscreen checks do not count.

| # | Check | Result | Notes |
|---|-------|--------|-------|
| F1 | VoiceOver reads each library row, the offer banner, the installed-row status, the action bar statements, the destination picker, the Move sheet (progress and outcome) and the scan-folder row in a sensible order | NOT RUN | |
| F2 | Increase Contrast, Reduce Transparency and Reduce Motion: every new view stays legible and nothing relies on motion | NOT RUN | |
| F3 | A custom theme and the high-contrast theme, light and dark | NOT RUN | |
| F4 | Full Keyboard Access: every new button and the Move sheet are reachable and operable | NOT RUN | |
| F5 | Disabled buttons look disabled in an active window (offscreen renders drew them undimmed) | NOT RUN | |
| F6 | Largest text size and an RTL language (Arabic or Hebrew) on the libraries section, banner, picker and Move sheet | NOT RUN | |
| F7 | A native speaker reads the new sentences in one language of your choice (the 21 translations have no native review) | NOT RUN | |

## Crashes and hangs

None recorded. Add a paragraph per incident.

## Summary

Not run. Fill this in last: how many rows passed, failed or were skipped, and
which failures block a release.
