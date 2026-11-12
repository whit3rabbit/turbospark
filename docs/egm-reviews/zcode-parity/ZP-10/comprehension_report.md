# Stage 6: Comprehension report (ZP-10 whiteboard canvas)

Reviewer: egm-comprehension (CO), read the document alone. Completed. No Blockers.

## What this proposes

Add a "whiteboard" tab to the app's side pane. Each workspace gets a list of drawing boards, each with an auto-generated unique title. The user draws with pen or eraser, six color swatches and a width slider, and can undo, redo or clear. A one-click "Export to chat" renders the board to a PNG and attaches it to the chat composer as an ordinary image attachment. Drawing uses Apple's PencilKit on macOS. Boards persist across closing the pane and restarting the app. A new store keeps JSON metadata via "AppJSONStore" and the drawing data as separate files. Agent-side reading of the canvas is out of scope.

## How the existing system works, per this document

The app is a Swift/macOS app with a side pane that routes to different "pane targets" (SplitChatPanes.swift). It has a chat composer that already accepts image attachments, for example by paste, and an "@ attachment panel". It has a profile state root and an "AppJSONStore" persistence helper. It has design rules (DESIGN.md, a "token ratchet", `make design-check`) and a 21-language localization catalog. The document says no drawing canvas exists today. It does not explain how any of these pieces work internally.

## Findings

CO-1 (Major). Lines 85-87, 144-148. "AppJSONStore" is undefined; "the archive" (line 87) has no antecedent; "quarantine behavior on corrupt metadata" appears only in the test list. Behavior on corrupt metadata or a missing blob is unspecified.

CO-2 (Major). Lines 62-64, 85-86, 115-116, 141-143, 147. Export is unclear: "PNG for export" sits in the stroke-data line as if PNGs are stored; no statement on whether the PNG is persisted, in memory or a temp file; hand-off to the composer (callback, shared state, file URL) is not described; "same path as a pasted image" is not described; "natural size" is undefined; the PNG-dimensions test has no expected value.

CO-3 (Major). Lines 17, 35-36, 51-54, 76-81. Board lifecycle (create, rename, delete, switch) and the auto-naming pattern and collision scope are unspecified. Scoping is an open question yet Goals and CUJ 2 treat per workspace as settled.

CO-4 (Major). Lines 36, 91-93. "Undo/redo ride the canvas's coordinator" is vague: does undo history persist across restart, is clear undoable, does clear replace the stored blob immediately through the debounced write.

CO-5 (Major). Lines 36, 76-81. The six colors are not listed; eraser type (PencilKit tool or custom) unstated; width range unstated; color and width per-board or global unstated.

CO-6 (Major). Lines 76-78, 91-93. Platform scope is unclear: "Apple Pencil" on macOS is unexplained; no statement of what non-macOS builds do.

CO-7 (Major). Lines 39-40. The "`@` attachment panel" goal has no CUJ, architecture element or target file.

CO-8 (Major). Lines 115-116. No debounce interval and no flush-on-quit or failure behavior, which risks data loss against CUJ 2.

CO-9 (Nit). Lines 3, 7. ZCode, "Finding 5.8", "ZP-*" and "EGM" are not glossed.

CO-10 (Nit). Lines 117-118, 156-157. DESIGN.md, "token ratchet", `make design-check`, `make swift-lib` are project jargon without a gloss.

CO-11 (Nit). Lines 119-121. Accessibility covers only a strokes count; VoiceOver and keyboard exposure of undo/redo/clear and shortcuts unstated.

CO-12 (Nit). Lines 149-150, 158-159. "Mutation checks" is unexplained jargon; end-to-end attachment path has only manual verification.

CO-13 (Nit). Lines 100-103, 109-111. Both rejections are "Rejected by author"; stability of PKDrawing serialization across macOS versions is not addressed.

CO-14 (Nit). Lines 22-26. The "only audio waveform views" search claim cannot be reproduced by the reader.

Section-level verdict: every section can be paraphrased. Weakest links are export to composer (CO-2) and persistence failure rules (CO-1, CO-8).
