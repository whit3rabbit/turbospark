# Stage 8: Intermediate zoom out report (ZP-10 whiteboard canvas)

Reviewer: egm-zoom-out (S8), full file, with the three earlier reports as input. Completed. The document was not edited between stages, so all earlier findings remain open. Verdict: revise before approval. Context sufficiency passes.

## Open findings at zoom-out altitude

S8-B1 (Blocker). Lines 17, 77, 136-139. Wrong host: `SplitChatPanes.swift` renders split chats; the right column is arbitrated by `AppRightColumnClaimant` (Theme/AppRightColumnClaimant.swift:11). Fix: describe the whiteboard as a new claimant, state its precedence position, re-base file 3. [ZO-1, DC-1]

S8-B2 (Blocker). Lines 38-40, 140-143. Composer path mis-framed: `AppModel.addPromptAttachment(_:toChatID:)` (State/AppModel+Chat.swift:430) already exists. Drop item 4, hand an image file to that seam; cut the `@` panel goal or give it a file (cut recommended). [ZO-2, DC-2, CO-7]

S8-M1 (Major). Lines 49-53 vs 33, 71. Record "per workspace" as a decision; say export goes to the focused chat. [ZO-4, DC-9, ST-3, SY-3, IC-5]

S8-M2 (Major). Lines 83-89. Persistence inconsistent: undefined "archive", JSON-only `AppJSONStore`, unverified "profile state root". Fix: one layout reusing an existing store pattern, one source-of-truth rule (for example blob per board, metadata rebuildable), and a stated corrupt-blob outcome. [ZO-3, CO-1, DC-4, ST-7, IC-1]

S8-M3 (Major). Lines 36-37, 76-81. Custom toolbar re-implements `PKToolPicker`; adopt the picker or justify by DESIGN.md in one sentence. [ZO-5, IC-4]

S8-M4 (Major). Lines 91-93. PencilKit premise unstated: minimum macOS and mouse input. Make a short feasibility spike the first task. [ZO-6, IC-3, DC-6]

S8-M5 (Major). Lines 62-64, 140-143. Add one line for Export when no vision model is active (`visionIsActive` gating in `AppModel+Files.swift`) and one for export bounds, pixel cap and opaque background. [DC-3, DC-5, IC-2, ZO-7]

S8-M6 (Major). Lines 95-111. Alternatives not argued; replace "Rejected by author" with a one-line rationale; add the ZP-11 shared-surface question (reuse or explicitly separate). [ZO-9, ST-1, DC-7, SY-3, IC-7]

S8-N1 (Nit). Lines 118, 157. "21 languages" vs INDEX "ten localization parity gates": cite the gate by name. [ZO-8, IC-6, DC-10, ST-9, SY-11]

S8-N2 (Nit). Lines 126-131. New `Whiteboard/` directory: check repo convention. [ZO-10]

S8-N3 (Nit). Lines 149-150. Mutation-check bullet is low altitude: move or drop. [ST-8]

S8-N4 (Nit). INDEX size note "medium" understated if right-column host work is real. [ZO-11]

## Prior asks to reject as scope-adding or too low in altitude

- A new Risks and Limits section with eight or more items (DC-8, ST-2): absorb each risk in a line inside existing sections instead.
- Vault and encryption decision for blobs (DC-4 part): one-line note that blobs follow existing profile storage rules.
- Debounce interval, flush-on-quit wiring, `AppShutdownCoordinator` (CO-8, DC-4, SY-8): one clause "flushed on shutdown".
- Swatch values, eraser type, width range, tool-state scope, undo persistence, clear undoability (CO-4, CO-5, DC-6, ST-6).
- `PKDrawing.strokes.count`, `undoManager` lifecycle, `dataRepresentation()` naming (DC-6, SY-2, SY-5).
- A resolve-ordering test spec (DC-1): keep only the precedence statement.
- Full board lifecycle spec and naming scheme (CO-3, ST-6, ZO-5, IC-7): one sentence suffices.
- Board concurrency semantics (IC-5), extra failure, empty-board and board-management CUJs (DC-10).
- Workspace move or delete, orphan cleanup, corrupt-blob test expansion (DC-4, ZO-3).
- Non-macOS platform scope (CO-6): moot, app is macOS-only.
- Off-main-thread export claim (ZO-6, IC-2, DC-5): drop it rather than defend it.
- Non-Goal expansions beyond adding "vector editing" (ST-4).
- Jargon glossing, accessibility expansion, citation and style nits (CO-9 to CO-14, SY-1, SY-4, SY-6, SY-10, ST-10, ST-11).
- Exact-command rewrite of Verification or end-to-end composer tests (ZO-6, IC-6, DC-10): at most one automated check at the export-handoff seam.

Net direction: about five targeted edits (host, attachment seam, scoping, persistence layout, alternatives rationale); the rest waits for implementation.
