# Consolidated review report: ZP-10 whiteboard canvas

Target: docs/designs/zcode-parity/ZP-10-whiteboard-canvas.md (full file, 163 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 8, Nit 8.

## Blocker

B1. Lines 17, 77, 136-139. Wrong pane host: `SplitChatPanes.swift` is the split-chat column with no pane targets or tabs; the right column is one slot arbitrated by `AppRightColumnClaimant`. Fix: make the whiteboard a new claimant, state its precedence position, and re-base the file list on that enum, RootView and the AppModel flag. [ZO-1, DC-1, S8-B1]

B2. Lines 38-40, 140-143. Composer path mis-framed: no pasted-image path exists in `PromptComposerView.swift`; `AppModel.addPromptAttachment(_:toChatID:)` already accepts image files; `@` panel has no file or design. Fix: drop item 4, export a PNG file and call the existing seam, name it, and cut the `@` goal. [ZO-2, DC-2, CO-2, CO-7, IC-2, S8-B2]

## Major

M1. Lines 49-53 vs 33, 71, 84. Scoping is an open question but already assumed everywhere; export target chat undefined with ZP-21 and ZP-23. Fix: record per workspace as a decision, state export goes to the focused chat. [ZO-4, IC-5, DC-9, ST-3, SY-3, CO-3, S8-M1]

M2. Lines 83-89, 144-148. Persistence is an undefined hybrid: "archive" has no antecedent, `AppJSONStore` is JSON only, "profile state root" unverified, no consistency rule between blob and metadata, corrupt-blob outcome unstated, quarantine test targets undefined behavior. Fix: one layout reusing an existing store pattern, blob as source of truth with rebuildable metadata, state corrupt-blob handling, one clause on flush at shutdown. [ZO-3, IC-1, CO-1, CO-8, DC-4, ST-7, S8-M2]

M3. Lines 36-37, 76-81, 117-122. Custom toolbar re-implements `PKToolPicker` with no justification; undo/redo wiring vague. Fix: adopt the picker or justify the custom toolbar by DESIGN.md in one sentence. [ZO-5, IC-4, DC-6, SY-2, S8-M3]

M4. Lines 17-20, 91-93, 115. PencilKit on macOS unverified (minimum macOS, mouse input, `drawingPolicy`), and off-main-thread export unjustified. Fix: first task is a short feasibility spike; cite the minimum OS; drop the off-main claim. [ZO-6, IC-3, DC-6, SY-2, CO-6, S8-M4]

M5. Lines 62-64, 85-86, 140-147. Export under-specified and no vision gate: "natural size" undefined, no pixel cap or background (transparent export unreadable in dark mode), empty-board export undefined, and images are accepted only when `visionIsActive` (`AppModel+Files.swift`). Fix: clamp to drawing bounds, cap pixels, opaque background, define Export when no vision model is active. [DC-3, DC-5, IC-2, ZO-7, CO-2, ST-6, SY-7, S8-M5]

M6. Lines 95-111. Alternatives are "Rejected by author" with no trade-off; missing ZP-11 shared drawing and export surface. Fix: one-line concrete rationale each and one line on reuse or separation from ZP-11. [ZO-9, ST-1, DC-7, SY-3, IC-7, CO-13, S8-M6]

M7. Whole document. No Risks and Mitigations section required by the standard. Fix: add a short section (blob loss or corruption, quit during write, unbounded growth, export size, PencilKit availability). Note S8 advises absorbing these in a line each rather than an eight-item section. [DC-8, ST-2]

M8. Lines 36-37 vs 55-72, 41-47. Goals (swatches, width, eraser, undo/redo/clear, naming) lack acceptance behavior; board create, rename, delete, switch and the naming scheme unstated; Non-Goals omit vector editing. Fix: one sentence on board management and naming, add vector editing to Non-Goals. [CO-3, ST-4, ST-6, DC-10]

## Nit

N1. Lines 118, 157. "21 languages" vs INDEX "ten localization parity gates". Fix: cite the gate by name. [ZO-8, IC-6, CO-10, DC-10, ST-9, SY-11, S8-N1]

N2. Lines 126-131. New `Whiteboard/` directory; convention unchecked. Fix: confirm against sibling feature folders. [ZO-10, S8-N2]

N3. Lines 149-150, 157-159. Mutation-check bullet is low altitude and mis-nested; manual-only end-to-end check has no observable. Fix: move to Verification Plan, add one automated check at the export handoff. [ST-8, CO-12, SY-11, S8-N3]

N4. Lines 3-11, 22-30, 43-44. Provenance gaps: no ZP-ID or size line; ZCode whiteboardStore.ts and Swift source links omitted; "stroke-based" claim uncited; ZCode, EGM, ZP-* unglossed. Fix: add links and one-line gloss. [ST-10, ST-11, SY-6, CO-9]

N5. Lines 22-29. Search claim has no scope or date; "first-class composer attachments" promotional. Fix: state path and date, describe the measurable behavior. [SY-1, CO-14]

N6. Lines 98-100, 108-109. "worse feel" subjective, palm rejection irrelevant to mouse, "sufficient" unsupported. Fix: use concrete costs. [SY-4, SY-5, CO-13]

N7. Lines 17-20, 115-121. Filler and vague wording ("native", "one-click", "strokes count", no debounce interval, accessibility for undo/redo). Fix: tighten, one clause on interval. [SY-8, SY-9, SY-10, CO-11]

N8. INDEX size note "medium". Fix: revisit after B1 if host work is real. [ZO-11, S8-N4]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-11 (zoom-out), IC-1 to IC-7 (inventive catalyst). Feeds B1, B2, M1 to M6, N1 to N4, N8.
- comprehension_report.md: CO-1 to CO-14. Feeds B2, M1, M2, M4, M5, M8, N-items.
- critic_report.md: DC-1 to DC-10, ST-1 to ST-11, SY-1 to SY-11. Feeds all Blockers and Majors, N1 to N7.
- intermediate_zoom_out_report.md: S8-B1, S8-B2, S8-M1 to S8-M6, S8-N1 to S8-N4. Confirms all earlier findings open; recommends about five targeted edits (host, attachment seam, scoping, persistence layout, alternatives rationale) and rejecting detail-level asks: a large Risks section, vault decision, debounce wiring, swatch and eraser specifics, board concurrency, extra CUJs, off-main export claim, jargon glossing.
- Merge notes: DC-1 and ZO-1 merged into B1; DC-2, ZO-2 and CO-7 into B2; DC-8 and ST-2 into M7 (S8 disagrees on scope, flagged there). Reviewer-quoted repo facts (AppRightColumnClaimant, addPromptAttachment, visionIsActive) were reported by reviewers; the orchestrator did not re-verify them independently.
