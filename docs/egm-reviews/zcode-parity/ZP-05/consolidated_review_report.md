# Consolidated review report: ZP-05 builtin marketplace seeding and CDN zip source

Target: docs/designs/zcode-parity/ZP-05-builtin-marketplace-seeding.md (full file, 182 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO/IC = zoom_out_report (zoom-out, inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 9, Nit 11.
Note: S8 recommends downgrading B1 to Major and rejecting B2 as written if cdnZip is split out (see Major M1).

## Blocker

B1. Lines 116, 124 (and 109-125). Both rejections read "Rejected by author" with no link and thin reasons. Fix: state technical reasons (bundle is a signed read-only .app; writes fail, updates swap the target under ledger rows), drop the attribution; replace strawman Alternative B with the directory-source approach. [DC-B1, ST-M2, ST-M3, SY-M5, SY-N14, ZO-5; S8 says downgrade to Major]

B2. Lines 98-106, 129, 142-164. No zip extraction design or archive-safety requirement (zip-slip, symlinks, zip bomb, size cap); existing installer does directories and git only. Fix: split cdnZip out of v1, or add one line requiring extraction bounds and traversal rejection plus a named extractor. [DC-B2, IC-1, CO-6; S8 recommends not specifying the extractor in the design]

## Major

M1. Lines 42-46, 62-65, 98-102, 146-149. cdnZip has no named consumer (no first-party CDN, builtins ship in the bundle); manifest shape question is open yet assumed by the plan. Fix: ship seeding first, defer cdnZip or name a consumer; resolve the manifest shape. [ZO-1, IC-6, CO-4, DC-M6, ST-M5, SY-M4, S8]

M2. Lines 99-100. Downloader reuse points at Rust `crates/repack ranged_download`, not callable from Swift. Fix: name the Swift code reused (`SteeringVectorDownload.swift`, `ManagedAssetStore.swift`) or state a size-capped URLSession download with streaming sha256. [ZO-2, IC-1, DC-M4, SY-N4, CO-6]

M3. Lines 104-107, 142-145. Seeding bypasses the existing directory-source install and stage-move-ledger path. Fix: make the builtin row a directory-source marketplace over bundle resources, or explain why not. [ZO-3, S8]

M4. Lines 35-41, 94-97, 104-107. Seed marker in the ledger, suppression in settings.json: two stores, crash divergence, ledger schema undefined, per-profile scope unstated, no bundled-version-bump policy, lock location and stale-lock behavior undefined. Fix: one per-builtin ledger record (seeded, suppressed, version) with idempotent startup reconcile; state the version policy; justify or drop the file lock. [ZO-4, IC-3, IC-4, CO-1, CO-2, CO-7, DC-M3, SY-N5, S8]

M5. Lines 62-65, 80-82, 129-130. A digest in the same row as the URL only catches corruption; trust anchor, HTTPS, redirects and missing or malformed digest behavior unstated. Fix: reword the Security NFR to say what the digest does and does not defend against; one sentence each on HTTPS and malformed digests (only if cdnZip stays). [IC-2, DC-M1, CO-6, SY-M3]

M6. Lines 129-132. Whether builtin hooks still need SHA-256 approval, and whether seeded files are re-verified, unstated; "seeding never runs code" and "existing trust model" are unsupported claims. Fix: state the verified behavior and name the trust gate by file or type. [DC-M2, IC-7, SY-M2, CO-9]

M7. Lines 17-19, 36, 135-137, 146-153. `builtin` source kind has no payload, is Codable-persisted (downgrade risk), and may collide with names reserved in SWIFT_PLUGINS.md (lines 51, 118); an always-present empty row ships with no user value. Fix: keep builtin as an app-owned non-persisted row, confirm no name collision, hide the row when empty. [DC-M5, IC-5, IC-7, SY-M6, CO-5, CO-11, ZO-7]

M8. Lines 127-136. No Risks and Mitigations section. Fix: fold the main failure modes (partial download, corrupt resource, stale lock, ledger divergence) into the NFR section rather than a new section (S8 rejects a new section). [ST-M1]

M9. Lines 26-30, 71. The Problem overstates "consequences" for a feature that does not exist yet; CUJ 1 hard-codes "guide skills" while Open Question 1 says the mechanism is independent. Fix: reword as a requirement, cite SWIFT_PLUGINS.md at first claim, mark CUJ 1 illustrative. [SY-M1, SY-M4, CO-8]

## Nit

N1. Lines 3-7, 28, 52, 58-60. ZCode links point at mutable main, no commit pin (INDEX pins a60ba6a7 / v3.17.1); finding 4.9, ZP-18, ZP-20 unlinked. Fix: pin and link. [ST-N1, SY-N1, SY-N10, SY-N11]
N2. Lines 18-22. Overview vague and repeats Non-Goals; "same capability" imprecise. Fix: one sentence naming the deliverable. [SY-N2, SY-N3, ST-N2]
N3. Lines 147-149, 42. "digest verification data" redundant; "a new source kind" versus two kinds. Fix: align. [SY-N6]
N4. Lines 158-164. Mutation checks and the concurrent-seed test not in the listed cases or Verification Plan; cdnZip tests live in a seeder test file; partial-download refusal has no matching Goal. Fix: align lists. [CO-10, ST-M4, SY-N7, ZO-6]
N5. Lines 171-178. Manual step run-on; release gate and localization or design-check gates unnamed; no metrics. Fix: numbered steps, name the gates. [SY-N8, DC-N1, DC-N3]
N6. Lines 90-91, 104. Seeding off main actor and launch cost unstated. Fix: one sentence. [DC-N2]
N7. Lines 142-179. Inconsistent field labels (item 2, item 5). Fix: uniform labels. [ST-N3]
N8. INDEX consistency. No dependency note; INDEX sequences ZP-05 after its consumers ZP-20 and ZP-18. Fix: one Dependencies line or note a test fixture as first builtin. [ST-N4, ZO-7, IC-6]
N9. Lines 1-2. Title says "CDN zip source", prose uses `cdnZip`. Fix: unify. [SY-N9]
N10. Lines 5-9 terminology. Terms (seed lock, profile, official segment, ledger, trust gates) undefined. Fix: brief inline definitions. [CO-9]
N11. Lines 51-52. Long parenthetical; "Partial" needs a link. Fix: split. [SY-N10]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-7, IC-1 to IC-7. Feeds B1, B2, M1 to M7, N4, N8.
- comprehension_report.md: CO-1 to CO-11. Feeds B2, M1, M2, M4 to M7, M9, N4, N10.
- critic_report.md: DC-B1, DC-B2, DC-M1 to DC-M6, DC-N1 to DC-N3; ST-M1 to ST-M5, ST-N1 to ST-N4; SY-M1 to SY-M6, SY-N1 to SY-N11. Feeds all Blockers and Majors, N1 to N11. SY numbers follow the report's list order (SY-N1 = item 7 through SY-N11 = item 18 mapped by topic).
- intermediate_zoom_out_report.md: all Stage 5 findings still open. Accepts CO-1/2/3/5/7, DC-M1/M2, DC-B1 as Major. Rejects DC-B2 as a Blocker, DC-M4 detail, ST-M1 and ST-M3 as new sections, and most polish nits. Net recommendation: split cdnZip out and route seeding through the directory-source path with one ledger record.
