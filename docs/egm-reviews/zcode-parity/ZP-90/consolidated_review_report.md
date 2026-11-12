# Consolidated review report: ZP-90 excluded and decision findings

Target: docs/designs/zcode-parity/ZP-90-excluded-and-decision-findings.md (full file, 174 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only. All reviewers completed; none failed. Reviewers hit "Concurrent subagent limit reached" several times and were re-run one at a time; no reviewer is missing.
Applicability: ZP-90 is a decision log, not a feature design. Problem Statement maps to the Overview; Architecture, CUJs, NFRs, Goals and Verification Plan do not apply and were not scored. The Alternatives Rule and accuracy and consistency checks were applied.
Unverifiable here: `.kiro/steering/roadmap.md` and `.kiro/specs/desktop-shell-hardening` are absent from this worktree, so the "approved policy" and "log export and retention" claims could not be checked.
Source IDs: ZO, IC = zoom_out_report; CO = comprehension_report; DC, ST, SY = critic_report (design, structural, style); S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 0, Major 10, Nit 10.

## Blocker
None. ST-1 (dead roadmap link) was rated Blocker by the structural reviewer and downgraded in Stage 8 to Nit because the file is only missing in this worktree.

## Major

M1. L91-111 (also 3, 35, 162). Section 7.1 still says "author's call... does not pre-decide" and suggests "chat-only, tools local-only", contradicting the table, Overview, log and ZP-22. Fix: replace with a short "Adopted as ZP-22, API keys only; open questions live there" entry. [ZO-1, ZO-2, IC-1, CO-1, CO-6, DC-1, ST-2, SY-1, SY-5, S8-1]
M2. L12-24, 28-41 (INDEX L68-70). "Twelve... nine... one" accounts for 10; "not portable" is false for 7.1; "nine under policy" overstates since 1.4, 1.10, 3.6 rest on security or ownership judgment; "decision-ready" line is false. Fix: state the breakdown once so it sums to 12, mark judgment exclusions, drop the line. [ZO-3, ZO-12, IC-9, CO-2, DC-2, DC-11, ST-8, SY-7, SY-15, S8-2]
M3. L32 vs 68-76. 3.6 table reason "hosted bot backends" contradicts the body ("feasible client-side"). Fix: relabel as inbound control channel needing its own security design. [IC-2, DC-2, SY-4, S8-3]
M4. L160-170 (also 75). Reopen list omits 1.10, 3.6, 7.11; names "accounts" despite 7.2 "never"; 7.3/7.4/7.12 need a 7.2 reversal first. Fix: one sentence covering these. [ZO-5, IC-4, IC-7, CO-4, DC-3, SY-2, SY-8, S8-4]
M5. L16-18, 160-170. "No cloud services" policy not reconciled with ZP-22 hosted APIs. Fix: one sentence scoping policy to TurboSpark-operated services and accounts. [ZO-4, DC-8, S8-5]
M6. L46-56. 1.4 "Bonjour-advertised loopback API" is self-contradictory; the Tailscale bind rule is unsourced and sits oddly with "loopback-only". Fix: say LAN-bound, cite the rule or soften. [CO-5, DC-5, SY-3, SY-17, IC-4, S8-6]
M7. L113-119, 162-163. 7.2 lacks a Recommendation line, ZCode detail and the ZP-22 key-half pointer; rests on an unlinked author decision; "account login of any kind" is broad. Fix: add decision line and pointer; consider "TurboSpark-hosted account login". [ZO-6, IC-6, CO-7, DC-7, ST-3, ST-4, SY-14, S8-7]
M8. L78-84. 6.5 lumps SSH, WSL, Docker; trust-boundary argument covers SSH only; sandbox claim uncited. Fix: one clause on Docker; cite. [ZO-8, IC-5, CO-11, DC-4, SY-16, S8-8]
M9. L58-66. 1.10 omits that ZCode's gate fails open; "Sparkle already delivers security updates" unsourced; Sparkle critical-update flag not weighed. Fix: one sentence. [ZO-7, IC-3, DC-6, SY-10, S8-9]
M10. L136-145. 7.9 half "already specced" is unverifiable here; redaction not tied to ZP-22 CUJ 4 key redaction; ZCode redaction source and owner unnamed. Fix: verify in main checkout, link ZP-22, name owner. [ZO-10, IC-8, CO-9, DC-9, SY-6, S8-11]

## Nit

N1. L16, 80. Roadmap link (and desktop-shell-hardening) does not resolve in this worktree; same path convention as INDEX. Fix: confirm in main checkout; optionally quote the policy categories inline. [ST-1 downgraded, ST-9]
N2. L7, 164-166. Status "Draft" beside final decisions; log entries for 2.1, 6.12, 8.5 are outside the 12; ZP-15/16 unlinked; "Different" undefined. Fix: status "Decided"; label as related decisions; link. [ZO-11, CO-12, ST-5, ST-10, SY-13, S8-10]
N3. L3, 14, 47, 115, 122. ZCode, Missing, Different, Z.AI undefined. Fix: one-sentence note pointing to ZCODE_COMPARISON.md. [CO-3]
N4. L128-134. 7.4 ZP-22 usage ledger and follow-up unconfirmed; no Recommendation bullet. Fix: link ZP-22 section, add bullet. [CO-8, SY-11, SY-14]
N5. L121-126, 154-158. 7.3 and 7.12 "no adaptation" thinly argued. Fix: one sentence each (optional). [DC-10]
N6. L55-56, 74, 98, 124. Subjective wording ("honest transport", "parity checkbox", "real user value", "building a business"). Fix: neutral wording. [SY-12]
N7. L3, 28-41. Table shows numbers only, vocabulary inconsistent. Fix: titles and fixed verdict vocabulary (optional). [ST-6, ST-7]
N8. L147-152. 7.11 profile export unlinked, "excluded category" unnamed, tied to 6.5. Fix: link or describe. [CO-10, SY-9, DC-12, IC-7]
N9. L96-102. "server-client knowledge" vague; "product-identity change" has no ZP-22 task. Fix: moot once M1 applied. [ZO-9, ST-11, S8-12]
N10. L82. Permission ladder and sandbox designs unlinked. Fix: link. [ST-9, CO-11]

## Stage 8 triage (advisory)
Stage 8 recommends a minimum edit set: M1, M2, M3, M4, M5, M6, M7 and N2, with M8, M9, N4 as optional one-clause edits. It recommends rejecting scope-adding asks: splitting 1.4/6.6, 3.6, 6.5, 1.10 into new sub-decisions (IC-2 to IC-5), per-finding reopen tables, ZCode preamble, anchors and vocabulary polish, extra 7.3/7.12 rationale, Alternatives Rule rationale for 7.2.

## Traceability
- zoom_out_report.md: ZO-1 to ZO-12, IC-1 to IC-9. Feeds M1-M10, N1-N9.
- comprehension_report.md: CO-1 to CO-12.
- critic_report.md: DC-1 to DC-12, ST-1 to ST-11, SY-1 to SY-17.
- intermediate_zoom_out_report.md: S8-1 to S8-12 plus the ST-1 downgrade.
