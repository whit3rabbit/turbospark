# Stage 7: Critic report, ZP-90

Reviewers egm-design (standard feature-spec-standard), egm-structural (feature-spec-standard), egm-style all completed, one round, run one at a time after the concurrent limit error. The standard applies only partly to a decision log: Problem Statement is the Overview; Architecture, CUJs, NFRs, Goals, Verification Plan do not apply. Alternatives Rule applied to rejection rationales.

## Design (DC)
DC-1 Major, L3, 91-111. 7.1 stale, contradicts adoption and ZP-22 (tools, identity-docs claim with no ZP-22 task). Fix: "Adopted as ZP-22" pointer.
DC-2 Major, L15-18, 28-41. "Nine under policy" wrong by own reasoning; 1.4, 1.10, 3.6 are judgment calls; 3.6 table label "hosted bot backends" contradicted by body. Fix: split labels and fix 3.6 label.
DC-3 Major, L160-170. Reopen list incomplete (3.6, 1.4 Tailscale path, 7.11 with 6.5, 1.10). Fix: per-finding trigger.
DC-4 Major, L78-84. 6.5 lumps SSH, WSL, Docker; sandbox claim unsourced. Fix: split per transport.
DC-5 Major, L47-52. 1.4: "Bonjour-advertised loopback API" self-contradictory; Tailscale rule unsourced (comparison says 127.0.0.1 only). Fix: say LAN-bound, cite rule or delete.
DC-6 Major, L58-66. 1.10 omits that ZCode's check fails open; Sparkle critical-update flag not evaluated; "Sparkle already delivers security updates" unsupported. Fix: state fail-open, evaluate Sparkle flag.
DC-7 Major, L113-119, 162-163. 7.2 rests on unlinked author decision (Alternatives Rule); scope vs HF token, MCP OAuth undefined. Fix: link record, add scope sentence.
DC-8 Major, L13-24. 7.1 adoption not reconciled with "no cloud services" policy. Fix: define policy as TurboSpark-operated services; name roadmap edit.
DC-9 Major, L136-145. 7.9 spec unverifiable; redaction not tied to ZP-22 CUJ 4. Fix: verify, require provider-key redaction.
DC-10 Nit, L121-126, 154-158. 7.3/7.12 "no adaptation" thinly argued. Fix: one sentence each.
DC-11 Nit, L22-24, 160-168. "Decision-ready" self-certification false for 7.1; log has out-of-scope entries. Fix: remove; label extras.
DC-12 Nit, L49-52, 147-152. Profile export uncited; no ZCode source links. Fix: add references.

## Structural (ST)
ST-1 Blocker (reviewer rating; see consolidation note), L16, 80. Roadmap link dead in this worktree; policy premise unverifiable. Fix: restore file or quote policy inline.
ST-2 Major, L91-111 vs 28-36, 162. 7.1 stale, no ZP-22 link. Fix: rewrite as adoption record.
ST-3 Major, L20, 36, 115-119, 163. Alternatives Rule: 7.2 cites only "author decision". Fix: link source, add rationale.
ST-4 Major, L113-119. 7.2 lacks Recommendation line and ZCode description. Fix: add.
ST-5 Nit, L162-166. Decision log beyond scope; ZP-15/16 unlinked. Fix: link, relabel.
ST-6 Nit, L3, 28-41. Findings numbered only, table has no titles or anchors. Fix: add titles and anchors.
ST-7 Nit, L28-41. Verdict/Decision vocabulary inconsistent. Fix: fixed vocabulary.
ST-8 Nit, L12-24. 12/9/1 arithmetic accounts for 10. Fix: state 9+1+1+1.
ST-9 Nit, L55, 82, 131, 139, 143, 150. Plain-text references need links. Fix: link.
ST-10 Nit, L7, 162-170. Status "Draft" vs final. Fix: "Decided".
ST-11 Nit, L98. "server-client knowledge" unclear. Fix: delete.
Link check: ../../ZCODE_COMPARISON.md, ZP-22, ZP-23 resolve; ../../../.kiro/steering/roadmap.md does not resolve here.

## Style (SY)
SY-1 Major, L110-111 vs 18-20, 35, 162. 7.1 stale. Fix: adopted wording.
SY-2 Major, L115-117 vs 167-170. "Permanent no accounts" vs reopen list naming accounts. Fix: require 7.2 reversal first.
SY-3 Major, L48-53. "loopback-only" vs "loopback or Tailscale"; Bonjour loopback. Fix: cite rule, say LAN-bound.
SY-4 Major, L70-75 vs 32. 3.6 reason mismatch; missing from reopen list. Fix: align, add.
SY-5 Major, L94-96, 98, 105. "Nothing calls a remote API", "server-client knowledge", "cover the space" unsupported. Fix: cite or drop.
SY-6 Major, L139-146. 7.9 redaction ambiguity; spec unlinked, wording differs from INDEX. Fix: cite, link.
SY-7 Major, L23-24. "Fabricated design... decision-ready" disclaimer. Fix: drop.
SY-8 Major, L167-170. Reopen list silent on 1.10, 3.6, 7.11. Fix: add or "others stay closed".
SY-9 Minor, L148-151. Profile export unlinked; "weak" subjective. Fix: link, state gap.
SY-10 Minor, L60-65. Sparkle claim unsourced. Fix: cite config.
SY-11 Minor, L130-134. ZP-22 ledger/follow-up refs unverified; "honest" subjective. Fix: confirm.
SY-12 Minor, L55-56, 74, 98, 124. Rhetoric ("honest transport", "parity checkbox", "real user value", "building a business"). Fix: neutral wording.
SY-13 Minor, L164-166. Out-of-scope entries; "Different" undefined. Fix: move or mark.
SY-14 Nit, L113-119, 128-134. 7.2 and 7.4 lack Recommendation bullet; 7.2 heading promises uncovered content. Fix: add, trim.
SY-15 Nit, L14-16, 21-24. Overview count implicit; per-entry promise overreaches. Fix: explicit count.
SY-16 Nit, L78-83 vs 33. WSL phrasing differs; ladder claim uncited. Fix: unify, cite.
SY-17 Nit, L48. "live somewhere" vague. Fix: "hosted".
Repo rules (ASCII, no emojis, no em dashes) pass.
