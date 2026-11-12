# Stage 5: Zoom out and inventive catalyst report, ZP-90

Target: docs/designs/zcode-parity/ZP-90-excluded-and-decision-findings.md (full file, 174 lines). Both reviewers completed. The zoom-out standard barely applies (decision log, no design or code); reviewers applied it as "is each exclusion justified and consistent". `.kiro/steering/roadmap.md` and `.kiro/specs/desktop-shell-hardening` do not exist in this worktree: flagged unverifiable, not wrong.

## Zoom-out (ZO)

ZO-1 Major, L91-111. Section 7.1 still says "author's call... does not pre-decide" while table (L35), Overview (L18-20) and Decision log (L162) say adopted as ZP-22. Fix: replace with a short adopted-as-ZP-22 entry pointing to ZP-22.
ZO-2 Major, L107-109. "Chat-only, tools local-only" and "compaction degrade" contradict ZP-22 (tools through the permission ladder, compaction unchanged). Fix: delete with ZO-1.
ZO-3 Major, L15-18, L80, table L30-32. "Nine under policy" overstates: 3.6, 1.10, 1.4 are security or ownership judgments, not roadmap categories. Fix: split verdicts into policy-excluded and judgment-excluded; fix count.
ZO-4 Major, L16-18, L160-170. ZP-22 sends prompts to hosted APIs; "no cloud services" policy is not reconciled. Fix: one sentence scoping the policy to project-operated services and accounts.
ZO-5 Major, L166-170. "Still reopenable" omits 1.10, 3.6, 7.11, and account-tied 7.3/7.4/7.12 also need a 7.2 reversal. Fix: per-finding reopen conditions.
ZO-6 Nit, L113-119. 7.2 lacks ZCode detail and a Recommendation line; API-key half delivered by ZP-22 not recorded in table. Fix: add one line.
ZO-7 Nit, L58-66. 1.10 omits that ZCode's gate fails open; Sparkle critical-update alternative unconsidered. Fix: one sentence.
ZO-8 Nit, L78-84. 6.5 lumps local Docker with SSH/WSL; trust-boundary argument covers SSH only. Fix: state Docker reasoning.
ZO-9 Nit, L96-102. "server-client knowledge" vague; "product-identity change" has no ZP-22 work item. Fix: drop or add task.
ZO-10 Nit, L139-145. 7.9 redaction pass and ZP-22 CUJ 4 redaction not cross-linked; owner unnamed. Fix: name owner, link ZP-22.
ZO-11 Nit, L1-9, L164-166. Status "Draft" beside final decisions; log entries for 2.1, 6.12, 8.5 are outside the 12 findings. Fix: status "Decided"; keep or drop out-of-scope entries.
ZO-12 Nit, L11-14. "Twelve ... not portable" wrong for 7.1. Fix: "twelve decided: nine excluded, one split, one adopted, one permanently excluded" (note this list itself double counts 7.2; reconcile).
Verified consistent: 23+5+12 = 40 in INDEX; 7.4 pointer matches ZP-22 non-goals; Tailscale bind claim; profile export exists; Sparkle claims.

## Inventive catalyst (IC)

IC-1 Major, L91-111. Same stale 7.1 section as ZO-1. Fix: pointer to ZP-22.
IC-2 Major, L68-76, L32. 3.6: table says "hosted bot backends", body says feasible client-side; real objection is inbound control. Fix: split: exclude inbound channel, consider outbound-only notifications (separate decision).
IC-3 Major, L58-66, L31. 1.10: false choice between hosted blocking gate and nothing; Sparkle appcast critical-update is advisory with no new endpoint. Fix: record as adapted half (verify attribute names).
IC-4 Major, L46-56, L86-89, L167-170. 1.4/6.6 reopen only via hosted infra, yet body calls Tailscale-bound server the honest transport; 6.6 "depends on 1.4" is weak. Fix: split hosted relay (excluded) from self-hosted Tailscale UI (deferred pending security design); add reopen trigger.
IC-5 Major, L78-84, L33. 6.5 rationale applies to SSH only; WSL non-macOS; Docker is local. Fix: split SSH/WSL from local containers.
IC-6 Nit to Major, L113-119, L162-163. 7.2 "account login of any kind" overbroad vs third-party OAuth (e.g. MCP). Fix: reword to no TurboSpark-hosted identity, or state MCP OAuth out of scope.
IC-7 Nit, L147-152. 7.11 depends on unresolved 6.5 Docker question. Fix: add reopen trigger.
IC-8 Nit, L128-134, L143-146. Missing ZP-22 cross-links (usage endpoint follow-up tracked once; redaction dependency). Fix: add.
IC-9 Nit, L14-18. Nine findings not listed by number; 1.10 and 3.6 not obviously in roadmap categories. Fix: list them and mark judgment ones.
