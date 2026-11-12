# Stage 8: Intermediate zoom out report, ZP-90

Reviewer egm-zoom-out completed. Verdict: sound in shape; one real defect (stale 7.1) and one count error drive most findings. About eight small edits make the log accurate. Most asked-for splits add scope.

S8-1 Major, L91-111. 7.1 stale vs table, Overview, log, INDEX. Fix: 3-4 line adopted-as-ZP-22 entry.
S8-2 Major, L12-18, 22-24. Count accounts for 10 not 12; "not portable" false for 7.1; "decision-ready" line false. Fix: restate counts once, drop the line.
S8-3 Major, L32 vs 68-76. 3.6 table label contradicts body. Fix: "inbound control channel needs its own security design". Do not add outbound-notification sub-decision.
S8-4 Major, L160-170. Reopen list omits 1.10, 3.6, 7.11; 7.3/7.4/7.12 need 7.2 reversal. Fix: one sentence; no trigger table.
S8-5 Major, L16-18, 160-170. Policy vs ZP-22 not reconciled. Fix: one sentence scoping policy to TurboSpark-operated services; do not claim a roadmap edit.
S8-6 Major, L46-56. 1.4 "Bonjour-advertised loopback API" and unsourced Tailscale rule. Fix: "LAN-bound API"; cite or soften. Reject hosted/self-hosted split.
S8-7 Minor, L113-119. 7.2 needs a decision line and ZP-22 key-half pointer; "TurboSpark-hosted account login" wording if author agrees. No MCP/HF scope widening.
S8-8 Minor, L78-84. 6.5 Docker: add one clause; no sub-decision split.
S8-9 Minor, L58-66. 1.10: note fail-open and Sparkle critical-update flag unevaluated; no new item.
S8-10 Minor, L7, 164-166. Status "Decided"; label 2.1/6.12/8.5 as related decisions; link ZP-15/ZP-16.
S8-11 Nit, L128-145. Link ZP-22 for 7.4 and provider-key redaction for 7.9; verify desktop-shell-hardening in main checkout.
S8-12 Nit, L98-102. Vague phrases moot once 7.1 rewritten.
ST-1 downgrade: the dead roadmap link is an artifact of this worktree lacking .kiro (INDEX uses the same path). Downgrade Blocker to Nit; confirm in main checkout; optionally quote the three policy categories inline.

Rejected as scope-adding: splits of 1.4/6.6, 3.6, 6.5, 1.10 (IC-2 to IC-5); per-finding reopen table (DC-3, ZO-5, IC-7); ZCode preamble (CO-3); anchors and vocabulary polish (ST-6, ST-7, ST-9, DC-12); rhetoric and extra 7.3/7.12 rationale (SY-12, DC-10); Alternatives Rule for 7.2 (ST-3, DC-7).
