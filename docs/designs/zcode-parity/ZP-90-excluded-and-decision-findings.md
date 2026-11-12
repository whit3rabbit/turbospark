# ZP-90: Excluded and decision-required findings

Findings 1.4, 1.10, 3.6, 6.5, 6.6, 7.1, 7.2, 7.3, 7.4, 7.9, 7.11, 7.12
from [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).

* Author: whit3rabbit
* Status: Draft (decision document)
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Twelve of the 40 Missing findings were not portable design work in the
plain sense: nine fall under the exclusion policy this repository already
approved ([roadmap](../../../.kiro/steering/roadmap.md): no cloud services
and accounts, no remote workspaces, no non-macOS rows) and one has its
portable half specced elsewhere. Author decisions 2026-11-11 resolved the
two judgment calls: remote providers are adopted as
[ZP-22](ZP-22-remote-providers.md), and cloud OAuth is excluded
permanently. This document records, per finding: what ZCode has, whether
a local-first adaptation exists, what infrastructure it would need, and
the decision. Nothing here is a fabricated design; each entry is
decision-ready material.

## Summary table

| Finding | Verdict | Decision |
|---|---|---|
| 1.4 mobile/web relay | excluded (cloud/pairing service) | do not adopt |
| 1.10 force-update kill switch | excluded (needs a hosted config) | do not adopt |
| 3.6 bots (Feishu/Telegram) | excluded (hosted bot backends) | do not adopt |
| 6.5 remote workspaces | excluded (remote category; WSL non-macOS) | do not adopt |
| 6.6 mobile/web remote UI | excluded (depends on 1.4) | do not adopt |
| 7.1 remote providers | adopted | [ZP-22](ZP-22-remote-providers.md), API keys only |
| 7.2 OAuth account login | excluded permanently | never cloud OAuth (author decision 2026-11-11) |
| 7.3 billing/purchase | excluded (commerce backend) | do not adopt |
| 7.4 quota display/reset | excluded (needs an entitlement server) | do not adopt |
| 7.9 feedback center | split | log half already specced; form half excluded |
| 7.11 cross-device sync | excluded (depends on 6.5) | do not adopt |
| 7.12 rewards/referral | excluded (marketing backend) | do not adopt |

## Per-finding analysis

### 1.4 Mobile/web remote control relay

ZCode's desktop runs a WebSocket relay so paired phone/web clients can
drive sessions and receive pushes. A faithful port requires a pairing
broker and push transport that live somewhere. A LAN-only adaptation
(Bonjour-advertised loopback API plus a pairing handshake) is
engineering-feasible but creates a remote control surface for a local
inference app whose security posture is loopback-only today; the server
binds loopback or a single verified Tailscale IPv4 by explicit rule.
* **Recommendation**: do not adopt. If remote control ever becomes a
  goal, the Tailscale-bound server is the honest transport, and it needs
  its own security review, not a parity port.

### 1.10 Remote force-update kill switch

ZCode reads a minimum version from a hosted client-config endpoint and
blocks the UI below it. TurboSpark has no hosted config endpoint; adding
one (GitHub raw file, or the update feed) makes updates remotely
blockable, which cuts against local-first ownership: the user's app
should not refuse to start because a remote file says so. The Sparkle
updater already delivers security updates.
* **Recommendation**: do not adopt.

### 3.6 Bots (Feishu/Telegram)

ZCode drives task flow from chat platforms through hosted bot backends
with per-bot credentials. Feasible client-side (Telegram's bot API is
plain HTTPS), but it exposes a control channel into a local agent from a
third-party platform, with credential storage and permission-escalation
questions that deserve their own design rather than a parity checkbox.
* **Recommendation**: do not adopt. Revisit only with a dedicated
  security design for the channel.

### 6.5 Remote workspaces (SSH / WSL / Docker)

Excluded category in the roadmap (remote workspaces; WSL additionally
Windows-only). SSH workspaces would also move the agent execution trust
boundary off-device, which the permission ladder and sandbox designs do
not model.
* **Recommendation**: do not adopt.

### 6.6 Mobile / web remote control UI

Entirely dependent on 1.4's relay. Excluded with it.
* **Recommendation**: do not adopt.

### 7.1 Remote model provider configuration

ZCode configures remote providers (API keys, endpoints, model lists).
TurboSpark serves and runs local weights and its "providers" are on-disk
model libraries; nothing in the app calls a remote chat API. This is the
one finding that is portable engineering with real user value (a
local-first app that can *optionally* call a hosted model when local
hardware is insufficient, using the app's own server-client knowledge).
Costs: a full client stack (streaming chat clients per API family),
context-window accounting for remote tokenizers (the exact local ring
would not apply), credential storage per the Keychain rules, and a
product-identity change recorded in docs and marketing surfaces.
* **Open questions if adopted**: which API families (Anthropic and
  OpenAI-compatible cover the space); whether remote models appear in
  the same composer picker (suggested: a separate "Remote" section with
  clear labeling); how compaction and window fitting degrade for remote
  models; whether agent tools (Bash, browser) are allowed against remote
  providers at first (suggested: chat-only, tools local-only, to keep
  the risk surface stable).
* **Recommendation**: author's call. Draft-ready if the answer is yes;
  ZP-90 deliberately does not pre-decide it.

### 7.2 Account login (OAuth) and provider family connection

ZCode is account-first with Z.AI/BigModel OAuth. Excluded permanently by
author decision 2026-11-11: the app never adds cloud OAuth, hosted
identity, or account login of any kind. Remote providers under ZP-22
connect exclusively through user-managed API keys stored per the
existing Keychain rules.

### 7.3 Coding plan billing, purchase, upgrade

Commerce surface for Z.ai plans. TurboSpark has no service, no plans,
nothing to bill. No local-first adaptation exists short of building a
business.
* **Recommendation**: do not adopt.

### 7.4 Plan quota display and quota reset

Server-driven entitlement metering; needs an entitlement server the app
does not have. Do not adopt. Note: ZP-22's usage ledger shows
provider-tagged token counts, which is the honest local analog; a live
quota view is possible only if a configured provider exposes a usage
endpoint, recorded as a ZP-22 follow-up rather than a goal.

### 7.9 Feedback center with log export

Split verdict. The portable half (desktop log export that zips app logs
with redaction, prefilling a GitHub issue) is already in scope of the
`desktop-shell-hardening` spec ("log export and retention"). The server
half (submit form, ticket list, screenshot upload) needs the hosted
backend the policy excludes.
* **Recommendation**: keep the log-export half in
  `desktop-shell-hardening`; do not adopt the hosted half. Ensure the
  spec's export includes the redaction pass ZCode documents.

### 7.11 Cross-device config sync (data sync)

Pushes skills/MCP/plugins to remote devices; depends on 6.5's remote
device fleet. A weak local analog (settings export/import bundle)
already exists via profile export. Excluded category.
* **Recommendation**: do not adopt.

### 7.12 Rewards and referral center

Marketing surface tied to the Z.ai account and rewards backend.
No local-first adaptation exists.
* **Recommendation**: do not adopt.

## Decision log

* 2026-11-11: 7.1 adopted as ZP-22 (upstream APIs, API keys only).
* 2026-11-11: 7.2 excluded permanently; the app never adds cloud OAuth.
* 2026-11-11: concurrent background chats (finding 2.1, Different)
  adopted as [ZP-23](ZP-23-concurrent-chats.md); finding 6.12 and 8.5
  deferrals confirmed reversed via ZP-15 and ZP-16.
* Still reopenable: a decision to build hosted infrastructure (accounts,
  commerce, relay) would reopen 7.3, 7.4, 7.12, 1.4, 6.6, and the hosted
  half of 7.9, each with its own security design; off-device agent
  execution would reopen 6.5 with a remote trust-boundary design.

--------------------------------------------------------------------------------

*Drafted with EGM*
