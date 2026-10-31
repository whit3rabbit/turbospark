---
description: "tslab serve: the read-only TurboSpark Lab dashboard and JSON API over every ledger in the lab dir. What each page shows, the API routes, the security guards, and examples."
icon: chart-line
---

# The tslab dashboard (tslab serve)

`tslab serve` is a small local web server that renders every ledger in the lab
dir as HTML pages and a JSON API. It reads, it never writes: no POST route,
nothing written under the lab dir, and it never takes the machine lock. That
means you can leave it open beside a measurement and refresh while runs land.

<figure><img src="../.gitbook/assets/tslab-dashboard.png" alt="The TurboSpark Lab dashboard home page: the experiments table with filters and the compare tray"><figcaption><p>The experiments table at <code>/</code>.</p></figcaption></figure>

## What it shows

The dashboard measures nothing itself. It shows what tslab already wrote:

| You want to see | Page |
| --- | --- |
| Every experiment, its run count, verdicts, and last activity | `/` |
| One experiment: summary, notes, verdicts, sweeps, and the report charts | `/experiment/{name}` |
| Several experiments' series on shared axes | `/compare?exp=a,b` |
| Every paired verdict (WIN, LOSS, NOISE, INCONCLUSIVE) with its effect and interval | `/verdicts` |
| Every sweep, one verdict per candidate, and which candidate was `best` | `/sweeps` |
| Each family profile beside the data this lab has for it | `/families`, `/family/{family}` |
| One ledger record as formatted JSON, with the worker's stderr tail for runs | `/record/{name}/{id}` |

Dirty runs show as hollow marks on the charts. They stay visible so you can see
how noisy the host was, but they never count toward a verdict.

{% hint style="warning" %}
A shared axis makes a difference visible, not significant. Two experiments'
bars on one compare chart were not paired with each other. The verdict board
under the charts holds the paired results. Quote those.
{% endhint %}

## Starting it

```sh
tslab serve [--port 7878] [--bind 127.0.0.1] [--lab DIR] [--open]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--port` | `7878` | TCP port |
| `--bind` | `127.0.0.1` | Listen address. Anything off loopback prints a warning (see below) |
| `--lab` | the lab dir | Serve a different results root, for example a copy from another Mac |
| `--open` | off | Open the home page in the default browser |

The lab dir is `$TURBOSPARK_LAB_DIR`, else `$TURBOSPARK_HOME/lab`, else
`~/.turbospark/lab`. Ctrl-C stops the server.

The index re-reads a ledger when its size or modification time changes, so a
page refresh picks up new runs without restarting.

## Examples

Open the dashboard for this machine's results:

```sh
tslab serve --open
```

Look at results copied from another Mac without mixing them into yours:

```sh
rsync -a other-mac:.turbospark/lab/ ~/lab-other/
tslab serve --lab ~/lab-other --port 7879
```

Compare two experiments in the browser:

```
http://127.0.0.1:7878/compare?exp=lfm3-prefill,lfm3-ffn
```

Every WIN for one family, from a script:

```sh
curl -s 'http://127.0.0.1:7878/api/verdicts?family=lfm3' \
  | jq '.verdicts[] | select(.verdict == "win") | {experiment, candidate, metric, median, pairs}'
```

Pull one experiment's plottable series as JSON:

```sh
curl -s 'http://127.0.0.1:7878/api/series?exp=qwen36-mma8&kind=kernel_curve' | jq '.series'
```

## JSON API

GET only. Every response is a JSON object stamped `"schema": "tslab/1"` with
the payload under a named key (`experiments`, `verdicts`, `sweeps`, `series`,
`records`, `record`, `profile`). This is not the CLI's `--json` envelope: there
is no `ok` or `result` on success. Errors come back as
`{"schema": "tslab/1", "ok": false, "error": ...}`.

| Route | Returns |
| --- | --- |
| `GET /api/experiments` | The lab dir, every experiment's summary, and per-kind totals |
| `GET /api/experiment/{name}` | One experiment's summary, notes, verdicts, sweeps, and profiles |
| `GET /api/ledger/{name}` | Every record line of one experiment |
| `GET /api/series?exp=a,b[&kind=K]` | Plottable series. `exp` is required, `kind` filters |
| `GET /api/verdicts[?family=&model=&metric=&exp=a,b]` | Verdict rows under those filters |
| `GET /api/sweeps[?family=&model=]` | Sweep rows under those filters |
| `GET /api/family/{family}` | The embedded profile plus that family's verdicts, sweeps, and profiles |
| `GET /api/record/{name}/{id}` | One record. Ids are `<unix_ms>-<pid>-<n>` |

## Self-contained pages

Pages are rendered on the server and fetch nothing from anywhere: no CDN, no
web fonts, the favicon is an inline data URI. The one inline script adds hover
tooltips, legend toggles, and the compare tray count. A page saved to disk
reads the same as the live one, which makes it safe to attach to an issue.

## Security

There is no authentication. The guards are about keeping the server local and
keeping a crafted ledger from reaching the browser or the filesystem:

- **DNS-rebind guard.** On a loopback bind, a request whose `Host` header does
  not name this machine (`localhost`, `127.0.0.1`, `[::1]`, with an optional
  port) is refused. A forged form such as `[::1]evil.example` is refused too.
- **Off-loopback bind.** `--bind 0.0.0.0` or a LAN address turns the Host check
  off and prints a warning. Anyone who can reach the port can read every
  ledger, including model paths. Use `tslab export --redact-paths` for a copy
  you plan to share instead.
- **Headers.** Every response carries `X-Content-Type-Options: nosniff` and a
  `default-src 'none'` Content Security Policy (with `img-src data:` for the
  favicon alone).
- **Ledger input.** Every ledger-sourced string is HTML-escaped. A run record's
  `run_id` is validated before the record page reads its stderr tail, and a
  run line's `record` pointer is followed only when it resolves inside the lab
  dir.

`tests/dash_pages.rs` holds a test per guard.

## Related pages

- [TurboSpark Lab (tslab)](tslab.md)
- [The tslab agent skill](tslab-skill.md)
- [lab (crate reference)](../crates/lab.md)
