# Computer use

Lets a local vision model read and operate native macOS apps. Off by default.

## When the model sees the tools

All computer tools are hidden unless all of these hold:

- Computer use is turned on (`computerUse.enabled` in settings.json).
- The helper app is running and reports the current command surface
  (`ComputerUseProtocol.currentVersion`).
- The loaded model has active vision (`session.info.vision.active`).

A text-only model gets none of the tools, not only the screenshot one. The
tools join the main turn's tool capture only. Subagents never see them.

Tools: `computer_list_apps`, `computer_request_access`, `computer_snapshot`,
`computer_screenshot`, `computer_click`, `computer_type`, `computer_key`,
`computer_scroll`, `computer_open_app`.

## Settings

Settings > Computer Use holds the on/off switch, the Accessibility and Screen
Recording rows with their Allow buttons, and a note when the loaded model has
no vision. Each project's permissions add a Computer Use row (Ask by default,
Allow under the permissive presets, Deny under read-only).

## Grants

The model must call `computer_request_access` for each app. That call always
shows an approval card, in every permission mode and despite "always allow",
because it is the user's consent to desktop control. Grants live for the chat
and drop on model change, on the kill switch, and when the feature is turned
off.

Each app gets a tier from `ComputerUseAppPolicy`:

| Tier | Apps | Allowed |
| --- | --- | --- |
| refused | TurboSpark itself, System Settings, Keychain Access, password managers | nothing |
| read | browsers | snapshot, screenshot |
| click | terminals, code editors | plus single or double left click, scroll, open |
| full | everything else | plus right click, typing, keys |

TurboSpark is refused because its approval cards are in TurboSpark: an agent
that could click its own app could approve itself. Browsers are read-only
because the in-app browser tools enforce origin approvals that this surface
cannot.

Observing a granted app passes the project's Computer Use permission on Ask.
Input follows the permission (ask, allow, deny). Read-only mode denies all of
it.

## Targeting

`computer_snapshot` returns a bounded accessibility tree with element refs
(`e12`). Clicking a ref is the reliable path for small models. Pixel clicks
take `x`, `y` and the `screenshot_id` they were read from. The app maps them
through that screenshot's window frame and image size, and refuses an id that
is not the app's latest screenshot.

## The helper app

`TurboSpark.app/Contents/Helpers/TurboSpark Computer Use.app`
(`com.whit3rabbit.turbospark.computeruse`) owns the Accessibility and Screen
Recording grants. It exists because TurboSpark is ad-hoc signed: macOS ties an
ad-hoc app's grants to its code hash, so grants on TurboSpark itself would reset
every update.

- `scripts/make-app-bundle.sh` caches the signed helper by a hash of
  `Sources/ComputerUseKit`, `Sources/TurboSparkComputerUse`, the minimum macOS
  version and the signing identity, and reuses it while the hash holds
  (`COMPUTER_USE_HELPER_CACHE`, default
  `~/Library/Caches/com.whit3rabbit.turbospark/computer-use-helper`). A change
  to either source directory produces a new helper and users grant it again
  once. The helper's Info.plist carries a fixed version for the same reason.
- CI runners do not share that cache, so a release built on a fresh runner
  ships a new helper unless the release workflow restores the cache. Not wired
  yet.
- The app launches the helper with `NSWorkspace.openApplication`. A `Process`
  child would be attributed to TurboSpark as its responsible app, and the
  grants would land on the wrong app.

### Connection and trust

The app creates a 0700 session directory under its temporary directory and
writes a one-time token file (0600) there, then launches the helper with
`--session-dir`. The helper reads and deletes the token, listens on a Unix
socket in that directory, and serves one connection. It accepts the peer only
if the peer runs as the same user, has a valid code signature, is the app
bundle the helper ships inside (path from the socket's audit token), and
presents the token. The helper exits when the connection closes.

Known limit: with ad-hoc signing there is no developer identity to pin. A
same-user process that can rewrite and re-sign the whole TurboSpark.app can
pass the check. A Developer ID signature would close this.

### Prompts

Status reads never prompt (`AXIsProcessTrusted`,
`CGPreflightScreenCaptureAccess`). The helper calls the prompting APIs only on
a `requestPermission` request, which the app sends only from the user's Allow
click. `RecordingPermissionsTests` forbids the main app from calling
`AXIsProcessTrustedWithOptions`.

### Input safety

Before any input the helper brings the app to the front and checks it is
frontmost, and typing re-checks before each chunk. A click or scroll point must
land on a window that belongs to the app and is the topmost window there.
Screenshots capture only the app's front window, never the display.

### Kill switch

The helper shows a menu bar item while it runs, with Stop Agent Control. A
physical Escape press within 10 seconds of agent input also stops it (the
helper marks its own events so they do not count). After a stop every input is
refused, the app drops all grants, and only a newly approved grant re-arms
input.

## Testing

Test the permission flow in a bundled build (`make app-bundle`), not `swift
run`: the helper refuses to run outside an app bundle. Reset with:

```sh
tccutil reset Accessibility com.whit3rabbit.turbospark.computeruse
tccutil reset ScreenCapture com.whit3rabbit.turbospark.computeruse
```

Unit tests: `ComputerUseToolTests`, `ComputerUseHelperProtocolTests`.
