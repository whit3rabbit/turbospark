# Automatic app updates

TurboSpark uses Sparkle 2 to detect and install macOS app updates. This page
owns the runtime behavior and its boundary with the release pipeline. The
release key bootstrap and the real two-version smoke remain in
[RELEASE.md](../../docs/RELEASE.md#in-app-updates-sparkle-fed-from-github-releases).

## What the app does

The updater starts once from `TurboSparkApp.init`. It starts only when
`Bundle.main` is a real `.app`; `swift run` produces a bare executable and
leaves update controls disabled. The implementation is
[SparkleUpdateController.swift](../TurboSparkApp/Sources/TurboSparkApp/App/Updates/SparkleUpdateController.swift).

The release bundle sets these keys in `Info.plist`:

- `SUEnableAutomaticChecks = true`: a fresh install checks by default and
  does not show Sparkle's permission prompt. The user can turn checks off in
  General settings. Sparkle persists that choice in the app's preferences.
- `SUFeedURL`: the `appcast.xml` asset on the latest GitHub Release.
- `SUPublicEDKey`: the public EdDSA key used to verify the downloaded DMG.

Starting the app schedules the next automatic check. Sparkle runs it at once
only when the saved interval is due. A user can force a check from the app
menu or General settings. Both controls observe Sparkle's
`canCheckForUpdates` value and stay disabled while a check or download owns
the updater.

Automatic checking is detection, not unattended installation. The bundle
does not set `SUAutomaticallyUpdate`, so a fresh install does not download or
install an update in the background. The standard Sparkle UI asks the user
before it downloads and relaunches the app.

## Detection path

1. `SPUStandardUpdaterController` starts and reads the feed URL. A local
   `TURBOSPARK_UPDATE_FEED_URL` value overrides `SUFeedURL`; otherwise the
   delegate returns `nil` and Sparkle uses the bundle value.
2. Sparkle downloads `appcast.xml`. The feed contains the current release
   DMG at a version-specific GitHub URL.
3. Sparkle compares the appcast `sparkle:version` with the installed
   `CFBundleVersion`. `CFBundleShortVersionString` is the version shown in the
   UI. The bundle script stamps both with the workspace version.
4. An automatic check with no newer item stays quiet. A manual check reports
   that the app is current. A newer valid item opens the standard Sparkle
   update UI.

The feed URL uses `releases/latest/download/appcast.xml`. Every published
release must therefore contain `appcast.xml`. A missing asset is a 404 for
all installed apps, not a valid no-update response. `release.yml` fails the
macOS build when `SPARKLE_ED_PRIVATE_KEY` is absent so the previous latest
release and its feed remain active.

## Trust and packaging

[`make-app-bundle.sh`](../../scripts/make-app-bundle.sh) copies
`Sparkle.framework`, writes the three update keys, signs Sparkle's nested
helpers from the inside out, then verifies the complete app bundle. The app
is currently ad-hoc signed, so Apple identity matching has no stable Team ID.
Sparkle accepts the update because the DMG has a valid EdDSA signature from
the existing bundle's key. The extracted app's ad-hoc code signature must
still be structurally valid.

[`make-sparkle-appcast.sh`](../../scripts/make-sparkle-appcast.sh) runs
Sparkle's `generate_appcast` over the release DMG. It refuses to write an
unsigned feed and checks that the enclosure uses the expected download URL.
The appcast tool version must stay aligned with the Sparkle version locked in
`Package.resolved`; both are currently 2.10.0.

The feed override does not replace signature verification. It exists for the
localhost smoke. `TURBOSPARK_UPDATE_SILENT=1` activates the unattended user
driver only when the feed override is also present. Normal launches always
use Sparkle's standard user driver.

## Verification

Run the focused controller test from the app package:

```sh
cd swift/TurboSparkApp
swift test --filter SparkleUpdateControllerTests
```

Check the release scripts and build the actual bundle from the repository
root:

```sh
bash -n scripts/make-app-bundle.sh scripts/make-dmg.sh \
  scripts/make-sparkle-appcast.sh
make app-bundle
make dmg
```

These checks prove compilation, framework presence, nested signature
validity, and the mounted DMG layout. They do not prove an installed app can
replace and relaunch itself. Use the two-version smoke in
[RELEASE.md](../../docs/RELEASE.md#in-app-updates-sparkle-fed-from-github-releases)
for that gate. The smoke log at `/tmp/turbospark-update-smoke.log` must retain
the detection, download, extraction, installation, and relaunch events.

## Failure checks

- Both manual buttons are disabled in a `swift run` build by design.
- A disabled button in a bundled app usually means Sparkle already has a
  check, download, prompt, or installer session in progress.
- A 404 for `appcast.xml` means the latest GitHub Release is missing the feed
  asset. Fix the release; do not treat it as no update.
- An EdDSA validation error means the DMG and appcast were not signed by the
  key matching `SUPublicEDKey`. Do not bypass the check or replace the public
  key in only the new bundle.
- A launch or installer connection failure after extraction requires the
  bundle and DMG gates above, then the real two-version smoke. A successful
  Swift build alone does not cover Sparkle's nested helpers.
