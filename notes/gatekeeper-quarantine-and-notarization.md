---
uuid: "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e12"
title: "Gatekeeper, quarantine, and notarization"
summary: "The app is ad-hoc signed, not notarized. brew install quarantines it and Gatekeeper refuses to open it. --no-quarantine or xattr -dr fixes it"
tags: ["release", "macos"]
source: "docs/RELEASE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## Why does the installed app say "cannot be verified"?

`make-app-bundle.sh` signs the app ad-hoc (`codesign --sign -`), which is
the minimum an arm64 binary needs to execute at all. It is not a Developer
ID signature, and there is no notarization step, because the repo holds no
signing identity.

Homebrew applies the quarantine attribute to cask downloads by default, so
`brew install --cask turbospark` puts an app in `/Applications` that
Gatekeeper refuses to open. Three ways around it, in the order to try:

```sh
brew install --cask --no-quarantine whit3rabbit/tap/turbospark
xattr -dr com.apple.quarantine /Applications/TurboSpark.app
# or: right-click the app in Finder and choose Open
```

## Don't

- Don't treat this as a bug to route around silently in a user-facing
  script. It's a documented state. Point people at one of the three
  workarounds above rather than reaching for a code signature they don't
  have.

## To fix it properly

In order: get a Developer ID Application certificate, put it in the CI
keychain, set `CODESIGN_IDENTITY` (the build script already reads it and
defaults to `-`), then add `xcrun notarytool submit --wait` plus
`xcrun stapler staple` after `make-dmg.sh`. Only the notarization pair is
new work. The signing seam already exists.
