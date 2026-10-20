---
uuid: "9810a22e-1d52-4e3c-a6c8-6bcc4d48e4c7"
title: "Bundling Nordwand Mono as a code face: what the release measures, and where it lands"
summary: "Nordwand Mono 1.3.1 (tywr/Nordwand-Mono; font under OFL 1.1, generator under MIT) is bundleable: Regular, Medium, SemiBold and Bold total 287 KiB at a 0.600 em advance. Measured against the bundled faces it is NOT narrower than JetBrains Mono, and it carries 312 glyphs with no box drawing, no math operators and no curly double quotes. Landing it means Resources/Fonts plus OFL-Nordwand.txt, two lists in AppFontCatalog, the assertion list in AppearanceSettingsTests, and the design-system tokens."
status: "active"
tags: ["swift", "turbosparkapp", "theme", "fonts", "appearance"]
source: "https://github.com/tywr/Nordwand-Mono release 1.3.1 NordwandMono-TTF.zip read with fontTools 4.60.2; compared against Sources/TurboSparkAppResources/Resources/Fonts"
created: "2026-10-20"
updated: "2026-10-20T19:19:38Z"
depends_on: ["1167d96c-1572-4b19-9c52-adfffe004814"]
---

# What the Nordwand Mono release measures

A candidate code face for the Appearance code picker. Every number is read
from the shipped binaries with fontTools, not from the README.

key: the license is split: LICENSE-FONT.txt is SIL OFL 1.1 for the font, LICENSE.txt is MIT for the Python generator
key: latest release is 1.3.1, published 2026-09-29, with OTF, TTF, WOFF and WOFF2 zips
key: the TTF zip holds roman weights 100 through 700 plus italics; the app bundles 4 faces per family and no italics, so Regular, Medium, SemiBold and Bold is the matching set
key: those four faces total 287 KiB, cheaper than every four-face family in the bundle (Geist Mono 584, Source Code Pro 810, JetBrains Mono 1077, Fira Code 1169)
key: advance width is 0.600 em on every glyph, identical to JetBrains Mono, Source Code Pro and Geist Mono; the README's "compact" claim and config.example.yml's `width: 348` describe the drawn body, not the shipped advance

## Coverage is the real trade

key: 312 glyphs and 300 codepoints, against Hack 1573/1548, JetBrains Mono 1743/1363, Source Code Pro 1568/1335, Iosevka 48725/7758
key: zero box drawing (U+2500 to U+257F), zero powerline, zero math operators, zero arrows beyond U+2192, and no curly double quotes (only U+2019)
key: no CJK and no Arabic; Cyrillic is complete at 104 codepoints
key: GSUB carries ccmp, liga (-> and => only), ss02 and the cv01/cv03/cv11/cv12 alternative sets; there is no GDEF table
key: name ID 1 splits per weight as "Nordwand Mono Medium" with name ID 16 "Nordwand Mono", the shape every other bundled family already has

## Where it lands

key: the four TTFs in Sources/TurboSparkAppResources/Resources/Fonts as NordwandMono-Regular.ttf, -Medium, -SemiBold, -Bold
key: OFL-Nordwand.txt beside them, matching OFL-JetBrainsMono.txt; OFL 1.1 clause 2 wants the license in each distributed copy
key: AppFontCatalog.bundledFamilies and offeredCodeFamilies in Theme/AppFontCatalog.swift, two list edits
key: testBundledFamiliesSurviveTheInstalledFilter in AppearanceSettingsTests, whose `code.contains(...)` lines name one family at a time
key: docs/design-system/tokens.json `type.fonts`, then `python3 docs/design-system/tools/derive_tokens.py` for tokens.css, plus README.md and assets.md

## Don't

- Don't ship it because the README says compact. At 0.600 em it is the
  same cell as JetBrains Mono, so it buys no characters per line.
- Don't skip the AppearanceSettingsTests assertion list. A family bundled
  but not asserted passes with no signal that anything is missing.
- Don't promise CJK, box-drawing, math or curly-quote coverage. The app is
  localized to 21 languages and every one of those falls back mid-line.
