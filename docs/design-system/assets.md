# Design system assets

Nothing binary is copied into this folder. Every asset lives where the app
loads it from. `RES` below means
`swift/TurboSparkApp/Sources/TurboSparkAppResources/Resources`.

## Logos

| File | Use |
| --- | --- |
| `RES/AppIcon.png` (1024px), `RES/AppIcon.icns` | The app icon: the mascot riding a rocket on the night sky in a macOS squircle. Use it as is, at any size from 16px up. Never crop out the squircle, recolor it, or redraw it. Size variants are in `assets/icons/`. |
| `assets/banner-nobg.png` | The lockup: the mascot beside the TurboSpark wordmark, transparent background. Light grounds only, because the navy wordmark disappears on `page` in the dark themes. On a dark ground, set the app icon beside the name in the `title` style. |

`RES/Logos/` holds model-provider marks (Qwen, Anthropic, Cohere, and
others) for the model hub. They are third-party marks, not TurboSpark brand
assets. Never use them as TurboSpark branding.

## Mascot

The cyan flame. Raster art, one character per screen at most. Never use it as
a status indicator, and never recolor or redraw it.

| File | Screen |
| --- | --- |
| `RES/WelcomeCharacter.png` | Welcome screen, above the greeting (waving). |
| `RES/progress_character.png` | Task progress (riding the rocket). `RES/TaskProgress.png` is an identical fallback. |
| `RES/AgentsCharacter.png` | The AGENTS.md file screen (tuxedo, holding the sign). |
| `RES/MemoryCharacter.png` | The MEMORY.md file screen (glasses and books). |
| `RES/SoulCharacter.png` | The SOUL.md file screen (halo). |

The signs show file names, so they stay in English in every language. Give
each image an accessible description, or hide it from VoiceOver when it is
decorative.

## Tool icons

`RES/ToolIcons/`: 51 Lucide icons, pinned to revision 266fd1c (ISC, see
`LICENSE.txt`). Each ships as SVG and PNG. `manifest.json` records every
file's source URL and sha256.

They are 24px and single ink: stroke `currentColor`, width 2, round caps and
joins, no fill. Inline them so they take `ink-secondary` or `accent-ink`.
Inside an `<img>`, `currentColor` resolves to black.

They mark tool calls and agent capabilities. Interface chrome uses SF Symbols
instead. No emoji.

## Fonts

`RES/Fonts/`, registered at launch by `AppFontRegistrar`, each with its
license file beside it.

| Family | Role | In `tokens.json` |
| --- | --- | --- |
| System (SF Pro, New York) | Default interface and greeting faces. SF Mono is a code option and the system fallback | `sans`, `serif` stacks |
| Hack | Default code face, bundled | `hack`, first in the `mono` stack, `type.fonts` |
| Geist, Geist Mono | Bundled options and the web fallback | `sans`, `mono` stacks, `type.fonts` |
| Inter | Bundled interface option | `inter`, Appearance options group |
| JetBrains Mono, Fira Code, Cascadia Code, Source Code Pro | Bundled code options | one family each, Appearance options group |
| Iosevka | Bundled code option | Not carried (11 MB per file) |

`tokens.json` lists font files as `fonts/<file>`, the layout of the rendered
artifact. In this repository the same files are in `RES/Fonts/`, and
`tokens.css` points there.
