# Swift localization (the string catalog, its compile step, and its gates)

How the app is translated: one Apple String Catalog as the source of truth,
a script that compiles it because SwiftPM will not, and a parity test suite
that keeps the content complete. The model is lmstudio-ai/localization's
governance (automated key validation, documented contribution flow) applied
to the xcstrings format this package already builds from -- not its
per-language folder layout, for the reason at the bottom.

## The pipeline

`swift build` and `swift run` are the only way this app is ever built, and
SwiftPM does not compile a `.xcstrings` catalog: it would copy the raw JSON
into the bundle inert, and every language would be dead weight. There is no
Xcode project to provide the build phase, so `scripts/compile-strings.sh`
runs `xcstringstool` (the same private tool Xcode calls) ahead of the build:

    swift/TurboSparkApp/Localization/Localizable.xcstrings   (source of truth)
        |  scripts/compile-strings.sh  (make compile-strings; wipe + recompile)
        v
    Sources/TurboSparkApp/Resources/<lang>.lproj/            (compiled output)
        |  Package.swift  .process("Resources")
        v
    Bundle.module at runtime

The source catalog lives OUTSIDE `Resources/` on purpose: it is a build
input, not a runtime resource, and shipping it beside its own compiled
output would be two sources of truth. The script wipes stale `.lproj`
folders first, so a key or language removed from the catalog does not
survive in the bundle. `make swift-app`, `make swift-app-build` and
`make swift-app-release` depend on `make compile-strings`; when iterating
with SwiftPM directly, run `make compile-strings` yourself after editing the
catalog or the app builds against the previous output.

## The catalog

`Localizable.xcstrings` carries 1,169 keys in 21 languages: en, es, fr, de,
it, pt-BR, ru, ja, ko, zh-Hans, zh-Hant, ar, he, hi, nl, pl, tr, uk, sv,
vi, id. RTL support (ar, he) comes from `AppLanguage.isRTL` feeding
`.environment(\.layoutDirection)`. The key IS the English source string
(no namespaces, no generated symbols); the `en` entry's value is the key.
Plurals use the catalog's `variations.plural` and compile to a
`Localizable.stringsdict` per language. Every language is fully translated;
`LocalizationParityTests` is what holds that sentence true (below).

The in-app picker is `AppLanguage` (`Theme/AppLanguage.swift`), persisted
under `TurboSpark.language`, applied as `.environment(\.locale, ...)` at the
Window, Settings and MenuBarExtra scenes. The compiled `.lproj` directories
come out lowercased (`pt-br.lproj`); the resolution through
`Bundle.preferredLocalizations` is case-insensitive, and the tests rely on
that rather than on path spellings.

## How a string reaches the catalog

A `Text(_:)` literal compiles to a `LocalizedStringKey` that resolves
against `Bundle.main` by default, and in a SwiftPM build `Bundle.main`
carries no strings -- so a bare `Text("Save")` renders English under every
language, silently. The one mechanical rule of this surface:

    Text("English source text", bundle: .module)

Controls whose key-only initializers have no `bundle:` parameter take the
text through a `Text` label instead:

    CommandMenu(Text("View", bundle: .module))
    Button { model.run() } label: { Text("Generate Response", bundle: .module) }
    Picker(selection: $x) { ... } label: { Text("Text Size", bundle: .module) }
    .help(Text("Click to change greeting phrase", bundle: .module))
    .accessibilityHint(Text("Searches previous chats by keyword", bundle: .module))

The command menus in `App/TurboSparkApp.swift` are the worked example.

Deliberately NOT localized, and not defects:

- `Text(someVariable)` -- the `String` overload takes no localization at
  all; dynamic content renders as-is.
- Numeric readouts (`Text("\(position) / \(count)")`) -- nothing to
  translate; the scan's allowlist names them with reasons.
- `Text(condition ? "a" : "b")` -- compiles to the `String` overload via
  the ternary; write it as a `label:` closure of two `Text`s when it
  matters. This form is also INVISIBLE to the source-scan gate, which can
  only see calls whose argument starts with a literal.
- The app name (`Window("TurboSpark")`, `MenuBarExtra("TurboSpark", ...)`).
- The wider untranslated surface: `Label`/`Button`/`Picker`/`Section`/
  `TextField` titles outside the commands block (~300 Label call sites),
  and most of the few hundred remaining English literals. The catalog
  covers the audited `Text` surface plus the menu bar; growing it is
  ordinary feature work, gated by the tests below.

## The gates (`Tests/TurboSparkAppTests/LocalizationParityTests.swift`)

Seven tests, each mutation-checked to redden only its own case:

1. **Full key x language parity.** Every key has a `translated`, non-empty
   entry for every `AppLanguage`. Adding a UI string means translating it
   into all 21 languages in the same change; the gate makes "translate it
   later" impossible rather than aspirational.
2. **Format-specifier parity.** Plain (non-plural) values must carry
   exactly English's multiset of printf specifiers (`%lld`, `%@`,
   positional `%1$@`, `%%` respected). Plural values use the SUBSET rule:
   a translation may omit a placeholder but never invent one, because the
   call site supplies English's argument count -- an extra `%@` is the
   crash direction, an omitted `%lld` is often CORRECT (Arabic zero/one/
   two and Hebrew one spell the count out, per CLDR convention).
3. **Plural structure.** A key with English plural variations carries
   plural variations in every language (categories legitimately differ
   per CLDR: ru has one/few/many/other, ja only other). The en-side
   plural key count must stay positive, so the fixture cannot silently
   vanish.
4. **Key hygiene.** No two keys collide after `...` -> `...` (ellipsis)
   normalization. Case is deliberately not normalized: sentence-case and
   title-case English are different surfaces.
5. **Source-scan guard.** Every `Text("` literal call under `Sources/`
   passes `bundle: .module`, via a string-aware paren matcher that handles
   interpolations and multi-line calls (where a line-based grep false-
   positives). Allowlisted exceptions carry file + prefix + reason.
   `LocalizedStringKey("literal")` wrappers are banned outright; the
   variable-argument form is the legitimate dynamic case.
6. **greetings.json parity.** The greeting data must cover exactly the
   picker's language set, per greeting, in both directions.
7. **Bundled literal has a key.** Every NON-INTERPOLATED literal that
   passes `bundle: .module` resolves to a catalog key. Gate 5 is only
   half the rule and this is the other half: a bundled literal with no
   key renders its English source under all 21 languages, silently and
   indistinguishably from never having been converted. That is not a
   hypothesis -- it is how **416 keyless bundled literals** shipped (58%
   of the bundled surface, whole settings paragraphs among them) with
   the whole suite green, found 2026-09-09 by diffing the scan against
   the catalog and fixed in the same change. `\u{2022}`-style escapes
   are resolved before the lookup, because a literal spelled that way
   and one spelled with the character itself are the SAME key.

The two gates fail in opposite directions and neither substitutes for
the other: gate 5 catches a literal that never reached the catalog's
lookup path at all, gate 7 catches one that reaches it and finds
nothing. Mutating a `Text(verbatim: "-")` back to
`Text("-", bundle: .module)` reddens 7 alone; deleting a catalog key
reddens 7 alone; blanking a translation reddens 1 alone.

### The interpolated hole, and why it is a hole rather than an allowlist

`Text("Loading \(name)...", bundle: .module)` looks up a FORMAT STRING
(`Loading %@...`), not the source text, and the specifier follows the
interpolated expression's TYPE -- `%@` for a `String`, `%lld` for an
`Int` -- which a source scan cannot infer. Gate 7 therefore SKIPS
interpolated literals rather than listing them: guessing would either
invent keys nothing looks up or, worse, seed a wrong specifier into 21
translations, which gate 2 calls the crash direction. As of 2026-09-09
there are **139 such call sites**, 50 of them carrying real translatable
prose (`"%@" and its conversation history will be removed.`) and 89 bare
numeric readouts (`+\(adds)`, `Lines \(start)-\(end)`) that belong to
the deliberately-not-localized class above. Closing the first 50 means
deriving each key from the expression's type at the call site, one at a
time; it is open work, not a covered case.

`LocalizationTests` (the older suite) proves the compile chain itself:
catalog parses, `xcstringstool` output resolves a real translation through
`Bundle.module`, and every offered language compiled.

## Adding a string / a language

A string: add the key to the catalog with all 21 languages (en's value is
the key), reference it with `bundle: .module`, run `make compile-strings`
and the tests. If the string is duplicated under two spellings
(`...`/ellipsis), delete one. `bundle: .module` WITHOUT the key is the
failure gate 7 exists for, so the two halves land together or not at all.

A string with nothing to translate -- a separator glyph, a layout spacer,
a unit, a key name, a fragment of a code sample -- takes
`Text(verbatim:)` instead. That spelling leaves the localized surface
entirely (both source-scan gates skip it by construction), which is
honest, where a catalog key whose 21 values are all `-` is noise. The
2026-09-09 pass moved 24 such call sites across 13 files: `-`, `*`, `|`,
`= {`, `themePreview:`, `Aa`, `px`, `Cmd+N`, `P`, `\u{2022}` and four
empty spacers.

A language: add the `AppLanguage` case (raw value, native `label`,
`isRTL`), add the catalog column for all keys, AND add that language to
every greeting in `Resources/greetings.json` -- test 6 keys the greetings
to the picker's enum, so the two cannot drift. Recompile; the parity suite
checks the rest.

## greetings.json is a second system, on purpose

`Resources/greetings.json` (100 phrases x 21 languages, consumed by
`GreetingProvider`) is per-greeting CONTENT, not UI chrome: it is
runtime-selected, swappable, and the wrong shape for a UI catalog (2,100
keys of pleasantries). It stays outside the catalog and is pinned to the
picker by test 6 instead of being merged.

## Why not lmstudio's per-language folders

lmstudio-ai/localization is a standalone repo of one folder per language of
flat JSON files (`en/sidebar.json`, `en/settings.json`, ...) with CI
validating JSON syntax and key parity, built so translators without Macs
can contribute via PRs. Copying the layout would mean abandoning
`xcstringstool`, the plurals/stringsdict path, and Xcode-side tooling to
hand-roll a loader and a validator for a format the platform already
optimizes; the folder model's own costs are visible in its naming drift
(`gr` for Greek, mixed `pt_PT`/`zh-CN` styles). What was worth taking is
here: the validation suite, the contributor flow in this page, and the
rule that a language or key cannot silently half-ship.
