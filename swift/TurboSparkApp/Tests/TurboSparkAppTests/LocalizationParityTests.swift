import XCTest
@testable import TurboSparkApp

/// Parity and hygiene gates over the string catalog -- the part of the
/// lmstudio-ai/localization model worth copying (automated validation of
/// every language's keys) expressed against the xcstrings catalog this
/// package actually builds from. `LocalizationTests` proves the compile
/// chain runs; these tests prove the catalog's CONTENT stays complete,
/// well-formed, and reachable from the call sites.
///
/// All of them parse `Localization/Localizable.xcstrings` directly through
/// `#filePath` (same reasoning as `LocalizationTests.sourceCatalogURL`): the
/// source catalog is the thing translators edit, and a stale compiled output
/// must not be able to green a content check.
final class LocalizationParityTests: XCTestCase {
    private var sourceCatalogURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // Tests/TurboSparkAppTests/
            .deletingLastPathComponent()   // Tests/
            .deletingLastPathComponent()   // TurboSparkApp/
            .appendingPathComponent("Localization/Localizable.xcstrings")
    }

    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
    }

    /// Every language the in-app picker offers, as BCP-47 raw values.
    /// Compared case-insensitively against the catalog: the catalog stores
    /// `pt-BR`, the compiled output lowercases to `pt-br`, and neither
    /// spelling is a promise the other side can rely on.
    private var offeredLanguages: [String] {
        AppLanguage.allCases.filter { $0 != .system }.map { $0.rawValue }
    }

    private func loadCatalogStrings() throws -> [String: Any] {
        let data = try Data(contentsOf: sourceCatalogURL)
        let json = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try XCTUnwrap(json["strings"] as? [String: Any])
    }

    /// Every (state, value) pair reachable under one localization entry:
    /// a plain entry carries a `stringUnit`, a plural entry nests several
    /// under `variations`. Walks generically so a future `substitutions`
    /// block is covered without a format change here.
    private func stringUnits(_ node: Any) -> [(state: String, value: String)] {
        var out: [(state: String, value: String)] = []
        if let dict = node as? [String: Any] {
            if let unit = dict["stringUnit"] as? [String: Any] {
                out.append((unit["state"] as? String ?? "",
                            unit["value"] as? String ?? ""))
            }
            for (_, child) in dict where child is [String: Any] || child is [Any] {
                out.append(contentsOf: stringUnits(child))
            }
        } else if let array = node as? [Any] {
            for child in array {
                out.append(contentsOf: stringUnits(child))
            }
        }
        return out
    }

    private func localizationEntry(
        _ localizations: [String: Any], language: String
    ) -> [String: Any]? {
        let lowered = Dictionary(
            uniqueKeysWithValues: localizations.map { ($0.key.lowercased(), $0.value) })
        return lowered[language.lowercased()] as? [String: Any]
    }

    // MARK: - Format specifiers

    /// printf-style specifiers, positional forms included, length modifiers
    /// consumed so `%lld` is one specifier rather than `%` + a stray `l`.
    private static let specifierPattern =
        "%(?:\\d+\\$)?(?:ll|hh|l|h)?[@dDuUxXoOfeEgGcCsSp]"

    private func specifiers(in text: String) -> [String] {
        let stripped = text.replacingOccurrences(of: "%%", with: "")
        guard let regex = try? NSRegularExpression(pattern: Self.specifierPattern) else {
            return []
        }
        let range = NSRange(stripped.startIndex..., in: stripped)
        return regex.matches(in: stripped, range: range).compactMap {
            Range($0.range, in: stripped).map { String(stripped[$0]) }
        }.sorted()
    }

    /// True when `candidate` adds no specifier `english` lacks -- the crash
    /// direction is a translation carrying a placeholder the call site never
    /// supplies an argument for. Omission is checked separately per key kind.
    private func specifiers(_ candidate: [String], fitIn english: [String]) -> Bool {
        var available = english
        for specifier in candidate {
            guard let index = available.firstIndex(of: specifier) else { return false }
            available.remove(at: index)
        }
        return true
    }

    // MARK: 1. Full key x language parity

    func testEveryCatalogKeyIsTranslatedInEveryOfferedLanguage() throws {
        let strings = try loadCatalogStrings()
        var problems: [String] = []
        for (key, entry) in strings {
            let localizations = try XCTUnwrap(
                (entry as? [String: Any])?["localizations"] as? [String: Any],
                "key \(key) has no localizations block")
            for language in offeredLanguages {
                guard let loc = localizationEntry(localizations, language: language) else {
                    problems.append("\(key): missing \(language)")
                    continue
                }
                let units = stringUnits(loc)
                if units.isEmpty {
                    problems.append("\(key): \(language) has no string units")
                }
                for unit in units where unit.state != "translated" || unit.value.isEmpty {
                    problems.append(
                        "\(key): \(language) unit state=\(unit.state) value=\(unit.value)")
                }
            }
        }
        XCTAssertTrue(
            problems.isEmpty,
            "incomplete translations (every offered language needs a translated, "
                + "non-empty value for every key):\n" + problems.joined(separator: "\n"))
    }

    // MARK: 2. Format-specifier parity

    func testEveryTranslationCarriesNoUnknownFormatSpecifier() throws {
        let strings = try loadCatalogStrings()
        var problems: [String] = []
        for (key, entry) in strings {
            guard let localizations =
                (entry as? [String: Any])?["localizations"] as? [String: Any],
                let english = localizationEntry(localizations, language: "en")
            else { continue }
            let englishValues = stringUnits(english).map { specifiers(in: $0.value) }
            let isPlural = (english["variations"] as? [String: Any])?["plural"] != nil
            for language in offeredLanguages {
                guard let loc = localizationEntry(localizations, language: language)
                else { continue }
                for unit in stringUnits(loc) {
                    let got = specifiers(in: unit.value)
                    if isPlural {
                        // Plural categories may legitimately OMIT the count
                        // (Arabic zero/one/two and Hebrew one spell the word
                        // out), but must never invent one.
                        if !englishValues.contains(where: { specifiers(got, fitIn: $0) }) {
                            problems.append("\(key) [\(language)]: \(unit.value) -> \(got)")
                        }
                    } else if got != englishValues.first {
                        problems.append("\(key) [\(language)]: \(unit.value) -> \(got)")
                    }
                }
            }
        }
        XCTAssertTrue(
            problems.isEmpty,
            "format-specifier mismatches against English:\n"
                + problems.joined(separator: "\n"))
    }

    // MARK: 3. Plural structure

    func testPluralKeysUsePluralVariationsInEveryLanguage() throws {
        let strings = try loadCatalogStrings()
        var problems: [String] = []
        var pluralKeyCount = 0
        for (key, entry) in strings {
            guard let localizations =
                (entry as? [String: Any])?["localizations"] as? [String: Any]
            else { continue }
            let englishVariations =
                (localizations["en"] as? [String: Any])?["variations"] as? [String: Any]
            guard let englishPlural = englishVariations?["plural"] as? [String: Any] else {
                continue
            }
            pluralKeyCount += 1
            XCTAssertFalse(
                englishPlural.isEmpty,
                "\(key): English plural block has no categories")
            for language in offeredLanguages {
                guard let loc = localizationEntry(localizations, language: language)
                else { continue }
                let variations = loc["variations"] as? [String: Any]
                let plural = variations?["plural"] as? [String: Any]
                if plural == nil || plural?.isEmpty == true {
                    problems.append("\(key): \(language) lacks plural variations")
                }
            }
        }
        XCTAssertGreaterThan(
            pluralKeyCount, 0,
            "no plural key found; the fixture this test guards has vanished")
        XCTAssertTrue(
            problems.isEmpty,
            "plural keys must carry plural variations in every language:\n"
                + problems.joined(separator: "\n"))
    }

    // MARK: 4. Key hygiene

    /// ASCII `...` and the ellipsis character are the same UI intent; two
    /// keys differing only there split the translation work in two (this
    /// repo once carried `Add Folder...` beside `Add Folder…`). Case is
    /// deliberately NOT normalized: sentence-case and title-case English
    /// copies are different surfaces.
    func testKeysDoNotCollideAfterEllipsisNormalization() throws {
        let strings = try loadCatalogStrings()
        var byNormalized: [String: [String]] = [:]
        for key in strings.keys {
            byNormalized[key.replacingOccurrences(of: "...", with: "…"), default: []]
                .append(key)
        }
        let collisions = byNormalized.filter { $0.value.count > 1 }
        XCTAssertTrue(
            collisions.isEmpty,
            "keys collide after `...` -> `…` normalization; merge them: "
                + collisions.map { "\($0.value)" }.joined(separator: ", "))
    }

    // MARK: 5. Source-scan guard

    /// Reads from the index OF the open paren, returns the call's argument
    /// text up to (not including) the matching close paren.
    private static func callBody(_ chars: [Character], from parenIndex: Int) -> String? {
        var depth = 0
        var inString = false
        var escape = false
        var body: [Character] = []
        var i = parenIndex
        while i < chars.count {
            let c = chars[i]
            if escape {
                escape = false
                if depth > 0 { body.append(c) }
            } else if c == "\\" {
                escape = true
                if depth > 0 { body.append(c) }
            } else if c == "\"" {
                inString.toggle()
                if depth > 0 { body.append(c) }
            } else if inString {
                if depth > 0 { body.append(c) }
            } else if c == "(" {
                depth += 1
                if depth > 1 { body.append(c) }
            } else if c == ")" {
                depth -= 1
                if depth == 0 { return String(body) }
                body.append(c)
            } else if depth > 0 {
                body.append(c)
            }
            i += 1
        }
        return nil
    }

    /// The deliberate English-only `Text("literal")` calls, by file name and
    /// argument prefix. Numeric readouts have nothing translatable in them;
    /// the empty one is a placeholder that gets its content elsewhere.
    private static let bundledCallAllowlist: [(file: String, prefix: String, reason: String)] = [
        ("ChatSearchResultRowView.swift", "\"\"",
            "empty Text placeholder; content arrives elsewhere"),
        ("ContextUsageRingView.swift", "\"\\(tokens.formatted",
            "compact token/percent numeric readout"),
        ("MessageEditingViews.swift", "\"\\(position) / \\(count)",
            "version-counter numeric readout"),
    ]

    /// Every `Text("...")` whose argument starts with a string literal must
    /// resolve through the module's catalog: a bare literal compiles to a
    /// `LocalizedStringKey` looked up in `Bundle.main`, which carries no
    /// strings in a SwiftPM build, so the site renders English under every
    /// language. This is the regression guard for the 2026 conversion pass,
    /// which fixed hundreds of sites by script and had nothing to catch the
    /// ones added afterwards. Known hole, documented rather than solved:
    /// `Text(condition ? "a" : "b")` starts with a condition, not a literal,
    /// and so is invisible to this scan.
    func testEveryLiteralTextCallPassesTheModuleBundle() throws {
        var problems: [String] = []
        let enumerator = FileManager.default.enumerator(
            at: sourcesRoot, includingPropertiesForKeys: nil)
        while let url = enumerator?.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            let source = try String(contentsOf: url, encoding: .utf8)
            let lineOf: (Int) -> Int = { offset in
                source[..<source.index(source.startIndex, offsetBy: min(offset, source.count))]
                    .components(separatedBy: "\n").count
            }
            for trigger in ["Text(", "LocalizedStringKey("] {
                var searchRange = source.startIndex..<source.endIndex
                while let found = source.range(of: trigger, range: searchRange) {
                    let offset = source.distance(from: source.startIndex, to: found.lowerBound)
                    let previous = offset > 0 ? Array(source)[offset - 1] : " "
                    let isBareCall = !previous.isLetter && !previous.isNumber && previous != "_"
                    searchRange = found.upperBound..<source.endIndex
                    guard isBareCall else { continue }
                    guard let body = Self.callBody(
                        Array(source), from: offset + trigger.count - 1)
                    else { continue }
                    guard body.hasPrefix("\"") else { continue }
                    if body.contains("bundle:") { continue }
                    if trigger == "LocalizedStringKey(" {
                        problems.append("\(url.lastPathComponent):\(lineOf(offset)) "
                            + "literal LocalizedStringKey wrapper: \(body.prefix(60))")
                        continue
                    }
                    let allowed = Self.bundledCallAllowlist.contains {
                        url.lastPathComponent == $0.file && body.hasPrefix($0.prefix)
                    }
                    if !allowed {
                        problems.append("\(url.lastPathComponent):\(lineOf(offset)) "
                            + "Text literal without bundle: .module -> \(body.prefix(60))")
                    }
                }
            }
        }
        XCTAssertTrue(
            problems.isEmpty,
            "unbundled Text literals (add `bundle: .module` and a catalog key, "
                + "or extend the documented allowlist):\n"
                + problems.joined(separator: "\n"))
    }


    // MARK: 7. Every bundled literal resolves to a catalog key

    /// The literal that starts a call body, with escapes resolved -- which is
    /// what `LocalizedStringKey` actually looks up, and therefore what the
    /// catalog must carry as a key. `\u{2022}` is resolved because a literal
    /// spelled that way and one spelled with the character itself are the SAME
    /// key; comparing raw source text would report a false gap for one of them.
    private static func leadingLiteral(_ body: String) -> (key: String, interpolated: Bool)? {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("\"") else { return nil }
        var raw: [Character] = []
        var escape = false
        var closed = false
        for character in trimmed.dropFirst() {
            if escape {
                raw.append(character)
                escape = false
            } else if character == "\\" {
                raw.append(character)
                escape = true
            } else if character == "\"" {
                closed = true
                break
            } else {
                raw.append(character)
            }
        }
        guard closed else { return nil }

        // Scanned on the RAW literal rather than on the whole call body: a
        // LATER argument containing an interpolation must not make this call
        // look interpolated and silently escape the check.
        var interpolated = false
        var scan = raw.startIndex
        while raw.index(after: scan) < raw.endIndex {
            if raw[scan] == "\\", raw[raw.index(after: scan)] == "(" {
                interpolated = true
                break
            }
            scan = raw.index(after: scan)
        }

        var out = ""
        var index = raw.startIndex
        while index < raw.endIndex {
            guard raw[index] == "\\", raw.index(after: index) < raw.endIndex else {
                out.append(raw[index])
                index = raw.index(after: index)
                continue
            }
            let next = raw[raw.index(after: index)]
            if next == "u", raw.index(index, offsetBy: 2) < raw.endIndex,
               raw[raw.index(index, offsetBy: 2)] == "{",
               let close = raw[raw.index(index, offsetBy: 3)...].firstIndex(of: "}"),
               let scalarValue = UInt32(String(raw[raw.index(index, offsetBy: 3)..<close]), radix: 16),
               let scalar = Unicode.Scalar(scalarValue) {
                out.append(Character(scalar))
                index = raw.index(after: close)
                continue
            }
            switch next {
            case "n": out.append("\n")
            case "t": out.append("\t")
            default: out.append(next)
            }
            index = raw.index(index, offsetBy: 2)
        }
        return (out, interpolated)
    }

    /// Passing `bundle: .module` is only HALF of localizing a string: the key
    /// still has to exist in the catalog, and when it does not the site renders
    /// its English source under all 21 languages -- silently, and identically
    /// to never having been converted at all. Test 5 cannot see this, because
    /// it only asks whether the argument is present. That gap is not
    /// hypothetical: it is how 416 bundled literals (58% of the bundled
    /// surface, whole settings paragraphs among them) shipped keyless while the
    /// whole suite stayed green, and it is what this test exists to make
    /// impossible to repeat.
    ///
    /// INTERPOLATED literals are skipped on purpose rather than allowlisted.
    /// `Text("Loading \(name)...", bundle: .module)` looks up a FORMAT STRING
    /// (`Loading %@...`), whose specifier depends on the interpolated
    /// expression's TYPE -- `%@` for a String, `%lld` for an Int -- which this
    /// scan cannot infer from source text. Guessing would either invent keys
    /// nothing looks up, or worse, put a wrong specifier in 21 translations,
    /// which test 2 calls the crash direction. Those call sites are a known,
    /// documented hole (`swift/docs/SWIFT_LOCALIZATION.md`), not a covered case.
    func testEveryBundledLiteralHasACatalogKey() throws {
        let strings = try loadCatalogStrings()
        var problems: [String] = []
        let enumerator = FileManager.default.enumerator(
            at: sourcesRoot, includingPropertiesForKeys: nil)
        while let url = enumerator?.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            let source = try String(contentsOf: url, encoding: .utf8)
            let lineOf: (Int) -> Int = { offset in
                source[..<source.index(source.startIndex, offsetBy: min(offset, source.count))]
                    .components(separatedBy: "\n").count
            }
            for trigger in ["Text(", "LocalizedStringKey("] {
                var searchRange = source.startIndex..<source.endIndex
                while let found = source.range(of: trigger, range: searchRange) {
                    let offset = source.distance(from: source.startIndex, to: found.lowerBound)
                    let previous = offset > 0 ? Array(source)[offset - 1] : " "
                    let isBareCall = !previous.isLetter && !previous.isNumber && previous != "_"
                    searchRange = found.upperBound..<source.endIndex
                    guard isBareCall else { continue }
                    guard let body = Self.callBody(
                        Array(source), from: offset + trigger.count - 1)
                    else { continue }
                    guard body.contains("bundle:") else { continue }
                    guard let literal = Self.leadingLiteral(body) else { continue }
                    // A literal carrying an interpolation resolves to a format
                    // string, not to itself; see the note above.
                    if literal.interpolated { continue }
                    if strings[literal.key] == nil {
                        problems.append("\(url.lastPathComponent):\(lineOf(offset)) "
                            + "bundled literal has no catalog key: \(literal.key)")
                    }
                }
            }
        }
        XCTAssertTrue(
            problems.isEmpty,
            "bundled literals with no catalog key (they render English under "
                + "every language; add the key with all 21 translations, or use "
                + "`Text(verbatim:)` when there is nothing to translate):\n"
                + problems.joined(separator: "\n"))
    }

    // MARK: 6. greetings.json parity

    /// The greetings are a deliberate SECOND translation surface -- content
    /// data (100 phrases x 21 languages), not UI chrome, so they live outside
    /// the catalog in `Resources/greetings.json`. That independence is what
    /// this test pins: the greeting data must cover exactly the languages the
    /// picker offers, in both directions, or one system drifts from the other.
    func testGreetingsCoverTheSameLanguagesAsThePicker() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "greetings", withExtension: "json"),
            "greetings.json missing from the compiled resource bundle")
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        let greetings = try XCTUnwrap(
            (json as? [String: Any])?["greetings"] as? [[String: Any]])
        XCTAssertFalse(greetings.isEmpty, "greetings.json parsed empty")

        let offered = Set(offeredLanguages.map { $0.lowercased() })
        var problems: [String] = []
        for greeting in greetings {
            let id = greeting["id"] as? String ?? "<no id>"
            let translations = try XCTUnwrap(
                greeting["translations"] as? [String: String], "\(id): no translations")
            let covered = Set(translations.keys.map { $0.lowercased() })
            if covered != offered {
                let missing = offered.subtracting(covered).sorted()
                let extra = covered.subtracting(offered).sorted()
                problems.append(
                    "\(id): missing \(missing), unknown \(extra)")
            }
            for (language, phrase) in translations where phrase.isEmpty {
                problems.append("\(id): empty phrase for \(language)")
            }
        }
        XCTAssertTrue(
            problems.isEmpty,
            "greetings.json language coverage drifted from AppLanguage:\n"
                + problems.joined(separator: "\n"))
    }
}
