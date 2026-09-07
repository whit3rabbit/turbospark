import XCTest
@testable import TurboSparkApp

/// Guards `docs/SWIFT_SETTINGS_AUDIT.md` item 1's localization fix.
///
/// `swift build`/`swift run` never compile a `.xcstrings` catalog -- only
/// Xcode's own build phase does, and this package has no Xcode project -- so
/// every one of the 21 languages was dead JSON sitting in the resource
/// bundle regardless of what any `Text(...)` call site passed as `bundle:`.
/// `scripts/compile-strings.sh` runs Xcode's own `xcstringstool` ahead of
/// the build, writing `.lproj` output straight into `Resources/`, which
/// SwiftPM's `.process()` rule DOES understand. This test proves the whole
/// chain: the source catalog still declares a known key, AND the compiled
/// bundle resolves it in a language other than English.
final class LocalizationTests: XCTestCase {
    /// `Sources/TurboSparkApp/Localization/Localizable.xcstrings`, found
    /// without going through a bundle. Same reasoning as
    /// `AppearanceSettingsTests.sourceFontFiles` and
    /// `FontPropagationTests.sourcesRoot`: `#filePath` reaches the checked-in
    /// tree directly, so a broken lookup elsewhere cannot make this test
    /// skip past the thing it exists to catch.
    private var sourceCatalogURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // Tests/TurboSparkAppTests/
            .deletingLastPathComponent()   // Tests/
            .deletingLastPathComponent()   // TurboSparkApp/
            .appendingPathComponent("Localization/Localizable.xcstrings")
    }

    func testTheSourceCatalogStillDeclaresAKnownKey() throws {
        let data = try Data(contentsOf: sourceCatalogURL)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let strings = try XCTUnwrap(json?["strings"] as? [String: Any])
        XCTAssertNotNil(
            strings["Appearance"],
            "the fixture key this test resolves through the compiled bundle must still exist "
                + "in the source catalog, or the second half of this test proves nothing")
    }

    /// The decisive half. `Bundle.module` here is `TurboSparkApp`'s own
    /// internal resource-bundle accessor, reachable because
    /// `TurboSparkAppTests` declares no resources of its own and therefore
    /// gets no `Bundle.module` of its own to collide with
    /// (`swift/CLAUDE.md` Gotcha 34).
    func testTheCompiledBundleResolvesAKnownKeyInFrench() throws {
        guard let frPath = Bundle.module.path(forResource: "fr", ofType: "lproj") else {
            XCTFail(
                "no fr.lproj in Bundle.module; run scripts/compile-strings.sh "
                    + "(or `make compile-strings`) before `swift build`")
            return
        }
        let frenchBundle = try XCTUnwrap(Bundle(path: frPath))
        let resolved = frenchBundle.localizedString(forKey: "Appearance", value: nil, table: nil)
        XCTAssertEqual(
            resolved, "Apparence",
            "the French .lproj must resolve a real translation, not fall back to the English key")
    }

    /// Every language the catalog declares must have compiled, or a
    /// language that IS offered in the picker (`AppLanguage.allCases`)
    /// would silently show untranslated text.
    ///
    /// Compares through `Bundle.preferredLocalizations`, NOT
    /// `path(forResource:ofType:)`: `xcstringstool` names its output folders
    /// in lowercase (`pt-br.lproj`, `zh-hans.lproj`), while `AppLanguage`'s
    /// raw values carry the mixed-case BCP-47 spelling
    /// (`AppLanguage.portuguese.rawValue == "pt-BR"`) that
    /// `.environment(\.locale, ...)` and every `Text(_:bundle:)` lookup
    /// actually resolve through. That resolution IS case-insensitive
    /// (confirmed directly: `Bundle.preferredLocalizations(from:
    /// forPreferences: ["pt-BR"])` returns `["pt-br"]`), so a raw path
    /// lookup here would fail on exactly the two-letter-plus-region and
    /// script-tagged languages while the app runs them correctly.
    func testEveryOfferedLanguageHasACompiledLproj() throws {
        let compiled = Set(Bundle.module.localizations.map { $0.lowercased() })
        for language in AppLanguage.allCases where language != .system {
            XCTAssertTrue(
                compiled.contains(language.rawValue.lowercased()),
                "\(language.rawValue).lproj missing from the compiled bundle")
        }
    }
}
