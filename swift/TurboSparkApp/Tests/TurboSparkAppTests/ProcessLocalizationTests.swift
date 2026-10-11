import XCTest
@testable import TurboSparkApp

/// The in-app language picker reaches the menu bar and system panels only
/// through `AppleLanguages`, read once at launch, and AppKit honors it only
/// for languages the MAIN bundle declares. Without the declaration the debug
/// binary and the release `.app` both kept an English menu bar under every
/// picker choice, measured 2026-10-11 (`swift/docs/SWIFT_LOCALIZATION.md`).
final class ProcessLocalizationTests: XCTestCase {
    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // Tests/TurboSparkAppTests/
            .deletingLastPathComponent()   // Tests/
            .deletingLastPathComponent()   // TurboSparkApp/
    }

    private var offeredCodes: [String] {
        AppLanguage.allCases.filter { $0 != .system }.map(\.rawValue)
    }

    func testMainBundlePlistDeclaresExactlyTheOfferedLanguages() throws {
        let data = try Data(contentsOf: packageRoot.appendingPathComponent("Localization/Info.plist"))
        let plist = try XCTUnwrap(
            PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any])
        let declared = try XCTUnwrap(plist["CFBundleLocalizations"] as? [String])
        XCTAssertEqual(declared.count, Set(declared).count, "duplicate localization")
        XCTAssertEqual(Set(declared), Set(offeredCodes),
            "a picker language missing here keeps an English menu bar after relaunch")
        XCTAssertEqual(plist["CFBundleDevelopmentRegion"] as? String, "en")
        XCTAssertNil(plist["CFBundleIdentifier"],
            "an identifier in the embedded plist moves the debug build's defaults domain")
    }

    /// A wiring guard only: it proves both build paths still name the shared
    /// file, not that the result is honored. The real-app check does that.
    func testBothBuildPathsUseTheSharedPlist() throws {
        let manifest = try String(contentsOf: packageRoot.appendingPathComponent("Package.swift"), encoding: .utf8)
        XCTAssertTrue(manifest.contains("\"__info_plist\""))
        XCTAssertTrue(manifest.contains("\"Localization/Info.plist\""))

        let script = try String(
            contentsOf: packageRoot.appendingPathComponent("../../scripts/make-app-bundle.sh"), encoding: .utf8)
        XCTAssertTrue(script.contains("plutil -replace CFBundleLocalizations"))
        XCTAssertTrue(script.contains("swift/TurboSparkApp/Localization/Info.plist"))
    }

    func testWriteProcessLanguagesStoresThePickOrClearsForSystem() throws {
        let suite = "TurboSparkTests.processLanguages.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let key = AppLanguage.processLanguagesKey
        XCTAssertEqual(key, "AppleLanguages")

        AppLanguage.writeProcessLanguages(for: .portuguese, to: defaults)
        XCTAssertEqual(defaults.persistentDomain(forName: suite)?[key] as? [String], ["pt-BR"])

        AppLanguage.writeProcessLanguages(for: .system, to: defaults)
        XCTAssertNil(defaults.persistentDomain(forName: suite)?[key],
            "System must remove the app's own list so the global one applies")
    }

    func testSystemResolvesAgainstTheCatalogAndFallsBackToEnglish() {
        XCTAssertEqual(AppLanguage.resolveSystem(preferences: ["fa-IR"]), .english)
        XCTAssertEqual(AppLanguage.resolveSystem(preferences: ["fa-IR", "de-DE"]), .german)
        XCTAssertEqual(AppLanguage.resolveSystem(preferences: ["he-IL"]), .hebrew)
        XCTAssertEqual(AppLanguage.resolveSystem(preferences: ["zh-Hant-TW"]), .traditionalChinese)
        // Compiled `.lproj` names are lowercased.
        XCTAssertEqual(
            AppLanguage.resolveSystem(preferences: ["pt-BR"], localizations: ["en", "de", "pt-br"]), .portuguese)
    }

    func testRelaunchNoticeOnlyWhenTheMenuBarLanguageWouldChange() {
        XCTAssertFalse(AppLanguage.needsRelaunch(for: .english, launchLocalization: "en"))
        XCTAssertTrue(AppLanguage.needsRelaunch(for: .german, launchLocalization: "en"))
        XCTAssertFalse(AppLanguage.needsRelaunch(for: .portuguese, launchLocalization: "pt-br"))
        XCTAssertFalse(AppLanguage.needsRelaunch(
            for: .system, launchLocalization: AppLanguage.systemResolved.rawValue))
    }

    /// Picking System after launching under German must not keep German:
    /// `Locale.current` and `Bundle.module` stay on the launch language.
    func testSystemLocaleLeavesTheStaleLaunchLocale() {
        let launchedGerman = Locale(identifier: "de_US")
        XCTAssertEqual(
            AppLanguage.systemLocale(resolved: .english, launchLocalization: "de", current: launchedGerman).identifier,
            "en")
        XCTAssertEqual(
            AppLanguage.systemLocale(resolved: .german, launchLocalization: "de", current: launchedGerman).identifier,
            "de_US", "while they agree, Locale.current keeps the user's own formats")
    }

    func testAppStringsSystemUsesTheResolvedLanguageTable() {
        XCTAssertEqual(
            AppStrings.bundle(for: .system).bundlePath,
            AppStrings.bundle(for: AppLanguage.systemResolved).bundlePath)
        XCTAssertNotEqual(AppStrings.bundle(for: .system).bundlePath, Bundle.module.bundlePath)
    }
}
