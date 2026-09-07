import XCTest
@testable import TurboSparkApp

/// Guards the font propagation fix in `docs/SWIFT_SETTINGS_AUDIT.md` section 1:
/// 972 of 1,206 `.font(...)` call sites read a hardcoded semantic or fixed
/// size instead of `\.appTheme`, so the font, weight and text-size pickers
/// moved 232 sites and left the rest frozen. Every hardcoded site was
/// converted to `.themedFont`/`.themedCode` (which read `\.appTheme`
/// themselves) or to `theme.ui(...)`/`theme.code(...)`. This test is the
/// regression guard the audit's "Recommended fix" section asked for: a new
/// `.font(` call site with none of those four spellings reddens here instead
/// of quietly sitting outside the theme for months, the way the original
/// 809 did.
final class FontPropagationTests: XCTestCase {
    /// Files that legitimately call `.font(` without going through the
    /// ambient theme, each for a reason specific to that file rather than an
    /// oversight:
    ///
    /// - `Theme/ThemedFont.swift` is the modifier's own implementation
    ///   (`content.font(resolvedFont)`) plus doc comments that quote the
    ///   hardcoded spelling as the "before" example.
    /// - `Theme/ThemeCodePreviewView.swift` and
    ///   `Theme/ThemeTypographyPreviewView.swift` render the Light and Dark
    ///   preference cards SIDE BY SIDE, so each resolves its OWN mode's font
    ///   directly from `AppearanceManager` rather than from the ambient
    ///   `\.appTheme` (which follows whichever mode is currently active) --
    ///   see `ThemeCodePreviewView`'s comment on `isDark`.
    /// - `Components/AppearancePreferencesCardView.swift` builds a
    ///   `Font` per candidate family/weight for the font PICKER rows
    ///   themselves, previewing choices the user has not made yet.
    private static let exemptRelativePaths: Set<String> = [
        "Theme/ThemedFont.swift",
        "Theme/ThemeCodePreviewView.swift",
        "Theme/ThemeTypographyPreviewView.swift",
        "Components/AppearancePreferencesCardView.swift",
    ]

    /// `Sources/TurboSparkApp`, found without going through a bundle.
    /// `#filePath` reaches the checked-in tree directly (the same reasoning
    /// as `AppearanceSettingsTests.sourceFontFiles`), so a broken lookup
    /// cannot make this test skip past the thing it exists to catch.
    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // Tests/TurboSparkAppTests/
            .deletingLastPathComponent()   // Tests/
            .deletingLastPathComponent()   // TurboSparkApp/
            .appendingPathComponent("Sources/TurboSparkApp")
    }

    private func swiftFiles(under root: URL) -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: nil)
        else { return [] }
        var files: [URL] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            files.append(url)
        }
        return files
    }

    func testEveryHardcodedFontCallSiteReadsTheTheme() throws {
        let root = sourcesRoot
        let files = swiftFiles(under: root)
        try XCTSkipIf(files.isEmpty, "Sources/TurboSparkApp not found relative to this test file")

        var offenders: [String] = []
        for url in files {
            let relativePath = url.path.replacingOccurrences(of: root.path + "/", with: "")
            if Self.exemptRelativePaths.contains(relativePath) { continue }
            guard let text = try? String(contentsOf: url, encoding: .utf8),
                  text.contains(".font(")
            else { continue }
            if !text.contains("themedFont") && !text.contains("themedCode") && !text.contains("theme.") {
                offenders.append(relativePath)
            }
        }

        XCTAssertTrue(
            offenders.isEmpty,
            "hardcoded .font( with no themedFont/theme. reader in: \(offenders.joined(separator: ", "))")
    }
}
