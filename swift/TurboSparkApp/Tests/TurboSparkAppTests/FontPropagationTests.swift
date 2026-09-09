import XCTest
@testable import TurboSparkApp

/// Guards the typography rules of `swift/docs/SWIFT_SETTINGS_AUDIT.md`
/// section 1, in two layers.
///
/// The 2026-09-06 propagation pass fixed REACTIVITY: 972 raw `.font(...)`
/// call sites stopped overriding the theme injector and started reading
/// `\.appTheme`. But it left every site free to pick its own size, which is
/// how the app ended up with two coexisting sizing conventions (named steps
/// and explicit `points:` literals) and 28 distinct point values playing the
/// same roles -- a popover's "small" visibly disagreeing with a sidebar
/// row's "small". The 2026-09-08 pass fixed SIZING: the `points:` spellings
/// were deleted, `AppFontStep` became the canonical scale, and every site
/// names a role. These tests keep both properties:
///
/// - every `.font(` call site reads the theme (per LINE now; the old test
///   checked per FILE, so one themed call whitewashed a hundred raw ones),
/// - no hardcoded size literal returns, through any spelling,
/// - the scale table itself is pinned, so a retuned role is a deliberate
///   test-visible edit rather than a silent reflow of every surface.
final class FontPropagationTests: XCTestCase {
    /// Files that legitimately build fonts WITHOUT the ambient theme, each
    /// for a reason specific to that file rather than an oversight:
    ///
    /// - `Theme/ThemedFont.swift` is the modifier's own implementation
    ///   (`content.font(resolvedFont)`) plus doc comments that quote the
    ///   hardcoded spellings as the "before" examples.
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

    /// A `.font(` line reads the theme when it spells one of these. A hoisted
    /// local can participate by carrying one in its name, the way
    /// `ToolResultOutputView`'s `let base = theme.code(.small)` does not
    /// need to -- the marker reaches the `.font(` line itself. The point of
    /// the list is that the SPELLINGS are enumerable: anything outside them
    /// is a new way to sit outside the theme, and should extend this list
    /// consciously rather than slip in silently.
    private static let themedMarkers = [
        "themedFont", "themedCode", "theme.", "uiFont", "codeFont", "resolvedFont",
    ]

    /// Hardcoded size spellings, by the way each escaped the last two passes.
    /// `Font.system(size:)` is the pre-propagation form; a constant after
    /// `fitting:` would reintroduce the pre-standardization drift through the
    /// one spelling that is supposed to take layout-derived values only.
    private static let sizeLiteralPatterns = [
        "Font.system(size:",
        ".font(.system(size:",
    ]
    private static let constantFittingPattern = #"fitting: ?\d"#

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

    private func swiftFiles(under root: URL) throws -> [(url: URL, relativePath: String, lines: [String])] {
        guard let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: nil)
        else { return [] }
        var files: [(url: URL, relativePath: String, lines: [String])] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let relativePath = url.path.replacingOccurrences(of: root.path + "/", with: "")
            let text = try String(contentsOf: url, encoding: .utf8)
            guard !text.isEmpty else { continue }
            files.append((url, relativePath, text.components(separatedBy: .newlines)))
        }
        return files
    }

    /// Every `.font(` call must read the theme, checked at the CALL SITE: a
    /// file mixing one themed call with fifty raw ones fails here now, where
    /// the per-file form of the 09-06 guard read it as clean.
    func testEveryFontCallSiteReadsTheTheme() throws {
        let root = sourcesRoot
        let files = try swiftFiles(under: root)
        try XCTSkipIf(files.isEmpty, "Sources/TurboSparkApp not found relative to this test file")

        var offenders: [String] = []
        for file in files where !Self.exemptRelativePaths.contains(file.relativePath) {
            for (index, line) in file.lines.enumerated() where line.contains(".font(") {
                let context = [line, file.lines.indices.contains(index + 1) ? file.lines[index + 1] : ""]
                    .joined(separator: " ")
                if !Self.themedMarkers.contains(where: context.contains) {
                    offenders.append("\(file.relativePath):\(index + 1)")
                }
            }
        }

        XCTAssertTrue(
            offenders.isEmpty,
            "hardcoded .font( with no themed reader in (every call site must read \\.appTheme): \(offenders.joined(separator: ", "))")
    }

    /// No hardcoded size literal, whatever the spelling. The numeric sizing
    /// APIs are gone from the source; this keeps them gone and keeps the one
    /// remaining numeric hatch (`themedFont(fitting:)`) honest about taking
    /// layout-derived values instead of constants.
    func testNoHardcodedFontSizeLiterals() throws {
        let files = try swiftFiles(under: sourcesRoot)
        try XCTSkipIf(files.isEmpty, "Sources/TurboSparkApp not found relative to this test file")

        var offenders: [String] = []
        for file in files where !Self.exemptRelativePaths.contains(file.relativePath) {
            for (index, line) in file.lines.enumerated() {
                let hasLiteral = Self.sizeLiteralPatterns.contains(where: line.contains)
                    || line.range(of: Self.constantFittingPattern, options: .regularExpression) != nil
                if hasLiteral {
                    offenders.append("\(file.relativePath):\(index + 1)")
                }
            }
        }

        XCTAssertTrue(
            offenders.isEmpty,
            "hardcoded font size literal (a constant size is a missing AppFontStep role): \(offenders.joined(separator: ", "))")
    }

    /// The canonical scale, pinned. `AppFontStep`'s doc table promises these
    /// rendered sizes at the 16pt default base; a factor edit that moves one
    /// reflows every surface playing that role, so it must come through here
    /// as a visible expectation change, not a silent reflow.
    func testCanonicalScaleRendersDocumentedSizes() {
        let documented: [AppFontStep: CGFloat] = [
            .micro: 9,
            .tiny: 11,
            .small: 12,
            .callout: 14,
            .base: 16,
            .large: 17,
            .title3: 18,
            .title2: 21,
            .title: 27,
            .hero: 30,
            .display: 36,
        ]
        XCTAssertEqual(Set(documented.keys), Set(AppFontStep.allCases))
        for (step, points) in documented {
            XCTAssertEqual(
                step.renderedPointsAtDefaultBase, points,
                "\(step) must render \(points)pt at the default base; update AppFontStep's doc table if this move is deliberate")
        }
        // The user's Text Size setting multiplies the base every role scales
        // from, so one notch moves every surface together.
        XCTAssertEqual(AppFontStep.base.factor, 1.0)
    }
}
