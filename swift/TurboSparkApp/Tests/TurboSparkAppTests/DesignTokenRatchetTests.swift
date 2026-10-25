import XCTest
@testable import TurboSparkApp

/// Holds DESIGN.md section 4's color rules as a ratchet.
///
/// The 2026-10-25 review of Models & Storage and the Model Settings inspector
/// found every one of these spellings in views that had passed review:
/// a bare static `TurboSparkTheme` accessor (wrong mode under
/// `.preferredColorScheme`, Gotcha 36), the system accent and secondary
/// styles (a custom theme never reached them), and borders hard-coded below
/// the theme's opacity (Increase Contrast could not raise them). Hundreds of
/// older sites still use them, so a zero gate would fail today. Instead each
/// count may only fall: a new site fails here, and a fixed site lowers its
/// baseline in the same change.
final class DesignTokenRatchetTests: XCTestCase {
    private struct Rule {
        let name: String
        let pattern: String
        let baseline: Int
        let fix: String
    }

    private static let rules: [Rule] = [
        Rule(
            name: "bare static TurboSparkTheme accessor",
            pattern: #"TurboSparkTheme\.\w+(Color|Background)\b(?!\s*\()"#,
            baseline: 77,
            fix: "use TurboSparkTheme.x(isDark: theme.isDark) or the resolved theme"),
        Rule(
            name: "hard-coded border opacity",
            pattern: #"\.appBorder\.opacity\("#,
            baseline: 18,
            fix: "stroke .appBorder as the theme gives it"),
        Rule(
            name: "system accent color",
            pattern: #"Color\.accentColor"#,
            baseline: 34,
            fix: "use .appAccent or theme.accent"),
        Rule(
            name: "system secondary style",
            // `.appAction(.secondary)` is a button role, not a color, so it is
            // not counted.
            pattern: #"Color\.secondary|Color\.primary\.opacity|(?<!appAction)\(\.secondary\)|\(\.tertiary\)"#,
            baseline: 265,
            fix: "use .appSecondary or .appSurface"),
    ]

    /// `Sources/TurboSparkApp` via `#filePath`, as `FontPropagationTests`
    /// finds it, so a broken lookup skips loudly instead of passing.
    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
    }

    func testThemeColorViolationsOnlyGoDown() throws {
        let root = sourcesRoot
        guard let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else {
            throw XCTSkip("Sources/TurboSparkApp not found relative to this test file")
        }
        // `Theme/` implements the tokens and is where the static accessors
        // legitimately live.
        var lines: [(path: String, line: Int, text: String)] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let path = url.path.replacingOccurrences(of: root.path + "/", with: "")
            guard !path.hasPrefix("Theme/") else { continue }
            let text = try String(contentsOf: url, encoding: .utf8)
            for (index, line) in text.components(separatedBy: .newlines).enumerated()
            where !line.trimmingCharacters(in: .whitespaces).hasPrefix("//") {
                lines.append((path, index + 1, line))
            }
        }
        try XCTSkipIf(lines.isEmpty, "Sources/TurboSparkApp not found relative to this test file")

        for rule in Self.rules {
            let regex = try NSRegularExpression(pattern: rule.pattern)
            var count = 0
            var sites: [String] = []
            for entry in lines {
                let matches = regex.numberOfMatches(
                    in: entry.text, range: NSRange(entry.text.startIndex..., in: entry.text))
                if matches > 0 {
                    count += matches
                    sites.append("\(entry.path):\(entry.line)")
                }
            }
            XCTAssertLessThanOrEqual(
                count, rule.baseline,
                "\(rule.name): \(count) sites, baseline \(rule.baseline). New sites must \(rule.fix) (DESIGN.md section 4). Sites: \(sites.joined(separator: ", "))")
        }
    }
}
