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
///
/// The rules live in `DesignRules.json`, shared with `scripts/check-design.py`
/// (`make design-check`), which runs the same scan without building the app.
/// CI does not run this suite, so that script and the pre-commit hook are what
/// catch a violation before review.
///
/// A baseline above the real count is slack that lets new violations in
/// unnoticed (two were 190 and 55 over when this was tightened), so a count
/// below its baseline fails too: lower the baseline to match.
final class DesignTokenRatchetTests: XCTestCase {
    private struct Rule: Decodable {
        let name: String
        let pattern: String
        let baseline: Int
        let fix: String
    }

    private struct RuleFile: Decodable {
        let rules: [Rule]
    }

    private var testsRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    }

    /// `Sources/TurboSparkApp` via `#filePath`, as `FontPropagationTests`
    /// finds it, so a broken lookup skips loudly instead of passing.
    private var sourcesRoot: URL {
        testsRoot
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
    }

    private func loadRules() throws -> [Rule] {
        let data = try Data(contentsOf: testsRoot.appendingPathComponent("DesignRules.json"))
        return try JSONDecoder().decode(RuleFile.self, from: data).rules
    }

    func testThemeColorViolationsOnlyGoDown() throws {
        let rules = try loadRules()
        XCTAssertFalse(rules.isEmpty, "DesignRules.json has no rules")
        let root = sourcesRoot
        guard let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else {
            throw XCTSkip("Sources/TurboSparkApp not found relative to this test file")
        }
        // `Theme/` implements the tokens and is where the static accessors
        // and the color literals legitimately live.
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

        for rule in rules {
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
            XCTAssertGreaterThanOrEqual(
                count, rule.baseline,
                "\(rule.name): \(count) sites, baseline \(rule.baseline). Lower the baseline in DesignRules.json to \(count) so the slack cannot hide a new violation.")
        }
    }

    /// The hexes the app draws as status are the ones the design system
    /// publishes. `make design-check` runs the same comparison without a build.
    func testStatusPaletteMatchesTheDesignSystemTokens() throws {
        let tokens = testsRoot
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("docs/design-system/tokens.json")
        let data = try Data(contentsOf: tokens)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        // tokens.json nests tokens under families; find each status by name.
        func values(_ name: String, in node: Any) -> [String: String]? {
            if let dict = node as? [String: Any] {
                if dict["name"] as? String == name, let v = dict["value"] as? [String: String] { return v }
                for child in dict.values { if let found = values(name, in: child) { return found } }
            } else if let list = node as? [Any] {
                for child in list { if let found = values(name, in: child) { return found } }
            }
            return nil
        }
        // token theme ids: dark, light, hc-dark, hc-light
        let modes: [(id: String, isDark: Bool, highContrast: Bool)] = [
            ("dark", true, false), ("light", false, false), ("hc-dark", true, true), ("hc-light", false, true),
        ]
        for (statusName, keyPath) in [("success", \AppStatusPalette.success),
                                      ("warning", \AppStatusPalette.warning),
                                      ("danger", \AppStatusPalette.danger)] {
            let token = try XCTUnwrap(values(statusName, in: json), "tokens.json has no \(statusName) token")
            for mode in modes {
                let swiftHex = AppStatusPalette.resolve(isDark: mode.isDark, isHighContrast: mode.highContrast)[keyPath: keyPath]
                XCTAssertEqual(
                    token[mode.id]?.uppercased(), swiftHex.uppercased(),
                    "\(statusName) \(mode.id): design-system tokens.json drifted from AppStatusPalette. Run docs/design-system/tools/derive_tokens.py")
            }
        }
    }
}
