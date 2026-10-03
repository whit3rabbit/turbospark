import XCTest
@testable import TurboSparkApp

final class REPLProjectPermissionsTests: XCTestCase {
    func testExplicitREPLFileRootsRoundTripThroughCodable() throws {
        let original = AppProjectPermissions(replFileAccessRoots: ["/tmp/turbospark-project"])

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(AppProjectPermissions.self, from: data)

        XCTAssertEqual(decoded.replFileAccessRoots, ["/tmp/turbospark-project"])
        XCTAssertEqual(decoded, original)
    }

    func testLegacyPermissionsDecodeWithNoREPLFileRoots() throws {
        let legacy = Data(
            #"{"mode":"auto","fileRead":"allow","fileWrite":"ask","terminal":"ask","web":"allow","mcp":"ask","automation":"ask","browser":"ask","browserOriginAllowlist":[],"mcpAllowRules":[],"mcpDenyRules":[]}"#.utf8)

        let decoded = try JSONDecoder().decode(AppProjectPermissions.self, from: legacy)

        XCTAssertEqual(decoded.replFileAccessRoots, [])
    }

    func testPermissionPresetsDoNotGrantREPLFileRoots() {
        let presets = [
            AppProjectPermissions.auto,
            .standard,
            .agent,
            .permissive,
            .alwaysAsk,
            .readOnly,
            .fullAccess,
            .preset(for: .auto),
            .preset(for: .agentAuto),
            .preset(for: .permissive),
            .preset(for: .fullAccess),
            .preset(for: .readOnly)
        ]

        XCTAssertTrue(presets.allSatisfy { $0.replFileAccessRoots.isEmpty })
    }
}
