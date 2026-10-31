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

    /// The project sheet rebuilt permissions from its own fields, so a save
    /// (a rename, say) dropped the REPL roots and any origin an approval
    /// card granted while the sheet was open.
    func testProjectSheetSaveKeepsLiveRootsAndOriginGrants() {
        let loadedOrigins = ["https://docs.example"]
        var live = AppProjectPermissions.standard
        live.replFileAccessRoots = ["/tmp/turbospark-project"]
        live.browserOriginAllowlist = loadedOrigins + ["https://granted.example"]
        live.mcpDenyRules = ["mcp__evil__*"]

        var edited = AppProjectPermissions.alwaysAsk
        edited.browserOriginAllowlist = loadedOrigins
        let untouched = ProjectSettingsSheet.mergedPermissions(
            edited: edited, live: live, loadedOrigins: loadedOrigins)
        XCTAssertEqual(untouched.mode, .ask, "The sheet's own edits still apply.")
        XCTAssertEqual(untouched.replFileAccessRoots, ["/tmp/turbospark-project"])
        XCTAssertEqual(untouched.browserOriginAllowlist, live.browserOriginAllowlist)
        XCTAssertEqual(untouched.mcpDenyRules, ["mcp__evil__*"])

        edited.browserOriginAllowlist = []
        let revoked = ProjectSettingsSheet.mergedPermissions(
            edited: edited, live: live, loadedOrigins: loadedOrigins)
        XCTAssertEqual(revoked.browserOriginAllowlist, [], "A revoke made in the sheet wins.")
    }

    /// Read Only refuses everything but file reads, so its preset says so.
    func testReadOnlyPresetDeniesEveryCategoryButFileRead() {
        let preset = AppProjectPermissions.readOnly
        XCTAssertEqual(preset.fileRead, .allow)
        for value in [preset.fileWrite, preset.terminal, preset.web, preset.mcp, preset.automation, preset.browser] {
            XCTAssertEqual(value, .deny)
        }
    }
}
