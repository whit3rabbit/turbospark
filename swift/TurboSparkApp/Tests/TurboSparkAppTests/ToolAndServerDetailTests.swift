import TurboSpark
import XCTest

@testable import TurboSparkApp

/// D2, D3, F1, G15 and G16: the smaller correctness items from the
/// 2026-09-03 state review that are reachable without a model or a socket.
final class ToolAndServerDetailTests: XCTestCase {
    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - D2: `end_line` is a bound, `limit` is a count

    func testEndLineIsAnAbsoluteBoundAsTheSchemaSays() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let lines = (1...600).map { "line \($0)" }.joined(separator: "\n")
        try lines.write(to: dir.appendingPathComponent("f.txt"), atomically: true, encoding: .utf8)

        // The schema advertises "start_line and end_line bounds" and its own
        // example passes 1/100. Treated as a COUNT, this returned 520 lines
        // starting at 500 instead of the 21 the caller asked for.
        let output = try await AppToolRegistry.readFile(
            relPath: "f.txt", rootURL: dir, startLine: 500, endLine: 520)

        XCTAssertTrue(output.contains("lines 500-520"), "Got header: \(output.prefix(80))")
        XCTAssertTrue(output.contains("line 520"))
        XCTAssertFalse(output.contains("line 521"), "`end_line` is inclusive and absolute.")
    }

    func testTheTwoKeysAreReadSeparatelyThroughTheToolCall() async throws {
        // **THROUGH `execute`, NOT `readFile`.** The aliasing that collapsed
        // `end_line` and `limit` into one value lives in the ARGUMENT
        // routing, so a test calling `readFile` directly cannot see it --
        // deleting the fix leaves such a test green.
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let lines = (1...600).map { "line \($0)" }.joined(separator: "\n")
        try lines.write(to: dir.appendingPathComponent("f.txt"), atomically: true, encoding: .utf8)
        let project = AppProject(name: "p", rootDirectoryPath: dir.path)

        // `limit` is a COUNT (the Claude/OpenAI convention).
        let byLimit = await AppToolRegistry.execute(
            call: AppToolCall(
                name: "read_file",
                arguments: ["path": "f.txt", "start_line": "500", "limit": "21"],
                category: .fileRead),
            in: project)
        XCTAssertTrue(
            byLimit.output.contains("lines 500-520"), "Got: \(byLimit.output.prefix(80))")

        // `end_line` is an absolute BOUND (what the schema advertises).
        let byEndLine = await AppToolRegistry.execute(
            call: AppToolCall(
                name: "read_file",
                arguments: ["path": "f.txt", "start_line": "500", "end_line": "520"],
                category: .fileRead),
            in: project)
        XCTAssertTrue(
            byEndLine.output.contains("lines 500-520"), "Got: \(byEndLine.output.prefix(80))")

        // And they mean DIFFERENT things for the same number, which is the
        // whole point: `limit: 520` reads to the end, `end_line: 520` stops.
        let wideLimit = await AppToolRegistry.execute(
            call: AppToolCall(
                name: "read_file",
                arguments: ["path": "f.txt", "start_line": "500", "limit": "520"],
                category: .fileRead),
            in: project)
        XCTAssertTrue(
            wideLimit.output.contains("lines 500-600"),
            "A count of 520 from line 500 runs off the end of a 600-line file. Got: "
                + "\(wideLimit.output.prefix(80))")
    }

    func testAnEndLineBelowTheStartIsClampedRatherThanTrapping() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try "a\nb\nc\n".write(
            to: dir.appendingPathComponent("f.txt"), atomically: true, encoding: .utf8)

        // An unclamped `eLine < sLine - 1` traps on the range construction.
        let output = try await AppToolRegistry.readFile(
            relPath: "f.txt", rootURL: dir, startLine: 3, endLine: 1)
        XCTAssertTrue(output.contains("lines 3-3"), "Got: \(output)")
    }

    // MARK: - D3: search_code strips only the leading root prefix

    func testSearchStripsTheLeadingRootPrefixOnlyOnce() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // **THE PREFIX IS THE WHOLE ABSOLUTE ROOT PATH**, so repeating one
        // component proves nothing -- the fixture has to repeat the ENTIRE
        // prefix inside itself for `replacingOccurrences` to strip twice.
        let root = dir.resolvingSymlinksInPath().standardizedFileURL
        let nested = root.appendingPathComponent("vendor", isDirectory: true)
            .appendingPathComponent(String(root.path.dropFirst()), isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try "needle here\n".write(
            to: nested.appendingPathComponent("hit.txt"), atomically: true, encoding: .utf8)

        let output = try AppToolRegistry.searchCode(pattern: "needle", relPath: ".", rootURL: root)

        XCTAssertTrue(output.contains("needle here"), "Got: \(output)")
        let expected = String(nested.appendingPathComponent("hit.txt").path
            .dropFirst(root.path.count + 1))
        XCTAssertTrue(
            output.contains(expected),
            "The reported path must be the root-relative one. Expected to contain \(expected), "
                + "got: \(output)")
    }

    // MARK: - F1: one derivation of the served model id

    @MainActor
    func testTheServedModelIDIsTheInstallDirectoryName() {
        let model = InstalledModel(
            alias: "gemma", repo: "", revision: "", path: "/Users/x/models/gemma4.gturbo",
            family: "gemma4", installBytes: 0, installedOn: "")
        XCTAssertEqual(AppModel.servedModelID(for: model), "gemma4.gturbo")
    }

    @MainActor
    func testAnInstallWhoseNameEndsWithAnothersIsNotConfusedWithIt() {
        // `serverModelRows` matched with `path.hasSuffix(id)`, so
        // `mygemma4.gturbo` matched an attached `gemma4.gturbo` and the row
        // was labelled with the wrong install's alias.
        let other = InstalledModel(
            alias: "mine", repo: "", revision: "", path: "/Users/x/models/mygemma4.gturbo",
            family: "gemma4", installBytes: 0, installedOn: "")
        XCTAssertNotEqual(AppModel.servedModelID(for: other), "gemma4.gturbo")
        XCTAssertTrue(other.path.hasSuffix("gemma4.gturbo"), "Precondition: the suffix DOES match.")
    }

    func testServerModelSteeringStatusReportsActiveSupportedAndUnsupported() throws {
        let active = try steeringInfo(
            active: true, supported: true, summary: "add 0.3 on layers 1-32")
        XCTAssertEqual(
            ServerModelSteeringStatus(info: active),
            .active(summary: "add 0.3 on layers 1-32"))

        let supportedOff = try steeringInfo(active: false, supported: true)
        XCTAssertEqual(ServerModelSteeringStatus(info: supportedOff), .off)
        XCTAssertEqual(ServerModelSteeringStatus(info: supportedOff).label, "Steering Off (supported)")

        let unsupported = try steeringInfo(
            active: false, supported: false, reason: "steering is not wired for family Qwen4Exp")
        XCTAssertEqual(
            ServerModelSteeringStatus(info: unsupported),
            .unsupported(reason: "steering is not wired for family Qwen4Exp"))
        XCTAssertTrue(ServerModelSteeringStatus(info: unsupported).isUnsupported)
    }

    private func steeringInfo(
        active: Bool,
        supported: Bool,
        reason: String? = nil,
        summary: String? = nil
    ) throws -> SessionInfo.Steering {
        let activeText = active ? "true" : "false"
        let supportedText = supported ? "true" : "false"
        let reasonText = reason.map { "\"\($0)\"" } ?? "null"
        let summaryText = summary.map { "\"\($0)\"" } ?? "null"
        let json = "{ \"active\": \(activeText), \"supported\": \(supportedText),"
            + " \"reason\": \(reasonText), \"mode\": null, \"scale\": null,"
            + " \"summary\": \(summaryText) }"
        return try JSONDecoder().decode(SessionInfo.Steering.self, from: Data(json.utf8))
    }

    // MARK: - G15: an unbalanced brace does not disable the scanner

    func testAStrayClosingBraceDoesNotHideALaterObject() {
        // `depth` went negative on the stray `}`, so every later `{`
        // incremented from below zero and the `depth == 0` that starts an
        // object was never reached again.
        let text = "} some prose <state_patch>{\"summary\":\"ok\"}</state_patch>"
        let parsed = AppSkillStatePatch.firstJSONObject(in: text)
        XCTAssertNotNil(parsed, "One unbalanced brace must not disable the rest of the string.")
        XCTAssertEqual(parsed?["summary"], AppJSONValue.string("ok"))
    }

    // MARK: - G16: a huge persisted number does not trap

    func testAnEnormousNumberRendersRatherThanTrapping() {
        // `Int(1e300)` traps. Model output is validated before it reaches
        // here; a persisted `skillState` is not.
        let value = AppJSONValue.number(1e300)
        XCTAssertNotNil(value.foundationValue)

        XCTAssertNotNil(AppJSONValue.number(.infinity).foundationValue)
        XCTAssertNotNil(AppJSONValue.number(.nan).foundationValue)

        // An ordinary whole number still renders as an integer.
        XCTAssertEqual(AppJSONValue.number(42).foundationValue as? Int, 42)
    }
}
