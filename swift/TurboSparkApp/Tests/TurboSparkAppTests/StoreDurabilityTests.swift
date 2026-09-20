import XCTest

@testable import TurboSparkApp

/// B1/G4: the two halves of a JSON store's failure handling, both of which
/// lost user data silently.
///
/// A swallowed WRITE (`try? data.write(...)`) discarded the whole archive on
/// quit with no sign. A swallowed DECODE was reported to stderr and returned
/// the empty default -- correct as far as it goes, except the unreadable file
/// is still on disk and the next mutation overwrites it whole and atomically,
/// so by the time anyone reads the log the original is gone.
final class StoreDurabilityTests: XCTestCase {
    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private struct Fixture: Codable, Equatable {
        var value: String
    }

    // MARK: - G4: an unreadable file is preserved, not overwritten

    func testACorruptFileIsQuarantinedRatherThanLeftToBeOverwritten() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("archive.json")
        try "{ this is not json at all".write(to: url, atomically: true, encoding: .utf8)

        let loaded = AppJSONStore.load(Fixture.self, from: url, label: "test archive")
        XCTAssertNil(loaded, "An unreadable file must not decode.")

        XCTAssertFalse(
            FileManager.default.fileExists(atPath: url.path),
            "The unreadable file must be moved aside so the next save cannot overwrite it.")

        let siblings = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        XCTAssertTrue(
            siblings.contains { $0.contains("corrupt-") },
            "It must still be on disk under a quarantine name. Found: \(siblings)")

        // And the store is usable again immediately: the next save writes a
        // fresh file beside the quarantined one rather than failing forever.
        XCTAssertTrue(AppJSONStore.save(Fixture(value: "new"), to: url, label: "test archive"))
        XCTAssertEqual(
            AppJSONStore.load(Fixture.self, from: url, label: "test archive"),
            Fixture(value: "new"))
    }

    func testAMissingFileIsNotTreatedAsCorrupt() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("absent.json")

        XCTAssertNil(AppJSONStore.load(Fixture.self, from: url, label: "absent"))
        let siblings = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        XCTAssertTrue(
            siblings.isEmpty, "A first run has nothing to quarantine. Found: \(siblings)")
    }

    // MARK: - B1: a failed write is reported, not swallowed

    func testAWriteThatCannotSucceedIsReportedRatherThanReturningQuietly() {
        AppJSONStore.clearLastWriteError()
        // A path under a file rather than a directory: the write cannot
        // succeed and no amount of retrying will change that.
        let impossible = URL(fileURLWithPath: "/dev/null/nested/archive.json")

        let ok = AppJSONStore.save(Fixture(value: "x"), to: impossible, label: "Test archive")

        XCTAssertFalse(ok, "The caller must be able to tell a failed write from a successful one.")
        XCTAssertNotNil(
            AppJSONStore.lastWriteError,
            "A silent failure here loses the whole session on quit with no sign.")
        XCTAssertTrue(AppJSONStore.lastWriteError?.contains("Test archive") ?? false)
        AppJSONStore.clearLastWriteError()
    }

    func testASuccessfulWriteRecordsNoError() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        AppJSONStore.clearLastWriteError()

        XCTAssertTrue(
            AppJSONStore.save(
                Fixture(value: "y"), to: dir.appendingPathComponent("ok.json"), label: "Test"))
        XCTAssertNil(AppJSONStore.lastWriteError)
    }

    // MARK: - the real chat archive goes through it

    func testMalformedLegacyChatArchiveIsNeverDeleted() throws {
        let root = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("chats_archive.json")
        try #"{"chats": "this used to be an array"}"#.write(
            to: url, atomically: true, encoding: .utf8)
        let vault = root.appendingPathComponent("private-vault", isDirectory: true)
        let store = ProfileVaultStore(
            rootProvider: { vault }, profileIDProvider: { "durability" },
            migrateLegacyData: false)
        _ = try store.prepareForLaunch()

        XCTAssertThrowsError(try ProfileRepository(store: store).loadChatArchive(legacyURL: url))
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: url.path),
            "A failed legacy import must keep the plaintext source for recovery.")
    }

    // MARK: - crash-orphaned tool calls are repaired at load

    /// A force-quit mid-turn freezes a call in `.running` or
    /// `.pendingApproval` on disk. `reconcilingOrphanedToolCalls` gives each
    /// its live-path terminal twin plus the synthetic result the live path
    /// would have recorded, so the model never sees a dangling call.
    private func chatWith(_ messages: [AppChatMessage]) -> [AppChat] {
        [AppChat(title: "t", messages: messages)]
    }

    func testAStuckRunningCallLoadsAsFailedWithASyntheticResult() {
        let call = AppToolCall(name: "run_command", status: .running)
        let repaired = AppModel.reconcilingOrphanedToolCalls(
            chatWith([AppChatMessage(role: .assistant, content: "", toolCalls: [call])]))

        let repairedCall = repaired[0].messages[0].toolCalls[0]
        XCTAssertEqual(repairedCall.status, .failed)
        let results = repaired[0].messages[0].toolResults
        XCTAssertEqual(results.count, 1, "A dangling call needs the result row prompt assembly pairs with it.")
        XCTAssertEqual(results[0].callID, repairedCall.id)
        XCTAssertTrue(results[0].isError)
        XCTAssertTrue(results[0].output.contains("quit"))
    }

    func testAStuckPendingApprovalCallLoadsAsDenied() {
        let call = AppToolCall(name: "write_file", status: .pendingApproval)
        let repaired = AppModel.reconcilingOrphanedToolCalls(
            chatWith([AppChatMessage(role: .assistant, content: "", toolCalls: [call])]))

        let repairedCall = repaired[0].messages[0].toolCalls[0]
        XCTAssertEqual(repairedCall.status, .denied)
        XCTAssertEqual(repaired[0].messages[0].toolResults.count, 1)
        XCTAssertTrue(repaired[0].messages[0].toolResults[0].isError)
    }

    func testTerminalCallsAreUntouchedAndResultRowsAreNotDuplicated() {
        let done = AppToolCall(name: "read_file", status: .completed)
        let stuck = AppToolCall(name: "run_command", status: .running)
        let existing = AppToolResult(callID: stuck.id, output: "partial output", isError: false)
        let message = AppChatMessage(
            role: .assistant, content: "", toolCalls: [done, stuck], toolResults: [existing])
        let repaired = AppModel.reconcilingOrphanedToolCalls(chatWith([message]))[0].messages[0]

        XCTAssertEqual(repaired.toolCalls[0].status, .completed)
        XCTAssertEqual(repaired.toolCalls[1].status, .failed)
        XCTAssertEqual(repaired.toolResults.count, 1, "An existing result row is kept, not shadowed by a synthetic one.")
        XCTAssertEqual(repaired.toolResults[0].output, "partial output")
    }

    func testAlternatesAreRepairedToo() {
        let stuck = AppToolCall(name: "run_command", status: .running)
        let alternate = AppChatMessage(role: .assistant, content: "old", toolCalls: [stuck])
        let repaired = AppModel.reconcilingOrphanedToolCalls(
            chatWith([AppChatMessage(role: .assistant, content: "new", alternates: [alternate])]))

        XCTAssertEqual(repaired[0].messages[0].alternates[0].toolCalls[0].status, .failed)
        XCTAssertEqual(repaired[0].messages[0].alternates[0].toolResults.count, 1)
    }
}
