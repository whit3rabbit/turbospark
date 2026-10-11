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

    // MARK: - the latched messages are toasts, so they follow the picker

    /// The latches reach `showToast`, so the sentence and the store name both
    /// resolve in the picker's language. Each template is a whole-sentence key
    /// with the name spliced in; an English verb glued to a French name was
    /// the shape this replaced.
    func testStoreFailureMessagesFollowThePickedLanguage() throws {
        let defaults = UserDefaults.standard
        let previousLanguage = defaults.object(forKey: AppLanguage.storageKey)
        defer {
            if let previousLanguage {
                defaults.set(previousLanguage, forKey: AppLanguage.storageKey)
            } else {
                defaults.removeObject(forKey: AppLanguage.storageKey)
            }
            AppJSONStore.clearLastReadError()
            AppJSONStore.clearLastWriteError()
        }
        defaults.set(AppLanguage.french.rawValue, forKey: AppLanguage.storageKey)
        let name = AppStrings.localized("Chat archive", language: .french)
        XCTAssertNotEqual(name, "Chat archive", "the store label must be a translated catalog key")

        AppJSONStore.clearLastWriteError()
        XCTAssertFalse(AppJSONStore.save(
            Fixture(value: "x"), to: URL(fileURLWithPath: "/dev/null/nested/archive.json"),
            label: "Chat archive"))
        let write = try XCTUnwrap(AppJSONStore.lastWriteError)
        XCTAssertTrue(write.contains(name), write)
        XCTAssertFalse(write.contains("could not be saved"), write)

        AppJSONStore.clearLastReadError()
        AppJSONStore.recordReadFailure(label: "Chat archive", error: CocoaError(.fileReadUnknown))
        let read = try XCTUnwrap(AppJSONStore.lastReadError)
        XCTAssertTrue(read.contains(name), read)
        XCTAssertFalse(read.contains("could not be read"), read)

        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let corrupt = dir.appendingPathComponent("corrupt.json")
        try "{ not json".write(to: corrupt, atomically: true, encoding: .utf8)
        AppJSONStore.clearLastReadError()
        XCTAssertNil(AppJSONStore.load(Fixture.self, from: corrupt, label: "Chat archive"))
        let decode = try XCTUnwrap(AppJSONStore.lastReadError)
        XCTAssertTrue(decode.contains(name), decode)
        XCTAssertFalse(decode.contains("failed to decode"), decode)
        XCTAssertFalse(decode.contains("unreadable file was kept"), decode)

        let locked = dir.appendingPathComponent("locked.json")
        try #"{"value":"kept"}"#.write(to: locked, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: locked.path)
        AppJSONStore.clearLastReadError()
        XCTAssertNil(AppJSONStore.load(Fixture.self, from: locked, label: "Chat archive"))
        let unreadable = try XCTUnwrap(AppJSONStore.lastReadError)
        XCTAssertTrue(unreadable.contains(name), unreadable)
        XCTAssertFalse(unreadable.contains("could not be read"), unreadable)
    }

    /// Every store label is a catalog key. A label is a literal handed to a
    /// `String.LocalizationValue` parameter, which neither source-scan gate in
    /// `LocalizationParityTests` sees, so a new store with a keyless label
    /// would toast its English name under all 21 languages.
    func testEveryStoreLabelIsACatalogKey() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let catalog = try XCTUnwrap(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: packageRoot.appendingPathComponent(
                    "Localization/Localizable.xcstrings"))) as? [String: Any])
        let keys = Set(try XCTUnwrap(catalog["strings"] as? [String: Any]).keys)

        let call = try NSRegularExpression(
            pattern: #"AppJSONStore\.(?:load|save|recordReadFailure|recordWriteFailure)\([^"]*?label:\s*"((?:[^"\\]|\\.)*)""#)
        var labels: [String: String] = [:]
        let sources = packageRoot.appendingPathComponent("Sources/TurboSparkApp")
        let files = try XCTUnwrap(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
        for case let file as URL in files where file.pathExtension == "swift" {
            let text = try String(contentsOf: file, encoding: .utf8)
            for match in call.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
                guard let range = Range(match.range(at: 1), in: text) else { continue }
                labels[formatKey(String(text[range]))] = file.lastPathComponent
            }
        }

        XCTAssertGreaterThanOrEqual(labels.count, 15, "the scan stopped finding store labels: \(labels)")
        let missing = labels.filter { !keys.contains($0.key) }
        XCTAssertTrue(missing.isEmpty, "store labels with no catalog key: \(missing)")
    }

    /// The key a label literal looks up. Labels interpolate only `String`
    /// values, so every `\(...)` is a `%@`.
    private func formatKey(_ literal: String) -> String {
        var key = ""
        var characters = literal[...]
        while let next = characters.first {
            if characters.hasPrefix("\\(") {
                characters = characters.dropFirst(2)
                var depth = 1
                while depth > 0, let inner = characters.popFirst() {
                    if inner == "(" { depth += 1 }
                    if inner == ")" { depth -= 1 }
                }
                key += "%@"
            } else {
                key.append(next)
                characters = characters.dropFirst()
            }
        }
        return key
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
