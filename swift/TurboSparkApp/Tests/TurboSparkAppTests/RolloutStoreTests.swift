import XCTest
@testable import TurboSparkApp

/// ZP-03: the per-chat trajectory log. Reads and writes go through the real
/// store over a temp directory; "crash" cases write the bytes a crash would
/// leave.
final class RolloutStoreTests: XCTestCase {
    private var directory: URL!
    private let chat = UUID()

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("RolloutStoreTests-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeStore(
        limits: RolloutStore.Limits = RolloutStore.Limits(), protected: Bool = false
    ) -> RolloutStore {
        RolloutStore(directory: directory, limits: limits, isProtected: { protected })
    }

    private func entry(_ n: Int, kind: RolloutEntry.Kind = .modelCall) -> RolloutEntry {
        RolloutEntry(
            ts: Date(timeIntervalSince1970: 1_800_000_000 + Double(n)), kind: kind,
            role: kind == .modelCall ? "assistant" : "tool", step: n, model: "m.gturbo",
            promptTokens: n, completionTokens: n * 2, latencyMs: n, finishReason: "endOfTurn",
            result: "out-\(n)")
    }

    private func fill(_ store: RolloutStore, _ count: Int) {
        for n in 1...count { store.append(entry(n), chatID: chat) }
        store.flush()
    }

    // MARK: round trip and schema

    func testAppendRoundTripsAndCarriesSchemaVersion() throws {
        let store = makeStore()
        store.append(entry(1), chatID: chat)
        store.flush()
        let page = store.readTail(chatID: chat)
        XCTAssertEqual(page.records.count, 1)
        XCTAssertEqual(page.records[0].entry.result, "out-1")
        XCTAssertEqual(page.records[0].entry.v, RolloutEntry.currentVersion)
        let line = try String(contentsOf: store.fileURL(chatID: chat), encoding: .utf8)
        XCTAssertTrue(line.contains("\"v\":1"))
        XCTAssertTrue(line.hasSuffix("\n"))
    }

    func testFilesAreOwnerOnly() throws {
        let store = makeStore()
        fill(store, 1)
        let file = try FileManager.default.attributesOfItem(atPath: store.fileURL(chatID: chat).path)
        XCTAssertEqual(file[.posixPermissions] as? Int, 0o600)
        let dir = try FileManager.default.attributesOfItem(atPath: directory.path)
        XCTAssertEqual(dir[.posixPermissions] as? Int, 0o700)
    }

    func testNewerSchemaLineIsSkippedAndCountedNotParsed() throws {
        let store = makeStore()
        fill(store, 2)
        let handle = try FileHandle(forWritingTo: store.fileURL(chatID: chat))
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(
            #"{"v":99,"ts":"2027-01-01T00:00:00Z","kind":"modelCall","role":"assistant"}"#.utf8 + [10]))
        try handle.close()
        let page = store.readTail(chatID: chat)
        XCTAssertEqual(page.records.count, 2)
        XCTAssertEqual(page.skippedLines, 1)
    }

    // MARK: bounded tail and paging

    func testTailReturnsExactlyTheLastNInOrder() {
        let store = makeStore()
        fill(store, 30)
        let page = store.readTail(chatID: chat, maxEntries: 10)
        XCTAssertEqual(page.records.map(\.entry.step), Array(21...30))
        XCTAssertTrue(page.hasEarlier)
        // One line over and one short would both show here.
        XCTAssertEqual(page.records.first?.entry.result, "out-21")
    }

    func testTailOfShortLogReturnsEverythingWithNoEarlier() {
        let store = makeStore()
        fill(store, 3)
        let page = store.readTail(chatID: chat, maxEntries: 500)
        XCTAssertEqual(page.records.count, 3)
        XCTAssertFalse(page.hasEarlier)
        XCTAssertEqual(page.startOffset, 0)
    }

    func testPagingBackCoversEveryEntryOnceAcrossManyChunks() {
        // Entries big enough that the backward read crosses several 64 KiB
        // chunks.
        let store = makeStore()
        for n in 1...120 {
            var e = entry(n)
            e.result = String(repeating: "x", count: 3_000)
            store.append(e, chatID: chat)
        }
        store.flush()
        var seen: [Int] = []
        var page = store.readTail(chatID: chat, maxEntries: 25)
        seen = page.records.compactMap(\.entry.step)
        while page.hasEarlier {
            page = store.readBefore(chatID: chat, offset: page.startOffset, maxEntries: 25)
            seen = page.records.compactMap(\.entry.step) + seen
        }
        XCTAssertEqual(seen, Array(1...120))
    }

    // MARK: crash safety (ZP-17 deferred CUJ 3)

    func testTruncatedFinalLineIsSkippedAndCountedNotParsed() throws {
        let store = makeStore()
        fill(store, 3)
        let handle = try FileHandle(forWritingTo: store.fileURL(chatID: chat))
        try handle.seekToEnd()
        // A crash mid-append: a prefix that is itself valid JSON up to the
        // cut. A reader that parsed it would invent an entry.
        try handle.write(contentsOf: Data(#"{"v":1,"ts":"2027-01-01T00:00:00Z","kind":"modelCall","role":"assis"#.utf8))
        try handle.close()
        let page = store.readTail(chatID: chat)
        XCTAssertEqual(page.records.map(\.entry.step), [1, 2, 3])
        XCTAssertTrue(page.truncatedFinalLine)
        XCTAssertEqual(page.unreadableCount, 1)
    }

    func testAppendAfterCrashFragmentDoesNotFuseWithIt() throws {
        let store = makeStore()
        fill(store, 2)
        let handle = try FileHandle(forWritingTo: store.fileURL(chatID: chat))
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(#"{"v":1,"ts":"20"#.utf8))
        try handle.close()
        store.append(entry(3), chatID: chat)
        store.flush()
        let page = store.readTail(chatID: chat)
        XCTAssertEqual(page.records.map(\.entry.step), [1, 2, 3])
        XCTAssertEqual(page.skippedLines, 1, "the fragment is closed off and counted")
        XCTAssertFalse(page.truncatedFinalLine)
    }

    func testBadLineInTheMiddleIsSkippedAndCounted() throws {
        let store = makeStore()
        fill(store, 2)
        let handle = try FileHandle(forWritingTo: store.fileURL(chatID: chat))
        try handle.seekToEnd()
        try handle.write(contentsOf: Data("not json at all\n".utf8))
        try handle.close()
        store.append(entry(3), chatID: chat)
        store.flush()
        let page = store.readTail(chatID: chat)
        XCTAssertEqual(page.records.map(\.entry.step), [1, 2, 3])
        XCTAssertEqual(page.skippedLines, 1)
        // And a search sees the same three, counting the bad line.
        let found = store.search(chatID: chat, query: "out-")
        XCTAssertEqual(found.records.count, 3)
        XCTAssertEqual(found.skippedLines, 1)
    }

    // MARK: corruption is quarantined, never overwritten

    func testCorruptFileIsQuarantinedBeforeAppend() throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let store = makeStore()
        let url = store.fileURL(chatID: chat)
        let garbage = Data("\u{0}\u{1}garbage with no schema\nsecond line\n".utf8)
        try garbage.write(to: url)
        store.append(entry(1), chatID: chat)
        store.flush()
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        let quarantined = names.filter { $0.hasPrefix("corrupt-") }
        XCTAssertEqual(quarantined.count, 1)
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent(quarantined[0])), garbage)
        XCTAssertEqual(store.readTail(chatID: chat).records.count, 1)
    }

    func testHealthyFileIsNotQuarantined() throws {
        let store = makeStore()
        fill(store, 2)
        let second = makeStore() // a fresh process verifies the file again
        second.append(entry(3), chatID: chat)
        second.flush()
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        XCTAssertTrue(names.filter { $0.hasPrefix("corrupt-") }.isEmpty)
        XCTAssertEqual(second.readTail(chatID: chat).records.count, 3)
    }

    // MARK: caps

    func testFieldsAreClippedAndFlagged() {
        var limits = RolloutStore.Limits()
        limits.maxFieldChars = 50
        limits.maxArgValueChars = 10
        limits.maxArgs = 2
        let store = makeStore(limits: limits)
        var e = entry(1, kind: .toolExec)
        e.result = String(repeating: "r", count: 500)
        e.args = ["a": String(repeating: "v", count: 100), "b": "ok", "c": "dropped"]
        store.append(e, chatID: chat)
        store.flush()
        let got = store.readTail(chatID: chat).records[0].entry
        XCTAssertEqual(got.truncated, true)
        XCTAssertLessThanOrEqual(got.result?.count ?? 0, 50 + " [truncated]".count)
        XCTAssertEqual(got.args?.count, 2)
        XCTAssertEqual(got.args?["b"], "ok")
    }

    func testFileCapDropsOldestWholeLines() {
        var limits = RolloutStore.Limits()
        limits.maxFileBytes = 6_000
        limits.trimToBytes = 3_000
        let store = makeStore(limits: limits)
        for n in 1...60 {
            var e = entry(n)
            e.result = String(repeating: "x", count: 200)
            store.append(e, chatID: chat)
        }
        store.flush()
        XCTAssertLessThanOrEqual(store.byteCount(chatID: chat), 6_000)
        let page = store.readTail(chatID: chat)
        XCTAssertEqual(page.skippedLines, 0, "the trim cuts on a line boundary")
        XCTAssertEqual(page.records.last?.entry.step, 60)
        XCTAssertGreaterThan(page.records.first?.entry.step ?? 0, 1)
    }

    // MARK: privacy

    func testRedactionRemovesCredentialsBeforeTheyReachDisk() throws {
        let store = makeStore()
        var e = entry(1, kind: .toolExec)
        e.args = ["command": "curl -H 'Authorization: Bearer abcdef1234567890' https://u:pw@host/x", "api_key": "plainvalue"]
        e.result = """
        OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwx
        export HF_TOKEN="hf_abcdefghijklmnopqrstuvwxyz"
        {"password": "hunter2"}
        ghp_abcdefghijklmnopqrstuvwxyz0123456789
        -----BEGIN PRIVATE KEY-----
        MIIEvQ
        -----END PRIVATE KEY-----
        harmless line
        """
        store.append(e, chatID: chat)
        store.flush()
        let raw = try String(contentsOf: store.fileURL(chatID: chat), encoding: .utf8)
        for secret in ["abcdef1234567890", "u:pw@", "plainvalue", "sk-abcdefghijkl", "hf_abcdefghij",
                       "hunter2", "ghp_abcdefghij", "MIIEvQ"] {
            XCTAssertFalse(raw.contains(secret), "\(secret) reached disk")
        }
        XCTAssertTrue(raw.contains("harmless line"))
        XCTAssertTrue(raw.contains(TrajectoryRedactor.marker))
    }

    func testSecretIsRedactedBeforeTheClipCanCutItInHalf() {
        // A 20-char token straddling the clip point must not survive as a
        // fragment shorter than the pattern's minimum.
        var limits = RolloutStore.Limits()
        limits.maxFieldChars = 30
        let store = makeStore(limits: limits)
        var e = entry(1)
        e.result = String(repeating: "a", count: 22) + " sk-abcdefghijklmnopqrstuvwx tail"
        store.append(e, chatID: chat)
        store.flush()
        let got = store.readTail(chatID: chat).records[0].entry.result ?? ""
        XCTAssertFalse(got.contains("sk-abc"))
    }

    func testProtectedProfileWritesNothing() {
        let store = makeStore(protected: true)
        store.append(entry(1), chatID: chat)
        store.flush()
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.fileURL(chatID: chat).path))
    }

    func testExportIsRedactedAndLeavesTheLogUnchanged() throws {
        let store = makeStore()
        fill(store, 3)
        // Plant an unredacted line the way an older build could have.
        let handle = try FileHandle(forWritingTo: store.fileURL(chatID: chat))
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(
            (#"{"v":1,"ts":"2027-01-01T00:00:00Z","kind":"toolExec","role":"tool","result":"TOKEN=abc123secret"}"# + "\n").utf8))
        try handle.close()
        let before = try Data(contentsOf: store.fileURL(chatID: chat))
        let out = directory.appendingPathComponent("export.jsonl")
        let count = try store.exportRedacted(chatID: chat, to: out)
        XCTAssertEqual(count, 4)
        let exported = try String(contentsOf: out, encoding: .utf8)
        XCTAssertFalse(exported.contains("abc123secret"))
        XCTAssertEqual(try Data(contentsOf: store.fileURL(chatID: chat)), before)
    }

    // MARK: search, delete, retention

    func testSearchScansTheWholeLogNotJustTheTail() {
        let store = makeStore()
        for n in 1...40 {
            var e = entry(n, kind: .toolExec)
            e.name = n == 3 ? "needle_tool" : "other"
            store.append(e, chatID: chat)
        }
        store.flush()
        XCTAssertEqual(store.readTail(chatID: chat, maxEntries: 5).records.count, 5)
        let found = store.search(chatID: chat, query: "NEEDLE")
        XCTAssertEqual(found.records.map(\.entry.step), [3])
    }

    func testSearchHonoursTheResultLimit() {
        let store = makeStore()
        fill(store, 10)
        let found = store.search(chatID: chat, query: "out-", maxResults: 4)
        XCTAssertEqual(found.records.count, 4)
        XCTAssertTrue(found.hitLimit)
    }

    func testDeleteRemovesTheLogAfterQueuedAppends() {
        let store = makeStore()
        store.append(entry(1), chatID: chat)
        store.delete(chatID: chat)
        store.flush()
        XCTAssertFalse(store.hasLog(chatID: chat))
        XCTAssertEqual(store.readTail(chatID: chat), .empty)
    }

    func testPruneExpiredRemovesOnlyOldFiles() throws {
        let store = makeStore()
        let other = UUID()
        store.append(entry(1), chatID: chat)
        store.append(entry(1), chatID: other)
        store.flush()
        let old = store.fileURL(chatID: chat)
        try FileManager.default.setAttributes(
            [.modificationDate: Date().addingTimeInterval(-40 * 86_400)], ofItemAtPath: old.path)
        XCTAssertEqual(store.pruneExpired(), 1)
        XCTAssertFalse(store.hasLog(chatID: chat))
        XCTAssertTrue(store.hasLog(chatID: other))
    }

    func testDeleteAllClearsLogsAndQuarantine() throws {
        let store = makeStore()
        fill(store, 2)
        try Data("x".utf8).write(to: directory.appendingPathComponent("corrupt-20270101T000000-rollout-x.jsonl"))
        store.deleteAll()
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
    }

    // MARK: eligibility (off by default, never ghost, never protected)

    func testEligibilityRule() {
        XCTAssertTrue(AppModel.trajectoryEligible(enabled: true, isGhost: false, profileProtected: false))
        XCTAssertFalse(AppModel.trajectoryEligible(enabled: false, isGhost: false, profileProtected: false))
        XCTAssertFalse(AppModel.trajectoryEligible(enabled: true, isGhost: true, profileProtected: false))
        XCTAssertFalse(AppModel.trajectoryEligible(enabled: true, isGhost: false, profileProtected: true))
    }

    func testLoggingIsOffByDefaultInSettings() throws {
        XCTAssertFalse(MacAppSettings().trajectoryLoggingEnabled)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertFalse(decoded.trajectoryLoggingEnabled)
    }
}
