import TurboSpark
import XCTest
@testable import TurboSparkApp

@MainActor
final class DownloadHistoryTests: XCTestCase {
    private func vault(at root: URL) throws -> ProfileVaultStore {
        let store = ProfileVaultStore(
            rootProvider: { root }, profileIDProvider: { root.lastPathComponent },
            migrateLegacyData: false)
        _ = try store.prepareForLaunch()
        return store
    }

    private func model(repository: ProfileRepository) -> AppModel {
        let model = AppModel()
        model.stopCronScheduler()
        model.downloadHistoryStore = ModelDownloadHistoryStore(repository: repository)
        model.restoreModelDownloads()
        return model
    }

    func testHistorySurvivesClosingAndReopeningTheEncryptedStore() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let firstVault = try vault(at: root)
        let first = ModelDownloadHistoryStore(repository: ProfileRepository(store: firstVault))
        let statuses: [ModelDownload.Status] = [
            .queued, .running, .paused, .packing, .verifying, .loading, .cancelling,
            .completed, .failed,
        ]
        let saved = statuses.map { status in
            var row = ModelDownload(request: .repository(
                repo: "private/model@revision", alias: status.rawValue,
                file: "weights.gguf", sidecarRepo: "private/tokenizer"), status: status)
            row.downloadedBytes = 123_000_000
            row.totalBytes = 456_000_000
            if status == .failed { row.failure = "connection interrupted" }
            return row
        }
        try first.save(saved)
        firstVault.lockVault()

        let secondVault = try vault(at: root)
        defer { secondVault.lockVault() }
        let second = ModelDownloadHistoryStore(repository: ProfileRepository(store: secondVault))
        let reopened = try second.load()
        XCTAssertEqual(
            reopened.map(\.status),
            [
                .interrupted, .interrupted, .interrupted, .interrupted, .interrupted,
                .completed, .cancelled, .completed, .failed,
            ])
        for (old, new) in zip(saved, reopened) {
            XCTAssertEqual(old.id, new.id)
            XCTAssertEqual(old.request, new.request)
            XCTAssertEqual(old.downloadedBytes, new.downloadedBytes)
            XCTAssertEqual(old.totalBytes, new.totalBytes)
            XCTAssertEqual(old.updatedAt, new.updatedAt)
            XCTAssertEqual(old.failure, new.failure)
        }
        let database = try Data(contentsOf: secondVault.databaseURL)
        XCTAssertNil(database.range(of: Data("private/model".utf8)))
    }

    func testProgressSnapshotsAreThrottledButShutdownFlushesAndRejectsLateEvents() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try vault(at: root)
        defer { store.lockVault() }
        let repository = ProfileRepository(store: store)
        let app = model(repository: repository)
        let (events, continuation) = AsyncThrowingStream<InstallEvent, Error>.makeStream()
        app.installModel(alias: "restart-fixture", stream: { _ in events })
        let task = app.installTask
        app.lastDownloadHistorySaveUptime = 0
        app.recordModelDownloadProgress(done: 100, total: 1000, at: 10)
        app.recordModelDownloadProgress(done: 200, total: 1000, at: 11)
        XCTAssertEqual(try app.downloadHistoryStore.load().first?.downloadedBytes, 100)
        app.recordModelDownloadProgress(done: 150, total: 1000, at: 11.5)
        XCTAssertEqual(app.installDownloadedBytes, 200, "out-of-order callbacks cannot reduce progress")
        app.prepareModelDownloadsForShutdown()
        let saved = try XCTUnwrap(app.downloadHistoryStore.load().first)
        XCTAssertEqual(saved.downloadedBytes, 200)
        XCTAssertEqual(saved.status, .interrupted)

        let reopened = model(repository: repository)
        XCTAssertFalse(reopened.isInstallingModel)
        XCTAssertNil(reopened.activeModelDownloadID)
        XCTAssertTrue(reopened.isDownloadManagerExpanded)
        XCTAssertTrue(
            reopened.canRetryModelDownload(saved),
            "retry may queue while the previous profile's writer still owns the install slot")
        var viewUpdates = 0
        let observation = reopened.objectWillChange.sink { viewUpdates += 1 }
        defer { observation.cancel() }

        continuation.yield(.bytes(done: 800, total: 1000))
        continuation.yield(.finished(InstalledModel(
            alias: "restart-fixture", repo: "", revision: "", path: "/tmp/restart-fixture",
            family: "gemma4", installBytes: 1000, installedOn: "")))
        continuation.finish(throwing: NSError(domain: "late error", code: 1))
        await task?.value
        XCTAssertEqual(try app.downloadHistoryStore.load().first, saved)
        XCTAssertNil(app.session)
        XCTAssertNil(app.error)

        XCTAssertTrue(reopened.canRetryModelDownload(saved))
        XCTAssertGreaterThan(viewUpdates, 0, "the newly opened manager must enable Retry when the old writer exits")
        reopened.clearFinishedDownloads()
        XCTAssertTrue(try reopened.downloadHistoryStore.load().isEmpty)
    }

    func testUnreadableHistoryIsNotOverwrittenByANewAttempt() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try vault(at: root)
        defer { store.lockVault() }
        let repository = ProfileRepository(store: store)
        let original = Data("not a download archive".utf8)
        try repository.saveRawRecord(original, key: ModelDownloadHistoryStore.key)
        let app = model(repository: repository)
        XCTAssertFalse(app.downloadHistoryWritable)
        app.beginModelDownload(.catalog(alias: "new-attempt"))
        app.setModelDownloadStatus(.failed)
        XCTAssertEqual(try repository.rawRecord(key: ModelDownloadHistoryStore.key), original)
    }

    func testHistoryIsIsolatedBetweenProfilesAndBounded() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let firstVault = try vault(at: root.appendingPathComponent("first"))
        let secondVault = try vault(at: root.appendingPathComponent("second"))
        defer { firstVault.lockVault(); secondVault.lockVault() }
        let first = ModelDownloadHistoryStore(repository: ProfileRepository(store: firstVault))
        let second = ModelDownloadHistoryStore(repository: ProfileRepository(store: secondVault))
        try first.save((0..<20).map { ModelDownload(request: .catalog(alias: "model-\($0)"), status: .completed) })
        XCTAssertEqual(try first.load().count, 12)
        XCTAssertEqual(try first.load().last?.request.alias, "model-11")
        XCTAssertTrue(try second.load().isEmpty)
    }

    func testOlderRowsAndUnknownStatusRetainTheirSource() throws {
        let row = ModelDownload(request: .catalog(alias: "old-model"))
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(row)) as? [String: Any])
        for key in ["downloadedBytes", "totalBytes", "startedAt", "updatedAt", "failure"] {
            object.removeValue(forKey: key)
        }
        object["status"] = "future-status"
        let restored = try JSONDecoder().decode(ModelDownload.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertEqual(restored.id, row.id)
        XCTAssertEqual(restored.request, row.request)
        XCTAssertEqual(restored.status, .interrupted)
        XCTAssertEqual(restored.downloadedBytes, 0)
        XCTAssertNil(restored.totalBytes)
    }
}
