import XCTest
import TurboSpark
@testable import TurboSparkApp

@MainActor
final class ModelRecommendationCacheTests: XCTestCase {
    private func directory() throws -> URL {
        let url = AppStorageRoot.machineRoot.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func key(
        chip: String = "Apple M4", memory: UInt64 = 16_000_000_000,
        workingSet: UInt64 = 12_000_000_000, pressure: String = "normal",
        lowPower: Bool = false, configuration: String = "4096:0:relaxed:0",
        catalog: String = "catalog", probe: Bool = true
    ) throws -> ModelRecommendationCache.Key {
        let telemetry = try JSONDecoder().decode(SystemTelemetry.self, from: Data(
            """
            {"chip":"\(chip)","physicalMemoryBytes":\(memory),
             "recommendedWorkingSetBytes":\(workingSet),"memoryPressure":"\(pressure)",
             "thermalLevel":"\(pressure)","lowPowerMode":\(lowPower)}
            """.utf8))
        return .init(telemetry: telemetry, configuration: configuration, catalog: catalog, probeIfNeeded: probe)
    }

    private func rows() throws -> [ModelRecommendation] {
        try JSONDecoder().decode([ModelRecommendation].self, from: Data(
            #"[{"alias":"fixture","name":"Fixture","family":"qwen3moe","verdict":"streams","verdictSummary":"Streams","runs":true,"countedBytes":123456,"countedSource":"estimated","installBytes":987654321,"slotCacheSlots":16,"largestContext":8192,"notes":["Estimated from the checkpoint"],"toksPerSecondMin":2.5,"toksPerSecondMax":3.5,"throughput":{"chip":"Apple M4","minTokensPerSecond":2.5,"maxTokensPerSecond":3.5,"measuredOnThisChip":true}}]"#.utf8))
    }

    func testRelaunchLoadsSavedRowsWithoutRecalculatingOrReportingProgress() async throws {
        let directory = try directory()
        let key = try key()
        let rows = try rows()
        var calculations = 0
        let first = try await ModelRecommendationCache(directory: directory).load(key: key) { progress in
            calculations += 1
            progress(1, 1)
            return rows
        }
        let second = try await ModelRecommendationCache(directory: directory).load(key: key, onProgress: { _, _ in
            XCTFail("A saved result must not restart the progress UI")
        }) { _ in
            calculations += 1
            return []
        }
        XCTAssertEqual(first, rows)
        XCTAssertEqual(second, rows)
        XCTAssertEqual(calculations, 1)
    }

    func testIdentityIsStableAcrossTransientTelemetryButInvalidatesFitInputs() throws {
        let baseline = try key().fingerprint
        XCTAssertEqual(baseline, try key(pressure: "critical", lowPower: true).fingerprint)
        let variants = try [
            key(chip: "Apple M4 Pro"), key(memory: 32_000_000_000),
            key(workingSet: 10_000_000_000), key(configuration: "8192:0:relaxed:0"),
            key(configuration: "4096:16:relaxed:0"), key(configuration: "4096:0:strict:0"),
            key(configuration: "4096:0:custom:9000000000"), key(catalog: "changed"), key(probe: false),
        ]
        for variant in variants { XCTAssertNotEqual(baseline, try variant.fingerprint) }
        var updatedSchema = try key()
        updatedSchema.version += 1
        XCTAssertNotEqual(baseline, try updatedSchema.fingerprint)
    }

    func testChangedConfigurationCalculatesThenReturningToOldSettingsReusesDisk() async throws {
        let directory = try directory()
        let rows = try rows()
        var calls = 0
        for configuration in ["4096:0:relaxed:0", "8192:0:relaxed:0", "4096:0:relaxed:0"] {
            _ = try await ModelRecommendationCache(directory: directory).load(
                key: try key(configuration: configuration)
            ) { _ in calls += 1; return rows }
        }
        XCTAssertEqual(calls, 2)
    }

    func testCorruptCacheRecalculatesAndReplacesTheEntry() async throws {
        let directory = try directory()
        let key = try key()
        let file = directory.appendingPathComponent(try key.fingerprint + ".json")
        try Data("partial json".utf8).write(to: file)
        let rows = try rows()
        let result = try await ModelRecommendationCache(directory: directory).load(key: key) { _ in rows }
        XCTAssertEqual(result, rows)
        XCTAssertEqual(try JSONDecoder().decode([ModelRecommendation].self, from: Data(contentsOf: file)), rows)
    }

    func testFailedCancelledAndEmptyCalculationsAreNotSaved() async throws {
        let directory = try directory()
        let cache = ModelRecommendationCache(directory: directory)
        let key = try key()
        for error: Error in [CocoaError(.fileReadUnknown), CancellationError()] {
            do {
                _ = try await cache.load(key: key) { _ in throw error }
                XCTFail("Expected the calculation failure")
            } catch { }
        }
        let empty = try await cache.load(key: key) { _ in [] }
        XCTAssertTrue(empty.isEmpty)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
        let rows = try rows()
        let retry = try await cache.load(key: key) { _ in rows }
        XCTAssertEqual(retry, rows)
    }

    func testConcurrentViewsShareProbeAndLeavingOneViewDoesNotDiscardTheResult() async throws {
        let directory = try directory()
        let cache = ModelRecommendationCache(directory: directory)
        let key = try key()
        let rows = try rows()
        var finish: CheckedContinuation<Void, Never>?
        var report: ModelRecommendationCache.Progress?
        var secondProgress: [UInt32] = []
        let first = Task {
            try await cache.load(key: key) { progress in
                report = progress
                progress(1, 2)
                await withCheckedContinuation { finish = $0 }
                return rows
            }
        }
        while finish == nil { await Task.yield() }
        let second = Task {
            try await cache.load(key: key, onProgress: { done, _ in secondProgress.append(done) }) { _ in
                XCTFail("The second view must share the first probe")
                return []
            }
        }
        while secondProgress.isEmpty { await Task.yield() }
        first.cancel()
        report?(2, 2)
        finish?.resume()
        let firstRows = try await first.value
        let secondRows = try await second.value
        XCTAssertEqual(firstRows, rows)
        XCTAssertEqual(secondRows, rows)
        XCTAssertEqual(secondProgress, [1, 2])
        let reloaded = try await ModelRecommendationCache(directory: directory).load(key: key) { _ in
            XCTFail("Navigation must not lose the completed result")
            return []
        }
        XCTAssertEqual(reloaded, rows)
    }
}
