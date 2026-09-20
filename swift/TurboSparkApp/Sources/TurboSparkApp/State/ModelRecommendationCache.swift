import CryptoKit
import Foundation
import TurboSpark

/// Machine data is shared across profiles. Only completed calculations are
/// saved, and leaving a screen does not abandon a probe needed by another.
@MainActor
final class ModelRecommendationCache {
    typealias Progress = @MainActor (UInt32, UInt32) -> Void
    typealias Calculate = @MainActor (@escaping Progress) async throws -> [ModelRecommendation]

    struct Key: Encodable {
        // Bump when recommendation arithmetic or the persisted shape changes.
        var version = 1
        let chip: String?
        let physicalMemoryBytes: UInt64
        let recommendedWorkingSetBytes: UInt64?
        let configuration: String
        let catalog: String
        let probeIfNeeded: Bool

        init(telemetry: SystemTelemetry?, configuration: String, catalog: String, probeIfNeeded: Bool) {
            chip = telemetry?.chip
            physicalMemoryBytes = telemetry?.physicalMemoryBytes ?? ProcessInfo.processInfo.physicalMemory
            recommendedWorkingSetBytes = telemetry?.recommendedWorkingSetBytes
            self.configuration = configuration
            self.catalog = catalog
            self.probeIfNeeded = probeIfNeeded
        }

        var fingerprint: String {
            get throws {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.sortedKeys]
                return SHA256.hash(data: try encoder.encode(self))
                    .map { String(format: "%02x", $0) }.joined()
            }
        }
    }

    static let shared = ModelRecommendationCache(
        directory: AppStorageRoot.machineRoot.appendingPathComponent("hardware-fit", isDirectory: true))

    private let directory: URL
    private var rowsByKey: [String: [ModelRecommendation]] = [:]
    private var inFlight: [String: Task<[ModelRecommendation], Error>] = [:]
    private var observers: [String: [UUID: Progress]] = [:]
    private var progressByKey: [String: (UInt32, UInt32)] = [:]

    init(directory: URL) { self.directory = directory }

    func load(
        key: Key,
        onProgress: @escaping Progress = { _, _ in },
        calculate: @escaping Calculate
    ) async throws -> [ModelRecommendation] {
        let fingerprint = try key.fingerprint
        if let rows = rowsByKey[fingerprint] { return rows }
        let file = directory.appendingPathComponent(fingerprint + ".json")
        // Cache files contain no user data. An unreadable entry is a miss.
        if let data = try? Data(contentsOf: file),
           let rows = try? JSONDecoder().decode([ModelRecommendation].self, from: data),
           !rows.isEmpty {
            rowsByKey[fingerprint] = rows
            return rows
        }

        let observerID = UUID()
        observers[fingerprint, default: [:]][observerID] = onProgress
        defer {
            observers[fingerprint]?[observerID] = nil
            if observers[fingerprint]?.isEmpty == true { observers[fingerprint] = nil }
        }
        if let progress = progressByKey[fingerprint] { onProgress(progress.0, progress.1) }
        if let task = inFlight[fingerprint] { return try await task.value }

        let task = Task { @MainActor in
            defer {
                inFlight[fingerprint] = nil
                progressByKey[fingerprint] = nil
            }
            let rows = try await calculate { completed, total in
                self.progressByKey[fingerprint] = (completed, total)
                if let observers = self.observers[fingerprint] {
                    for observer in observers.values { observer(completed, total) }
                }
            }
            guard !rows.isEmpty else { return rows }
            rowsByKey[fingerprint] = rows
            do {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try JSONEncoder().encode(rows).write(to: file, options: .atomic)
            } catch {
                // A full disk must not discard a usable result for this run.
                AppJSONStore.recordWriteFailure(label: "Hardware fit cache", error: error)
            }
            return rows
        }
        inFlight[fingerprint] = task
        return try await task.value
    }
}
