import Foundation

/// Hardware model fit recommendation for this machine.
public struct ModelRecommendation: Codable, Sendable, Identifiable, Equatable {
    /// Unique identifier for table presentation matching the model alias.
    public var id: String { alias }

    /// Fit verdict classifying whether and how the model runs on this machine.
    public enum FitVerdict: String, Codable, Sendable {
        /// Fully fits inside unified memory with plenty of headroom.
        case resident
        /// Streams expert weights from storage with active cache fitting memory.
        case streams
        /// Fits with minimal memory headroom remaining.
        case tight
        /// Exceeds hardware memory limits and cannot run safely.
        case refused
        /// Sizing or memory footprint could not be determined.
        case unknown
    }

    /// Model alias identifier in the catalog.
    public let alias: String
    /// Human-readable model display name.
    public let name: String
    /// Model family identifier.
    public let family: String?
    /// Hardware fit verdict.
    public let verdict: FitVerdict
    /// One-line explanation of the fit verdict.
    public let verdictSummary: String
    /// Whether the model can run on this hardware configuration.
    public let runs: Bool
    /// Estimated memory bytes allocated when loaded: expert slot cache plus
    /// KV. **Guard it on `countedSource`** -- a row nothing has read reports
    /// 0 here, and 0 bytes reads as "fits easily".
    public let countedBytes: UInt64
    /// Whether `countedBytes` was measured, computed, or never established.
    public let countedSource: CountedSource
    /// Total on-disk footprint in bytes.
    public let installBytes: UInt64
    /// Recommended slot cache slot count for MoE models.
    public let slotCacheSlots: Int
    /// Largest usable context window on this machine.
    public let largestContext: UInt32
    /// Informational notes regarding performance or memory constraints.
    public let notes: [String]
    /// Minimum expected decode rate in tokens per second, from a row taken
    /// on THIS chip. `nil` on any other silicon; see `throughput`.
    public let toksPerSecondMin: Double?
    /// Maximum expected decode rate in tokens per second, this chip only.
    public let toksPerSecondMax: Double?
    /// A decode band to report, which may have been measured on a DIFFERENT
    /// chip. Carries the chip so a host can label it rather than pass it off
    /// as this machine's answer. Deliberately not part of the ranking.
    public let throughput: ThroughputBand?
}

/// System hardware and power telemetry readings.
public struct SystemTelemetry: Decodable, Sendable, Equatable {
    /// Total physical RAM installed on the machine in bytes.
    public let physicalMemoryBytes: UInt64
    /// Maximum recommended memory working set in bytes.
    public let recommendedWorkingSetBytes: UInt64?
    /// Apple Silicon chip family string (e.g. M1, M2, M3 Max).
    public let chip: String?
    /// Whether macOS Low Power Mode is currently enabled.
    public let lowPowerMode: Bool
    /// Current macOS thermal pressure level.
    public let thermalLevel: String
    /// Current memory pressure: `normal`, `warn` or `critical`.
    ///
    /// **Polled here unconditionally**, unlike the decode loop's own probe,
    /// which follows the power profile and therefore does nothing under the
    /// default `performance`. This is the reading a status panel should
    /// show; `GenerationResult.peakMemoryPressure` reports `normal` on a
    /// default session because nothing watched, not because memory was fine.
    ///
    /// `decodeIfPresent` with a default, so a binding built against an older
    /// engine still decodes rather than throwing and losing the whole struct.
    public let memoryPressure: String

    private enum CodingKeys: String, CodingKey {
        case physicalMemoryBytes
        case recommendedWorkingSetBytes
        case chip
        case lowPowerMode
        case thermalLevel
        case memoryPressure
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        physicalMemoryBytes = try c.decode(UInt64.self, forKey: .physicalMemoryBytes)
        recommendedWorkingSetBytes = try c.decodeIfPresent(
            UInt64.self, forKey: .recommendedWorkingSetBytes)
        chip = try c.decodeIfPresent(String.self, forKey: .chip)
        lowPowerMode = try c.decodeIfPresent(Bool.self, forKey: .lowPowerMode) ?? false
        thermalLevel = try c.decodeIfPresent(String.self, forKey: .thermalLevel) ?? "nominal"
        memoryPressure = try c.decodeIfPresent(String.self, forKey: .memoryPressure) ?? "normal"
    }
}
