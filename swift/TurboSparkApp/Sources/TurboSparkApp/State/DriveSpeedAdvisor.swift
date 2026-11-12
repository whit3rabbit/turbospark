import Foundation

/// How a volume is attached to the Mac.
///
/// Probed from DiskArbitration and the I/O registry by the link probe (task
/// 5.4), cached in memory per volume until unmount, and never persisted.
enum DriveLinkKind: Equatable, Sendable {
    case internalDrive
    case usb
    case thunderbolt
    /// Any other protocol the system names (for example a network share).
    case other(String)
    case unknown
}

/// What the system said about one volume's connection.
struct DriveConnection: Equatable, Sendable {
    let kind: DriveLinkKind
    /// The negotiated link rate in bits per second, or nil when the system
    /// gave none. USB 2.0 is 480_000_000, a 10 Gb/s cable 10_000_000_000.
    let bitsPerSecond: UInt64?
    let volumeUUID: String?
    let isLocal: Bool

    init(kind: DriveLinkKind, bitsPerSecond: UInt64?, volumeUUID: String? = nil, isLocal: Bool = true) {
        self.kind = kind
        self.bitsPerSecond = bitsPerSecond
        self.volumeUUID = volumeUUID
        self.isLocal = isLocal
    }
}

/// What the app says about a whole library (requirement 5.2).
struct LibraryAdvice: Equatable, Sendable {
    /// The connection is USB 2.0 or slower, or a measurement came in under
    /// 150 MB/s. The library row shows the slow-connection warning.
    let slowLink: Bool
    /// The rate the estimates use, in decimal MB/s: the measurement when
    /// there is one, otherwise the nominal rate of the link. Nil when neither
    /// is known.
    let effectiveMBps: Double?
}

/// What the app says about one model on that library (requirements 5.3 to 5.5).
struct ModelAdvice: Equatable, Sendable {
    let estimatedLoadSeconds: Int?
    /// A mixture-of-experts model that likely does not fit in memory, so its
    /// experts are read from the drive while it generates. Dense models are
    /// mapped and wired at load: their drive cost is the load alone.
    let streamsFromDrive: Bool
    /// The load action is replaced by a warning with "Load anyway" (5.5).
    /// Advice never prevents a load.
    let warnBeforeLoad: Bool
}

/// The pure warning and estimate policy of the model-libraries design.
///
/// No system calls and no UI: every input is passed in, so the whole policy is
/// pinned by table tests. The rates below are the design's, not measurements
/// of this code's own.
enum DriveSpeedAdvisor {
    /// USB 2.0 and anything slower is a slow link (5.2).
    static let slowLinkBitsPerSecond: UInt64 = 480_000_000
    /// A measured read below this is a slow link too (decimal MB/s).
    static let slowLinkMeasuredMBps = 150.0
    /// A link below 10 Gb/s, or a measurement below this, limits an expert
    /// streaming model's generation speed (5.4).
    static let streamingLinkBitsPerSecond: UInt64 = 10_000_000_000
    static let streamingMeasuredMBps = 500.0
    /// The load guard's relaxed reserve: a model streams when it is larger
    /// than physical memory minus this.
    static let memoryReserveBytes: UInt64 = 4 << 30

    /// Nominal rates from the design, by link rate. Rates between two tiers
    /// take the lower tier.
    private static let usb2MBps = 40.0
    private static let nominalTiers: [(minimumBitsPerSecond: UInt64, mbps: Double)] = [
        (20_000_000_000, 2_500),
        (10_000_000_000, 700),
        (5_000_000_000, 400),
        (slowLinkBitsPerSecond, usb2MBps),
    ]
    private static let thunderboltMBps = 2_500.0

    // MARK: Library level

    static func library(_ connection: DriveConnection, measuredMBps: Double?) -> LibraryAdvice {
        // This Mac and any other internal volume: no advice at all (5.7).
        if connection.kind == .internalDrive {
            return LibraryAdvice(slowLink: false, effectiveMBps: nil)
        }
        let measured = usable(measuredMBps)
        let slowByLink = knownBitsPerSecond(connection).map { $0 <= slowLinkBitsPerSecond } ?? false
        let slowByMeasurement = measured.map { $0 < slowLinkMeasuredMBps } ?? false
        return LibraryAdvice(
            slowLink: slowByLink || slowByMeasurement,
            effectiveMBps: measured ?? nominalMBps(connection))
    }

    /// The design's nominal rate for a link, in decimal MB/s, or nil when the
    /// system reported no rate (the app then shows "Unknown" and no estimate).
    static func nominalMBps(_ connection: DriveConnection) -> Double? {
        switch connection.kind {
        case .internalDrive:
            return nil
        case .thunderbolt:
            // The probe cannot read a Thunderbolt rate yet; the design gives
            // the kind one rate.
            return thunderboltMBps
        case .usb, .other, .unknown:
            guard let bits = knownBitsPerSecond(connection) else { return nil }
            if let tier = nominalTiers.first(where: { bits >= $0.minimumBitsPerSecond }) {
                return tier.mbps
            }
            // Below USB 2.0 the design gives no figure. Scale the USB 2.0 rate
            // down with the link (a 12 Mb/s link is 1 MB/s), so a very old
            // device is not promised USB 2.0 speed.
            return usb2MBps * Double(bits) / Double(slowLinkBitsPerSecond)
        }
    }

    // MARK: Model level

    static func model(
        installBytes: Int64,
        isMoE: Bool,
        physicalMemoryBytes: UInt64,
        library: LibraryAdvice,
        connection: DriveConnection
    ) -> ModelAdvice {
        // Decided here as well as in `library(_:measuredMBps:)`: a caller
        // could hand over advice computed for another volume, and This Mac
        // must never be warned (5.7).
        if connection.kind == .internalDrive {
            return ModelAdvice(estimatedLoadSeconds: nil, streamsFromDrive: false, warnBeforeLoad: false)
        }

        let size = UInt64(max(installBytes, 0))
        let budget = physicalMemoryBytes > memoryReserveBytes ? physicalMemoryBytes - memoryReserveBytes : 0
        // A conservative estimate: the UI says "likely".
        let streams = isMoE && size > budget

        // Requirement 5.4 reads "slower than 10 Gb/s OR measured below 500
        // MB/s". An unknown link is not known to be slower, so it does not
        // trip the first half on its own.
        let linkBelowStreamingRate = connection.kind != .thunderbolt
            && (knownBitsPerSecond(connection).map { $0 < streamingLinkBitsPerSecond } ?? false)
        let rateBelowStreamingRate = library.effectiveMBps.map { $0 < streamingMeasuredMBps } ?? false
        let warn = library.slowLink || (streams && (linkBelowStreamingRate || rateBelowStreamingRate))

        return ModelAdvice(
            estimatedLoadSeconds: loadSeconds(bytes: size, mbps: library.effectiveMBps),
            streamsFromDrive: streams,
            warnBeforeLoad: warn)
    }

    // MARK: Helpers

    /// Seconds to read `bytes` at `mbps`, rounded up so a wait is never
    /// understated. Nil with no size, no usable rate, or a result no `Int`
    /// should hold.
    private static func loadSeconds(bytes: UInt64, mbps: Double?) -> Int? {
        guard bytes > 0, let mbps, mbps > 0 else { return nil }
        let seconds = (Double(bytes) / (mbps * 1_000_000)).rounded(.up)
        guard seconds.isFinite, seconds < 1e15 else { return nil }
        return Int(seconds)
    }

    /// A measurement the policy can use: finite and above zero.
    private static func usable(_ measuredMBps: Double?) -> Double? {
        guard let measuredMBps, measuredMBps.isFinite, measuredMBps > 0 else { return nil }
        return measuredMBps
    }

    /// A reported link rate, with a zero treated as no report: no link
    /// negotiates zero bits, so zero means the property was empty.
    private static func knownBitsPerSecond(_ connection: DriveConnection) -> UInt64? {
        guard let bits = connection.bitsPerSecond, bits > 0 else { return nil }
        return bits
    }
}
