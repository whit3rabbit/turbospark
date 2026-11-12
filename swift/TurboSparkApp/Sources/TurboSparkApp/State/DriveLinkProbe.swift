import Foundation

/// Reports how a volume is connected (requirement 5.1).
///
/// The design sketched this as synchronous; it is `async` because the probe is
/// an actor, so a caller on the main actor never waits on a drive.
protocol DriveLinkProbing: Sendable {
    func connection(for volume: URL) async -> DriveConnection
    /// Forgets a volume. Call it from the unmount event: the next probe of a
    /// volume mounted at the same place must read the system again.
    func invalidate(volume: URL) async
}

/// Probes a volume's connection type and negotiated link speed, off the main
/// actor, and caches the answer until the volume is unmounted.
///
/// The system is read through a `DriveRegistryReader`, so the classification
/// and the cache are tested without a drive. The cache key is the volume UUID,
/// or the mount path when the volume has none.
///
/// The reader is blocking and runs on the actor's executor, which is a
/// cooperative-pool thread: never the main thread, and only for the few
/// milliseconds DiskArbitration and the registry take. The cache means each
/// volume is read once per mount.
actor DriveLinkProbe: DriveLinkProbing {
    private let reader: DriveRegistryReader
    private var cache: [String: DriveConnection] = [:]
    /// Mount path to cache key, so an unmount (which knows only the path) can
    /// drop an entry that is keyed by UUID.
    private var keyByPath: [String: String] = [:]

    init(reader: DriveRegistryReader = LiveDriveRegistryReader()) {
        self.reader = reader
    }

    /// What the system says when it says nothing: the app shows "Unknown".
    static let unreadable = DriveConnection(kind: .unknown, bitsPerSecond: nil, volumeUUID: nil, isLocal: true)

    func connection(for volume: URL) -> DriveConnection {
        let path = Self.path(of: volume)
        if let key = keyByPath[path], let cached = cache[key] { return cached }

        // The read is synchronous inside the actor, so two callers asking
        // about one volume are serialized and the second hits the cache.
        guard let reading = reader.reading(forVolumeAt: volume) else {
            // Not cached: a disk that has only just appeared may not be
            // describable yet, and "Unknown" must not stick until unmount.
            return Self.unreadable
        }
        let connection = Self.connection(from: reading)
        let key = reading.volumeUUID.map { "uuid:\($0)" } ?? "path:\(path)"
        cache[key] = connection
        keyByPath[path] = key
        return connection
    }

    func invalidate(volume: URL) {
        guard let key = keyByPath[Self.path(of: volume)] else { return }
        cache.removeValue(forKey: key)
        // Every path that mapped to this volume goes with it.
        keyByPath = keyByPath.filter { $0.value != key }
    }

    // MARK: Classification

    /// Pure: how a reading becomes a `DriveConnection`.
    ///
    /// Precedence is network, then the internal flag, then the protocol. The
    /// rate is kept only for USB, the one link whose rate is read: a
    /// Thunderbolt link reports its kind alone (the advisor gives it one rate),
    /// and an internal drive has no link to warn about (requirement 5.7).
    static func connection(from reading: DriveRegistryReading) -> DriveConnection {
        let name = reading.protocolName?.trimmingCharacters(in: .whitespacesAndNewlines)
        let usableName = (name?.isEmpty ?? true) ? nil : name

        let kind: DriveLinkKind
        var bitsPerSecond: UInt64?
        if !reading.isLocal {
            kind = .other(usableName ?? "Network")
        } else if reading.isInternal == true {
            kind = .internalDrive
        } else if let usableName {
            if usableName.caseInsensitiveCompare("USB") == .orderedSame {
                kind = .usb
                // Zero means the registry had no real rate, not a stalled link.
                bitsPerSecond = reading.linkBitsPerSecond.flatMap { $0 > 0 ? $0 : nil }
            } else if usableName.range(of: "thunderbolt", options: .caseInsensitive) != nil {
                kind = .thunderbolt
            } else {
                kind = .other(usableName)
            }
        } else {
            kind = .unknown
        }
        return DriveConnection(
            kind: kind, bitsPerSecond: bitsPerSecond, volumeUUID: reading.volumeUUID, isLocal: reading.isLocal)
    }

    private static func path(of volume: URL) -> String {
        volume.standardizedFileURL.path
    }
}
