import DiskArbitration
import Foundation
import IOKit

/// What the system said about one volume, before the link probe turns it into
/// a `DriveConnection`. A plain value so tests can script every case.
struct DriveRegistryReading: Equatable, Sendable {
    /// DiskArbitration's device protocol: "USB", "Apple Fabric", "PCI-Express",
    /// "SATA", "Secure Digital", "Virtual Interface", and so on.
    var protocolName: String?
    /// DiskArbitration's "internal" flag; nil when the system did not say.
    var isInternal: Bool?
    var isLocal: Bool
    var volumeUUID: String?
    /// The negotiated link rate in bits per second, read only for USB.
    var linkBitsPerSecond: UInt64?
}

/// Reads a mounted volume's connection facts. The live implementation talks to
/// DiskArbitration and the I/O registry; tests inject a scripted one.
protocol DriveRegistryReader: Sendable {
    /// Blocking: the caller keeps it off the main thread. Nil when the system
    /// can say nothing about the volume (not mounted, or no longer there).
    func reading(forVolumeAt url: URL) -> DriveRegistryReading?
}

/// The live reader over DiskArbitration and IOKit.
///
/// - `DADiskCreateFromVolumePath` and `DADiskCopyDescription` give the device
///   protocol, the internal flag, the network flag and the UUID.
/// - `DADiskCopyIOMedia` gives the disk's I/O registry entry. For a USB disk
///   the walk up its IOService-plane parents reaches the `IOUSBHostDevice`,
///   whose `UsbLinkSpeed` property is the negotiated rate in bits per second
///   (10_000_000_000 on a Samsung T7, 480_000_000 for USB 2.0).
///
/// No entitlement is needed in an unsandboxed app, and TurboSpark is not
/// sandboxed. Every `io_object_t` is released on every path.
struct LiveDriveRegistryReader: DriveRegistryReader {
    /// The fields of a DiskArbitration description the probe uses.
    struct Described: Equatable {
        var protocolName: String?
        var isInternal: Bool?
        var isNetwork: Bool
        var volumeUUID: String?
    }

    /// Pure, so it is covered without a disk.
    static func parse(description: [String: Any]) -> Described {
        let rawProtocol = (description[kDADiskDescriptionDeviceProtocolKey as String] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return Described(
            protocolName: (rawProtocol?.isEmpty ?? true) ? nil : rawProtocol,
            isInternal: description[kDADiskDescriptionDeviceInternalKey as String] as? Bool,
            isNetwork: description[kDADiskDescriptionVolumeNetworkKey as String] as? Bool ?? false,
            volumeUUID: uuidString(description[kDADiskDescriptionVolumeUUIDKey as String]))
    }

    private static func uuidString(_ value: Any?) -> String? {
        guard let value, CFGetTypeID(value as AnyObject) == CFUUIDGetTypeID() else { return nil }
        return CFUUIDCreateString(kCFAllocatorDefault, (value as! CFUUID)) as String?
    }

    func reading(forVolumeAt requested: URL) -> DriveRegistryReading? {
        // DiskArbitration answers only for a volume's mount point, and a caller
        // holds a library root (`/Volumes/T7/TurboSpark`), so go up to the
        // volume first. Without this the read came back unknown for every
        // folder inside a volume.
        let url = (try? requested.resourceValues(forKeys: [.volumeURLKey]))?.volume ?? requested
        let values = try? url.resourceValues(forKeys: [.volumeUUIDStringKey, .volumeIsLocalKey, .volumeIsInternalKey])
        let session = DASessionCreate(kCFAllocatorDefault)
        let disk = session.flatMap { DADiskCreateFromVolumePath(kCFAllocatorDefault, $0, url as CFURL) }
        let described = disk.flatMap { DADiskCopyDescription($0) as? [String: Any] }.map(Self.parse(description:))
        // Neither source knows this path: it is not a mounted volume.
        guard values != nil || described != nil else { return nil }

        let isLocal = (values?.volumeIsLocal ?? true) && !(described?.isNetwork ?? false)
        let isInternal = described?.isInternal ?? values?.volumeIsInternal
        var protocolName = described?.protocolName
        var linkBits: UInt64?

        if let disk, isLocal {
            let media = DADiskCopyIOMedia(disk)
            if media != 0 {
                defer { IOObjectRelease(media) }
                if protocolName?.caseInsensitiveCompare("USB") == .orderedSame {
                    linkBits = Self.nearestUSBLinkSpeed(from: media)
                } else if protocolName?.caseInsensitiveCompare("PCI-Express") == .orderedSame,
                          isInternal == false, Self.hasThunderboltAncestor(media) {
                    // An external PCIe disk is one tunneled over Thunderbolt or
                    // USB4. NOT verified on hardware (task 5.4 had no such
                    // drive); the class-name test is a heuristic.
                    protocolName = "Thunderbolt"
                }
            }
        }

        return DriveRegistryReading(
            protocolName: protocolName,
            isInternal: isInternal,
            isLocal: isLocal,
            volumeUUID: values?.volumeUUIDString ?? described?.volumeUUID,
            linkBitsPerSecond: linkBits)
    }

    // MARK: I/O registry

    /// Bounds on the ancestor walk. A real chain is under 15 entries deep; the
    /// limits only keep a malformed registry from looping.
    static let maximumDepth = 40
    static let maximumVisited = 400

    /// `UsbLinkSpeed` of the NEAREST `IOUSBHostDevice` above the media. The
    /// nearest one is the drive's own device: the ones further up are hubs.
    static func nearestUSBLinkSpeed(from media: io_registry_entry_t) -> UInt64? {
        var speed: UInt64?
        var budget = maximumVisited
        _ = walkAncestors(of: media, depth: 0, budget: &budget) { entry in
            guard IOObjectConformsTo(entry, "IOUSBHostDevice") != 0 else { return false }
            speed = linkSpeed(of: entry)
            return true
        }
        return speed
    }

    static func hasThunderboltAncestor(_ media: io_registry_entry_t) -> Bool {
        var budget = maximumVisited
        return walkAncestors(of: media, depth: 0, budget: &budget) { entry in
            className(of: entry).range(of: "Thunderbolt", options: .caseInsensitive) != nil
        }
    }

    /// The property read, shared by the walk and the manual smoke. Nil for a
    /// missing, non-numeric, zero or negative value.
    static func linkSpeed(of device: io_registry_entry_t) -> UInt64? {
        guard let property = IORegistryEntryCreateCFProperty(
            device, "UsbLinkSpeed" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue(),
            let number = property as? NSNumber, number.int64Value > 0
        else { return nil }
        return number.uint64Value
    }

    static func className(of entry: io_registry_entry_t) -> String {
        var name = [CChar](repeating: 0, count: 128)
        guard IOObjectGetClass(entry, &name) == KERN_SUCCESS else { return "" }
        return String(cString: name)
    }

    /// Visits the IOService-plane parents of `entry`, then theirs, until
    /// `visit` returns true. Releases every iterator and entry it creates.
    /// Internal so a test can drive it with a predicate of its own.
    static func walkAncestors(
        of entry: io_registry_entry_t, depth: Int, budget: inout Int, visit: (io_registry_entry_t) -> Bool
    ) -> Bool {
        guard depth < maximumDepth else { return false }
        var iterator: io_iterator_t = 0
        guard IORegistryEntryGetParentIterator(entry, kIOServicePlane, &iterator) == KERN_SUCCESS else { return false }
        defer { IOObjectRelease(iterator) }

        var parent = IOIteratorNext(iterator)
        while parent != 0 {
            budget -= 1
            let overBudget = budget < 0
            let found = !overBudget
                && (visit(parent) || walkAncestors(of: parent, depth: depth + 1, budget: &budget, visit: visit))
            // Released exactly once per entry, before any early return.
            IOObjectRelease(parent)
            if found || overBudget { return found }
            parent = IOIteratorNext(iterator)
        }
        return false
    }
}
