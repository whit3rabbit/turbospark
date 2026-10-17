import DiskArbitration
import IOKit
import XCTest
@testable import TurboSparkApp

/// A registry reader whose answers are scripted by mount path, counting reads.
private final class FakeRegistryReader: DriveRegistryReader, @unchecked Sendable {
    private let lock = NSLock()
    private var answers: [String: DriveRegistryReading] = [:]
    private var paths: [String] = []

    func set(_ reading: DriveRegistryReading?, at url: URL) {
        lock.lock(); defer { lock.unlock() }
        answers[url.path] = reading
    }

    var reads: [String] { lock.lock(); defer { lock.unlock() }; return paths }
    func readCount(of url: URL) -> Int { reads.filter { $0 == url.path }.count }

    func reading(forVolumeAt url: URL) -> DriveRegistryReading? {
        lock.lock(); defer { lock.unlock() }
        paths.append(url.path)
        return answers[url.path]
    }
}

final class DriveLinkProbeTests: XCTestCase {
    private let t7 = URL(fileURLWithPath: "/Volumes/T7", isDirectory: true)
    private let stick = URL(fileURLWithPath: "/Volumes/Stick", isDirectory: true)

    private func usb(bits: UInt64?, uuid: String? = "UUID-T7") -> DriveRegistryReading {
        DriveRegistryReading(
            protocolName: "USB", isInternal: false, isLocal: true, volumeUUID: uuid, linkBitsPerSecond: bits)
    }

    private func connection(
        _ reading: DriveRegistryReading?, at url: URL? = nil
    ) async -> DriveConnection {
        let reader = FakeRegistryReader()
        let target = url ?? t7
        reader.set(reading, at: target)
        return await DriveLinkProbe(reader: reader).connection(for: target)
    }

    // MARK: Classification

    func testUSBWithALinkSpeedReportsTheKindAndTheRate() async {
        let result = await connection(usb(bits: 10_000_000_000))
        XCTAssertEqual(result, DriveConnection(
            kind: .usb, bitsPerSecond: 10_000_000_000, volumeUUID: "UUID-T7", isLocal: true))
    }

    func testUSB2ReportsFourHundredEightyMegabits() async {
        let result = await connection(usb(bits: 480_000_000))
        XCTAssertEqual(result.kind, .usb)
        XCTAssertEqual(result.bitsPerSecond, 480_000_000)
    }

    func testUSBWithoutALinkSpeedReportsNoRate() async {
        let result = await connection(usb(bits: nil))
        XCTAssertEqual(result.kind, .usb)
        XCTAssertNil(result.bitsPerSecond)
    }

    func testAZeroLinkSpeedCountsAsNoRate() async {
        let result = await connection(usb(bits: 0))
        XCTAssertEqual(result.kind, .usb)
        XCTAssertNil(result.bitsPerSecond)
    }

    func testTheInternalFlagMakesAnInternalDriveWhateverTheProtocolSays() async {
        let apple = await connection(DriveRegistryReading(
            protocolName: "Apple Fabric", isInternal: true, isLocal: true, volumeUUID: "UUID-MAC", linkBitsPerSecond: nil))
        XCTAssertEqual(apple, DriveConnection(
            kind: .internalDrive, bitsPerSecond: nil, volumeUUID: "UUID-MAC", isLocal: true))

        // A built-in USB reader is still internal, and its rate is not shown.
        let builtIn = await connection(DriveRegistryReading(
            protocolName: "USB", isInternal: true, isLocal: true, volumeUUID: nil, linkBitsPerSecond: 480_000_000))
        XCTAssertEqual(builtIn.kind, .internalDrive)
        XCTAssertNil(builtIn.bitsPerSecond)
    }

    func testANetworkVolumeIsOtherNetworkAndNotLocal() async {
        let result = await connection(DriveRegistryReading(
            protocolName: nil, isInternal: false, isLocal: false, volumeUUID: nil, linkBitsPerSecond: nil))
        XCTAssertEqual(result.kind, .other("Network"))
        XCTAssertFalse(result.isLocal)
        XCTAssertNil(result.bitsPerSecond)
    }

    func testAnotherProtocolKeepsItsNameAndHasNoRate() async {
        let result = await connection(DriveRegistryReading(
            protocolName: "SATA", isInternal: false, isLocal: true, volumeUUID: nil, linkBitsPerSecond: 6_000_000_000))
        XCTAssertEqual(result.kind, .other("SATA"))
        XCTAssertNil(result.bitsPerSecond)
    }

    func testThunderboltIsTheKindAloneWithNoRate() async {
        let result = await connection(DriveRegistryReading(
            protocolName: "Thunderbolt", isInternal: false, isLocal: true, volumeUUID: nil, linkBitsPerSecond: 40_000_000_000))
        XCTAssertEqual(result.kind, .thunderbolt)
        XCTAssertNil(result.bitsPerSecond)
    }

    func testAMissingProtocolIsUnknown() async {
        let result = await connection(DriveRegistryReading(
            protocolName: nil, isInternal: nil, isLocal: true, volumeUUID: "UUID-X", linkBitsPerSecond: nil))
        XCTAssertEqual(result, DriveConnection(kind: .unknown, bitsPerSecond: nil, volumeUUID: "UUID-X", isLocal: true))
    }

    func testAnUnreadableVolumeIsUnknown() async {
        let result = await connection(nil)
        XCTAssertEqual(result, DriveConnection(kind: .unknown, bitsPerSecond: nil, volumeUUID: nil, isLocal: true))
    }

    func testTheMappingIsAPureFunctionOfTheReading() {
        XCTAssertEqual(
            DriveLinkProbe.connection(from: usb(bits: 5_000_000_000)),
            DriveConnection(kind: .usb, bitsPerSecond: 5_000_000_000, volumeUUID: "UUID-T7", isLocal: true))
    }

    // MARK: Caching

    func testAResultIsCachedUntilItIsInvalidated() async {
        let reader = FakeRegistryReader()
        reader.set(usb(bits: 10_000_000_000), at: t7)
        let probe = DriveLinkProbe(reader: reader)
        let first = await probe.connection(for: t7)
        let second = await probe.connection(for: t7)
        XCTAssertEqual(first, second)
        XCTAssertEqual(reader.readCount(of: t7), 1)
    }

    func testInvalidatingAVolumeForcesAFreshReadOfThatVolumeOnly() async {
        let reader = FakeRegistryReader()
        reader.set(usb(bits: 480_000_000, uuid: "UUID-T7"), at: t7)
        reader.set(usb(bits: 5_000_000_000, uuid: "UUID-STICK"), at: stick)
        let probe = DriveLinkProbe(reader: reader)
        _ = await probe.connection(for: t7)
        _ = await probe.connection(for: stick)

        await probe.invalidate(volume: t7)
        // The drive comes back on a faster cable: the fresh read must show it.
        reader.set(usb(bits: 10_000_000_000, uuid: "UUID-T7"), at: t7)
        let again = await probe.connection(for: t7)
        _ = await probe.connection(for: stick)

        XCTAssertEqual(again.bitsPerSecond, 10_000_000_000)
        XCTAssertEqual(reader.readCount(of: t7), 2)
        XCTAssertEqual(reader.readCount(of: stick), 1, "the other volume stays cached")
    }

    func testVolumesWithoutAUUIDAreCachedByMountPath() async {
        let reader = FakeRegistryReader()
        reader.set(usb(bits: 480_000_000, uuid: nil), at: t7)
        reader.set(usb(bits: 5_000_000_000, uuid: nil), at: stick)
        let probe = DriveLinkProbe(reader: reader)
        let first = await probe.connection(for: t7)
        let second = await probe.connection(for: stick)
        XCTAssertEqual(first.bitsPerSecond, 480_000_000)
        XCTAssertEqual(second.bitsPerSecond, 5_000_000_000, "two UUID-less volumes must not share a cache slot")
        _ = await probe.connection(for: t7)
        _ = await probe.connection(for: stick)
        XCTAssertEqual(reader.readCount(of: t7), 1)
        XCTAssertEqual(reader.readCount(of: stick), 1)
    }

    func testTheCacheIsKeyedByTheVolumeUUIDWhenThereIsOne() async {
        // The same volume, remounted under another name, is still the same
        // volume: invalidating by either path drops it, and its entry is shared.
        let reader = FakeRegistryReader()
        let renamed = URL(fileURLWithPath: "/Volumes/T7 1", isDirectory: true)
        reader.set(usb(bits: 10_000_000_000, uuid: "UUID-T7"), at: t7)
        reader.set(usb(bits: 10_000_000_000, uuid: "UUID-T7"), at: renamed)
        let probe = DriveLinkProbe(reader: reader)
        _ = await probe.connection(for: t7)
        _ = await probe.connection(for: renamed)

        await probe.invalidate(volume: renamed)
        _ = await probe.connection(for: t7)
        XCTAssertEqual(reader.readCount(of: t7), 2, "dropping the volume by its new path drops its old path too")
    }

    func testAnUnreadableVolumeIsNotCached() async {
        let reader = FakeRegistryReader()
        let probe = DriveLinkProbe(reader: reader)
        let first = await probe.connection(for: t7)
        XCTAssertEqual(first.kind, .unknown)
        // The system could not answer yet (the disk had just appeared): the next ask must try again.
        reader.set(usb(bits: 10_000_000_000), at: t7)
        let second = await probe.connection(for: t7)
        XCTAssertEqual(second.kind, .usb)
        XCTAssertEqual(reader.readCount(of: t7), 2)
    }

    func testInvalidatingAVolumeThatWasNeverProbedIsHarmless() async {
        let reader = FakeRegistryReader()
        reader.set(usb(bits: 10_000_000_000), at: t7)
        let probe = DriveLinkProbe(reader: reader)
        await probe.invalidate(volume: stick)
        _ = await probe.connection(for: t7)
        await probe.invalidate(volume: stick)
        _ = await probe.connection(for: t7)
        XCTAssertEqual(reader.readCount(of: t7), 1)
    }

    func testConcurrentAsksForOneVolumeReadItOnce() async {
        let reader = FakeRegistryReader()
        reader.set(usb(bits: 10_000_000_000), at: t7)
        let probe = DriveLinkProbe(reader: reader)
        let url = t7
        let results = await withTaskGroup(of: DriveConnection.self) { group in
            for _ in 0..<12 { group.addTask { await probe.connection(for: url) } }
            var all: [DriveConnection] = []
            for await result in group { all.append(result) }
            return all
        }
        XCTAssertEqual(results.count, 12)
        XCTAssertEqual(Set(results.map(\.bitsPerSecond)), [10_000_000_000])
        XCTAssertEqual(reader.readCount(of: t7), 1)
    }

    func testTheProbeNeverRunsTheReaderOnTheMainThread() async {
        let sawMain = LockedFlag()
        let reader = ThreadRecordingReader(sawMain)
        let probe = DriveLinkProbe(reader: reader)
        _ = await probe.connection(for: t7)
        XCTAssertFalse(sawMain.isSet, "registry reads can block on a drive, so they must stay off the main thread")
    }

    // MARK: Description parsing

    func testADiskDescriptionIsParsedIntoItsFields() {
        let uuid = CFUUIDCreateFromString(kCFAllocatorDefault, "11111111-2222-3333-4444-555555555555" as CFString)!
        let description: [String: Any] = [
            kDADiskDescriptionDeviceProtocolKey as String: "USB",
            kDADiskDescriptionDeviceInternalKey as String: false,
            kDADiskDescriptionVolumeNetworkKey as String: false,
            kDADiskDescriptionVolumeUUIDKey as String: uuid,
        ]
        let parsed = LiveDriveRegistryReader.parse(description: description)
        XCTAssertEqual(parsed.protocolName, "USB")
        XCTAssertEqual(parsed.isInternal, false)
        XCTAssertFalse(parsed.isNetwork)
        XCTAssertEqual(parsed.volumeUUID, "11111111-2222-3333-4444-555555555555")
    }

    func testAnEmptyDiskDescriptionParsesToNothing() {
        let parsed = LiveDriveRegistryReader.parse(description: [:])
        XCTAssertNil(parsed.protocolName)
        XCTAssertNil(parsed.isInternal)
        XCTAssertFalse(parsed.isNetwork)
        XCTAssertNil(parsed.volumeUUID)
    }

    func testABlankProtocolStringCountsAsNone() {
        let parsed = LiveDriveRegistryReader.parse(description: [kDADiskDescriptionDeviceProtocolKey as String: "  "])
        XCTAssertNil(parsed.protocolName)
    }

    // MARK: Live reader

    func testTheLiveReaderReadsTheRootVolume() throws {
        let reading = try XCTUnwrap(LiveDriveRegistryReader().reading(forVolumeAt: URL(fileURLWithPath: "/")))
        XCTAssertTrue(reading.isLocal)
        XCTAssertNotNil(reading.volumeUUID)
    }

    /// A caller holds a library root such as `/Volumes/T7/TurboSpark`, not the
    /// mount point, and DiskArbitration answers only for the mount point. A
    /// folder inside a volume reads exactly as the volume does.
    func testTheLiveReaderReadsAFolderInsideAVolumeAsItsVolume() throws {
        let folder = FileManager.default.temporaryDirectory
        let volume = try XCTUnwrap(try folder.resourceValues(forKeys: [.volumeURLKey]).volume)
        XCTAssertNotEqual(folder.standardizedFileURL.path, volume.standardizedFileURL.path)

        let inside = try XCTUnwrap(LiveDriveRegistryReader().reading(forVolumeAt: folder))
        let atTheMountPoint = try XCTUnwrap(LiveDriveRegistryReader().reading(forVolumeAt: volume))

        XCTAssertNotNil(inside.protocolName, "the protocol is known, not unknown")
        XCTAssertEqual(inside, atTheMountPoint)
    }

    func testTheRegistryWalkVisitsAncestorsStopsWhenToldAndHonoursItsBudget() throws {
        let session = try XCTUnwrap(DASessionCreate(kCFAllocatorDefault))
        let disk = try XCTUnwrap(DADiskCreateFromVolumePath(
            kCFAllocatorDefault, session, URL(fileURLWithPath: "/") as CFURL))
        let media = DADiskCopyIOMedia(disk)
        try XCTSkipIf(media == 0, "no I/O registry entry for the root volume")
        defer { IOObjectRelease(media) }

        // Never satisfied: visits the whole chain, and there is a chain.
        var all = 0
        var budget = LiveDriveRegistryReader.maximumVisited
        XCTAssertFalse(LiveDriveRegistryReader.walkAncestors(of: media, depth: 0, budget: &budget) { _ in all += 1; return false })
        XCTAssertGreaterThan(all, 3)

        // Satisfied at the first ancestor: stops there.
        var first = 0
        budget = LiveDriveRegistryReader.maximumVisited
        XCTAssertTrue(LiveDriveRegistryReader.walkAncestors(of: media, depth: 0, budget: &budget) { _ in first += 1; return true })
        XCTAssertEqual(first, 1)

        // A small budget cuts the walk off: a registry that loops cannot hang the probe.
        var limited = 0
        budget = 3
        XCTAssertFalse(LiveDriveRegistryReader.walkAncestors(of: media, depth: 0, budget: &budget) { _ in limited += 1; return false })
        XCTAssertEqual(limited, 3)
    }

    func testTheLiveReaderGivesNothingForAPathThatIsNotAVolume() {
        let gone = URL(fileURLWithPath: "/Volumes/TurboSparkNoSuchVolume-\(UUID().uuidString)")
        XCTAssertNil(LiveDriveRegistryReader().reading(forVolumeAt: gone))
    }
}

// MARK: Test helpers

private final class LockedFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return flag }
    func raise() { lock.lock(); flag = true; lock.unlock() }
}

private final class ThreadRecordingReader: DriveRegistryReader, @unchecked Sendable {
    private let flag: LockedFlag
    init(_ flag: LockedFlag) { self.flag = flag }

    func reading(forVolumeAt url: URL) -> DriveRegistryReading? {
        if Thread.isMainThread { flag.raise() }
        return DriveRegistryReading(
            protocolName: "USB", isInternal: false, isLocal: true, volumeUUID: nil, linkBitsPerSecond: nil)
    }
}
