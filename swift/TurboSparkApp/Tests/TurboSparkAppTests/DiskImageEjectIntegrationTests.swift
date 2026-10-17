import Foundation
import XCTest

@testable import TurboSparkApp

/// OPT-IN: runs the REAL `DiskArbitrationEjector` against a throwaway disk
/// image that THIS file creates, so a mutant that never unmounts a whole disk
/// or never ejects goes red. Every fake-based suite passes with such a mutant,
/// because a fake ejector has no sequence to get wrong.
///
/// Skipped unless `TURBOSPARK_RUN_DISK_IMAGE_TESTS=1`: it attaches and ejects a
/// disk image, which a default test run must never do.
///
/// **NOTHING BUT THIS FILE'S OWN IMAGES IS EVER TOUCHED.**
/// - Each image has a unique file name and a unique volume name made here.
/// - The ejector is wrapped by `OwnImageGuard`, which refuses any disk whose
///   whole-disk name or volume path is not one the attach of THIS image reported.
/// - Cleanup finds an image to detach by its unique FILE NAME in `hdiutil info`,
///   never by a remembered device number (numbers are reused after an eject).
/// - The image and its folder are deleted in `tearDown`.
final class DiskImageEjectIntegrationTests: XCTestCase {
    static let gateVariable = "TURBOSPARK_RUN_DISK_IMAGE_TESTS"

    private var folder: URL!
    private var imageNames: [String] = []

    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment[Self.gateVariable] == "1",
            "set \(Self.gateVariable)=1 to attach and eject a throwaway disk image")
        folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("TurboSparkEjectImageTest-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    }

    override func tearDown() {
        // Only images this test created, found by their unique file names.
        for name in imageNames { Self.detachIfStillAttached(imageNamed: name) }
        if let folder { try? FileManager.default.removeItem(at: folder) }
        super.tearDown()
    }

    // MARK: - Process helpers

    @discardableResult
    private static func run(_ tool: String, _ arguments: [String]) -> (status: Int32, output: Data) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: tool)
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do { try process.run() } catch { return (-1, Data("\(error)".utf8)) }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, data)
    }

    private static func plist(_ result: (status: Int32, output: Data)) -> [String: Any]? {
        guard result.status == 0 else { return nil }
        return (try? PropertyListSerialization.propertyList(from: result.output, format: nil)) as? [String: Any]
    }

    /// The whole-disk device names `hdiutil info` lists for the image with this
    /// unique file name. Empty when it is not attached.
    private static func attachedDevices(ofImageNamed name: String) -> [String] {
        guard let info = plist(run("/usr/bin/hdiutil", ["info", "-plist"])),
            let images = info["images"] as? [[String: Any]]
        else { return [] }
        var devices: [String] = []
        for image in images {
            guard let path = image["image-path"] as? String, (path as NSString).lastPathComponent == name,
                let entities = image["system-entities"] as? [[String: Any]]
            else { continue }
            for entity in entities {
                guard let entry = entity["dev-entry"] as? String else { continue }
                let device = String(entry.dropFirst("/dev/".count))
                if DiskNaming.wholeDisk(ofSlice: device) == device { devices.append(device) }
            }
        }
        return devices
    }

    private static func detachIfStillAttached(imageNamed name: String) {
        // Detaching the first whole disk of the image detaches the image.
        if let device = attachedDevices(ofImageNamed: name).sorted().first {
            run("/usr/bin/hdiutil", ["detach", "/dev/\(device)", "-force"])
        }
    }

    // MARK: - An image this test owns

    private struct OwnImage {
        let name: String
        let volumeName: String
        /// Every whole disk the attach reported (the image's disk and, for APFS, the container).
        let wholeDisks: Set<String>
    }

    private func makeImage(
        kind: String, sizeMegabytes: Int, createArguments: [String], attachWithoutMounting: Bool = false
    ) throws -> OwnImage {
        // At most 11 characters: an ExFAT volume name cannot be longer.
        let token = UUID().uuidString.prefix(6)
        let volumeName = "TSE\(token)"
        let imageName = "ts-eject-\(kind)-\(token).dmg"
        let path = folder.appendingPathComponent(imageName).path
        imageNames.append(imageName)
        let created = Self.run(
            "/usr/bin/hdiutil",
            ["create", "-size", "\(sizeMegabytes)m"] + createArguments.map { $0.replacingOccurrences(of: "VOLNAME", with: volumeName) }
                + ["-ov", path])
        XCTAssertEqual(created.status, 0, String(decoding: created.output, as: UTF8.self))
        var attach = ["attach", "-nobrowse", "-plist"]
        if attachWithoutMounting { attach.append("-nomount") }
        let attached = Self.run("/usr/bin/hdiutil", attach + [path])
        let info = try XCTUnwrap(Self.plist(attached), String(decoding: attached.output, as: UTF8.self))
        let entities = try XCTUnwrap(info["system-entities"] as? [[String: Any]])
        var wholeDisks: Set<String> = []
        for entity in entities {
            guard let entry = entity["dev-entry"] as? String else { continue }
            let device = String(entry.dropFirst("/dev/".count))
            if DiskNaming.wholeDisk(ofSlice: device) == device { wholeDisks.insert(device) }
        }
        XCTAssertFalse(wholeDisks.isEmpty, "the attach reported no disk")
        return OwnImage(name: imageName, volumeName: volumeName, wholeDisks: wholeDisks)
    }

    /// What a DiskArbitration unmount request carries for a mounted volume,
    /// read back from the system by its mount path.
    private func identity(ofVolumeAt mount: String) throws -> DiskIdentity {
        let info = try XCTUnwrap(Self.plist(Self.run("/usr/sbin/diskutil", ["info", "-plist", mount])))
        let slice = try XCTUnwrap(info["DeviceIdentifier"] as? String)
        return DiskIdentity(
            bsdName: slice, wholeDiskBSDName: DiskNaming.wholeDisk(ofSlice: slice),
            volumeURL: URL(fileURLWithPath: mount, isDirectory: true))
    }

    /// Refuses any disk that is not one of this image's. Not a stand-in: it
    /// runs the real ejector after the check.
    private final class OwnImageGuard: DiskEjecting, @unchecked Sendable {
        let image: OwnImage
        let real = DiskArbitrationEjector(timeout: 30)
        private(set) var refused = false
        init(_ image: OwnImage) { self.image = image }

        func unmountAndEject(_ disk: DiskIdentity) async -> EjectResult {
            let volumeIsOurs = disk.volumeURL.map { $0.path.hasPrefix("/Volumes/\(image.volumeName)") } ?? false
            let diskIsOurs = disk.wholeDiskBSDName.map { image.wholeDisks.contains($0) } ?? false
            guard volumeIsOurs, diskIsOurs else {
                refused = true
                return .failed("refused: not a disk this test created")
            }
            return await real.unmountAndEject(disk)
        }
    }

    private func waitUntilDetached(_ image: OwnImage) async -> Bool {
        await eventuallyAsync(timeout: 15) { Self.attachedDevices(ofImageNamed: image.name).isEmpty }
    }

    // MARK: - The tests

    func testASingleExFATVolumeIsUnmountedAndEjectedSoTheImageIsGone() async throws {
        let image = try makeImage(kind: "exfat", sizeMegabytes: 40, createArguments: ["-fs", "ExFAT", "-volname", "VOLNAME"])
        let mount = "/Volumes/\(image.volumeName)"
        let disk = try identity(ofVolumeAt: mount)
        let ejector = OwnImageGuard(image)

        let result = await ejector.unmountAndEject(disk)

        XCTAssertFalse(ejector.refused)
        XCTAssertEqual(result, .ejected)
        XCTAssertFalse(FileManager.default.fileExists(atPath: mount), "the volume is unmounted")
        let detached = await waitUntilDetached(image)
        XCTAssertTrue(detached, "the eject detaches the image: a sequence that never ejects leaves it attached")
    }

    func testAnAPFSVolumeIsUnmountedThroughItsContainerAndThePhysicalDiskIsEjected() async throws {
        let image = try makeImage(kind: "apfs", sizeMegabytes: 100, createArguments: ["-fs", "APFS", "-volname", "VOLNAME"])
        XCTAssertGreaterThanOrEqual(image.wholeDisks.count, 2, "an image disk and its APFS container")
        let mount = "/Volumes/\(image.volumeName)"
        let disk = try identity(ofVolumeAt: mount)
        XCTAssertNotEqual(
            disk.wholeDiskBSDName, image.wholeDisks.sorted().first,
            "the volume's whole disk is the synthesized container, not the physical store")
        let ejector = OwnImageGuard(image)

        let result = await ejector.unmountAndEject(disk)

        XCTAssertFalse(ejector.refused)
        XCTAssertEqual(result, .ejected)
        XCTAssertFalse(FileManager.default.fileExists(atPath: mount))
        let detached = await waitUntilDetached(image)
        XCTAssertTrue(detached, "the PHYSICAL disk was ejected, not just the container unmounted")
    }

    func testTheWholeDiskIsUnmountedSoASecondVolumeOfTheDriveGoesWithIt() async throws {
        // Unique, and at most 11 characters (an ExFAT volume name limit).
        let first = "TSA\(UUID().uuidString.prefix(6))"
        let second = "TSB\(UUID().uuidString.prefix(6))"
        let image = try makeImage(
            kind: "two", sizeMegabytes: 120, createArguments: ["-layout", "GPTSPUD"], attachWithoutMounting: true)
        let device = try XCTUnwrap(image.wholeDisks.sorted().first)
        // Partitioning a disk this test attached; both names are made here.
        let partitioned = Self.run(
            "/usr/sbin/diskutil",
            ["partitionDisk", "/dev/\(device)", "GPT", "ExFAT", first, "50m", "ExFAT", second, "R"])
        XCTAssertEqual(partitioned.status, 0, String(decoding: partitioned.output, as: UTF8.self))
        let ownImage = OwnImage(name: image.name, volumeName: first, wholeDisks: image.wholeDisks)
        let disk = try identity(ofVolumeAt: "/Volumes/\(first)")
        XCTAssertTrue(FileManager.default.fileExists(atPath: "/Volumes/\(second)"))
        let ejector = OwnImageGuard(ownImage)

        let result = await ejector.unmountAndEject(disk)

        XCTAssertFalse(ejector.refused)
        XCTAssertEqual(result, .ejected)
        XCTAssertFalse(FileManager.default.fileExists(atPath: "/Volumes/\(first)"))
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: "/Volumes/\(second)"),
            "a volume-only unmount would leave the other volume of the drive mounted")
        let detached = await waitUntilDetached(ownImage)
        XCTAssertTrue(detached)
    }

    func testTheGuardRefusesAnyDiskThatIsNotThisTestsImage() async throws {
        let image = try makeImage(kind: "guard", sizeMegabytes: 30, createArguments: ["-fs", "ExFAT", "-volname", "VOLNAME"])
        let ejector = OwnImageGuard(image)
        // The internal data volume, named as DiskArbitration would name it.
        let foreign = DiskIdentity(
            bsdName: "disk3s5", wholeDiskBSDName: "disk3",
            volumeURL: URL(fileURLWithPath: "/System/Volumes/Data", isDirectory: true))
        let result = await ejector.unmountAndEject(foreign)
        XCTAssertTrue(ejector.refused)
        XCTAssertEqual(result, .failed("refused: not a disk this test created"))
        XCTAssertFalse(
            Self.attachedDevices(ofImageNamed: image.name).isEmpty, "and the image it owns was not touched either")
    }
}
