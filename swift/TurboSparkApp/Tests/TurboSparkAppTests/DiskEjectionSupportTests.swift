import DiskArbitration
import Foundation
import XCTest

@testable import TurboSparkApp

/// The pieces of the live eject path that can be checked without a drive:
/// disk names, the one-shot box the DiskArbitration callback completes, the
/// dissent text, and the volume locator on the internal disk.
///
/// The real unmount-and-eject is covered by a live check on a throwaway disk
/// image, not by this suite: a unit test must never touch a real drive.
final class DiskEjectionSupportTests: XCTestCase {
    // MARK: - Disk names

    func testASliceNameGivesItsWholeDisk() {
        XCTAssertEqual(DiskNaming.wholeDisk(ofSlice: "disk5s1"), "disk5")
        XCTAssertEqual(DiskNaming.wholeDisk(ofSlice: "disk12s3s1"), "disk12")
        XCTAssertEqual(DiskNaming.wholeDisk(ofSlice: "disk4"), "disk4")
    }

    func testAnythingThatIsNotADiskNameGivesNil() {
        for name in ["", "disk", "diskx", "disk5s", "disk5s1x", "sdisk5", "/dev/disk5", "disk5 s1", "map auto_home"] {
            XCTAssertNil(DiskNaming.wholeDisk(ofSlice: name), name)
        }
    }

    // MARK: - The one-shot completion

    private func complete(
        _ steps: [DiskArbitrationEjector.Step], armTimer: Bool = false
    ) async -> DiskArbitrationEjector.Step {
        await withCheckedContinuation { (continuation: CheckedContinuation<DiskArbitrationEjector.Step, Never>) in
            let completion = DiskArbitrationEjector.Completion(continuation)
            if armTimer { completion.arm(DispatchWorkItem {}) }
            for step in steps { completion.finish(step) }
        }
    }

    func testTheFirstCompletionWinsAndALaterOneIsIgnored() async {
        let first = await complete([.done, .refused("late"), .timedOut])
        XCTAssertEqual(first, .done)
        let timeoutFirst = await complete([.timedOut, .done])
        XCTAssertEqual(timeoutFirst, .timedOut, "a late callback after the timeout changes nothing")
        let refused = await complete([.refused("busy")], armTimer: true)
        XCTAssertEqual(refused, .refused("busy"))
    }

    func testATimerArmedAfterTheCompletionIsCancelled() async {
        let item = DispatchWorkItem {}
        _ = await withCheckedContinuation { (continuation: CheckedContinuation<DiskArbitrationEjector.Step, Never>) in
            let completion = DiskArbitrationEjector.Completion(continuation)
            completion.finish(.done)
            completion.arm(item)
        }
        XCTAssertTrue(item.isCancelled, "a timer must not outlive a finished call")
    }

    func testAFinishingCompletionCancelsItsTimer() async {
        let item = DispatchWorkItem {}
        _ = await withCheckedContinuation { (continuation: CheckedContinuation<DiskArbitrationEjector.Step, Never>) in
            let completion = DiskArbitrationEjector.Completion(continuation)
            completion.arm(item)
            XCTAssertFalse(item.isCancelled)
            completion.finish(.done)
        }
        XCTAssertTrue(item.isCancelled)
    }

    func testAStepSaysWhyItFailed() {
        XCTAssertNil(DiskArbitrationEjector.Step.done.failureText)
        XCTAssertEqual(DiskArbitrationEjector.Step.refused("Resource busy").failureText, "Resource busy")
        XCTAssertNotNil(DiskArbitrationEjector.Step.timedOut.failureText)
    }

    // MARK: - The dissent text

    func testTheSystemsOwnDissentTextIsUsedWhenItHasOne() {
        let dissenter = DADissenterCreate(kCFAllocatorDefault, DAReturn(kDAReturnBusy), "Resource busy" as CFString)
        XCTAssertEqual(DiskArbitrationEjector.message(for: dissenter), "Resource busy")
    }

    func testADissentWithNoTextFallsBackToItsStatusCode() {
        let busy = DAReturn(kDAReturnBusy)
        let hex = String(UInt32(bitPattern: busy), radix: 16)
        let dissenter = DADissenterCreate(kCFAllocatorDefault, busy, nil)
        XCTAssertEqual(DiskArbitrationEjector.message(for: dissenter), "DiskArbitration error 0x" + hex)
        XCTAssertEqual(DiskArbitrationEjector.message(status: busy, text: "  "), "DiskArbitration error 0x" + hex)
    }

    func testAUnixErrorStatusIsTheCLibrarysWords() {
        // A volume held open dissents with unix_err(EBUSY) and no string.
        XCTAssertEqual(DiskArbitrationEjector.message(status: 0xC010, text: nil), "Resource busy")
        XCTAssertEqual(DiskArbitrationEjector.message(status: 0xC010, text: "Mine"), "Mine", "the system's text wins")
        // A status that is neither text nor a Unix error stays a code.
        XCTAssertEqual(DiskArbitrationEjector.message(status: 0x1234, text: nil), "DiskArbitration error 0x1234")
    }

    // MARK: - The internal-disk guard (review N2)

    func testAnInternalFixedDiskGetsAVolumeOnlyUnmountAndNeverAnEject() {
        // Whatever the shape of the disk, a device inside this Mac whose media
        // is not ejectable is never unmounted as a whole and never ejected.
        for hasStore in [true, false] {
            XCTAssertEqual(
                DiskArbitrationEjector.plan(isInternal: true, hasStore: hasStore, ejectable: false),
                [.unmountVolume], "store: \(hasStore)")
        }
    }

    /// A built-in SD card reader reports `DeviceInternal` true, but the card is
    /// ejectable media and the user expects it to eject.
    func testEjectableMediaOnAnInternalDeviceStillEjects() {
        typealias Ejector = DiskArbitrationEjector
        XCTAssertEqual(
            Ejector.plan(isInternal: true, hasStore: false, ejectable: true), [.unmountWhole, .eject])
        XCTAssertEqual(
            Ejector.plan(isInternal: true, hasStore: true, ejectable: true),
            [.unmountWhole, .unmountStore, .eject])
        XCTAssertFalse(Ejector.treatsAsInternal(isInternal: true, ejectable: true))
        XCTAssertTrue(Ejector.treatsAsInternal(isInternal: true, ejectable: false))
        XCTAssertFalse(Ejector.treatsAsInternal(isInternal: false, ejectable: false))
    }

    func testAnExternalDriveKeepsTheWholeDiskUnmountThenEjectSequence() {
        typealias Ejector = DiskArbitrationEjector
        XCTAssertEqual(
            Ejector.plan(isInternal: false, hasStore: true, ejectable: true), [.unmountWhole, .unmountStore, .eject])
        XCTAssertEqual(Ejector.plan(isInternal: false, hasStore: false, ejectable: true), [.unmountWhole, .eject])
        // A fixed disk that is not ejectable is done after the unmount.
        XCTAssertEqual(Ejector.plan(isInternal: false, hasStore: false, ejectable: false), [.unmountWhole])
    }

    func testTheSystemsOwnDataVolumeIsReadAsAnInternalDevice() throws {
        // Read-only: DiskArbitration is asked about the disk this test runs
        // from. Nothing is unmounted and no ejector runs.
        let folder = FileManager.default.temporaryDirectory.path
        let located = LiveModelVolumeLocator().locate(path: folder)
        let bsd = try XCTUnwrap(located.bsdName)
        let mount = try XCTUnwrap(located.mountPoint)
        let identity = DiskIdentity(
            bsdName: bsd, wholeDiskBSDName: DiskNaming.wholeDisk(ofSlice: bsd),
            volumeURL: URL(fileURLWithPath: mount, isDirectory: true))
        let session = try XCTUnwrap(DASessionCreate(kCFAllocatorDefault))
        let target = try XCTUnwrap(
            DiskArbitrationEjector.target(for: identity, session: session, physical: IOKitPhysicalDiskResolver()))
        // A virtual machine's boot disk can report itself as external; the
        // guard is about devices that DO say they are inside the Mac.
        try XCTSkipUnless(target.isInternal, "this machine's boot disk does not report an internal device")
        XCTAssertFalse(target.ejectable, "the boot disk's media is fixed")
        XCTAssertTrue(
            DiskArbitrationEjector.treatsAsInternal(isInternal: target.isInternal, ejectable: target.ejectable),
            "so it is guarded")
        XCTAssertNotNil(target.volume, "the volume the request named is resolved for a volume-only unmount")
    }

    func testTheInternalDiskNoticeSaysItWasUnmountedButNotEjected() {
        XCTAssertTrue(DiskArbitrationEjector.internalDiskNotice.contains("internal"))
        XCTAssertTrue(DiskArbitrationEjector.internalDiskNotice.contains("not ejected"))
    }

    // MARK: - The locator (reads only the internal disk)

    func testTheLiveLocatorPlacesTheTemporaryFolderOnTheInternalDisk() {
        let folder = FileManager.default.temporaryDirectory.path
        let volume = LiveModelVolumeLocator().locate(path: folder)
        XCTAssertNotNil(volume.mountPoint)
        XCTAssertTrue(
            EjectDecisionState.isInternalMount(EjectDecisionState.normalized(volume.mountPoint ?? "")),
            "the system volume group is the internal disk: \(volume.mountPoint ?? "nil")")
        XCTAssertNotNil(volume.bsdName)
        XCTAssertTrue(volume.bsdName.map { $0.hasPrefix("disk") } ?? false)
    }

    func testTheLiveLocatorKeepsAnUnreadablePathAsItIs() {
        let path = "/Volumes/DefinitelyNotMounted-\(UUID().uuidString)/TurboSpark/text/x.gturbo"
        let volume = LiveModelVolumeLocator().locate(path: path)
        XCTAssertEqual(volume, ModelVolume.unresolved(path))
    }

    func testThePhysicalDiskOfTheInternalContainerIsAWholeDiskNameWhenItCanBeRead() {
        let folder = FileManager.default.temporaryDirectory.path
        guard let bsd = LiveModelVolumeLocator().locate(path: folder).bsdName,
            let whole = DiskNaming.wholeDisk(ofSlice: bsd)
        else { return XCTFail("the temporary folder has no device") }
        // Read-only IORegistry lookup of This Mac's own storage. A machine
        // whose registry hides it answers nil; any answer is a whole disk.
        if let physical = IOKitPhysicalDiskResolver().physicalWholeDisk(of: whole) {
            XCTAssertEqual(DiskNaming.wholeDisk(ofSlice: physical), physical)
        }
    }

    func testAnInternalVolumeIsNeverRememberedAsBackingADrive() {
        let state = EjectDecisionState()
        // A path that exists, so the locator can read its volume.
        let model = LoadedModel(category: .chat, name: "m", path: NSTemporaryDirectory())
        let volume = LiveModelVolumeLocator().locate(path: model.path)
        state.apply([EjectDecisionState.Entry(model: model, volume: volume)], ticket: state.nextTicket())
        XCTAssertTrue(state.snapshot.isEmpty)
    }
}
