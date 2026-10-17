import AppKit
import DiskArbitration
import XCTest
@testable import TurboSparkApp

/// Stands in for the DiskArbitration session: records start and stop, and lets
/// a test play the part of the DA callback by calling the handler the monitor
/// registered.
private final class FakeApprovalSource: UnmountApprovalSource, @unchecked Sendable {
    private let lock = NSLock()
    private var handler: (@Sendable (DiskIdentity) -> UnmountDissent?)?
    private var last: (@Sendable (DiskIdentity) -> UnmountDissent?)?
    private var starts = 0
    private var stops = 0

    var startCount: Int { lock.lock(); defer { lock.unlock() }; return starts }
    var stopCount: Int { lock.lock(); defer { lock.unlock() }; return stops }
    var isRegistered: Bool { lock.lock(); defer { lock.unlock() }; return handler != nil }
    /// The handler the monitor registered, kept after stop to play a DA
    /// callback that was already in flight when the session was torn down.
    var lastHandler: (@Sendable (DiskIdentity) -> UnmountDissent?)? { lock.lock(); defer { lock.unlock() }; return last }

    func start(_ handler: @escaping @Sendable (DiskIdentity) -> UnmountDissent?) {
        lock.lock(); defer { lock.unlock() }
        starts += 1
        self.handler = handler
        last = handler
    }

    func stop() {
        lock.lock(); defer { lock.unlock() }
        stops += 1
        handler = nil
    }

    /// The DA callback entry point: nil when nothing is registered, which is
    /// what DiskArbitration does for a callback that was unregistered.
    func requestUnmount(_ disk: DiskIdentity) -> UnmountDissent? {
        lock.lock()
        let current = handler
        lock.unlock()
        return current?(disk)
    }
}

/// An unmount delegate whose answer is a closure, counting how often it was asked.
private final class ScriptedDelegate: UnmountApprovalDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var disks: [DiskIdentity] = []
    private let script: @Sendable (DiskIdentity) -> UnmountDissent?

    init(_ script: @escaping @Sendable (DiskIdentity) -> UnmountDissent?) {
        self.script = script
    }

    var asked: [DiskIdentity] { lock.lock(); defer { lock.unlock() }; return disks }

    func unmountRequested(for disk: DiskIdentity) -> UnmountDissent? {
        lock.lock(); disks.append(disk); lock.unlock()
        return script(disk)
    }
}

final class VolumeMonitorTests: XCTestCase {
    private let t7 = URL(fileURLWithPath: "/Volumes/T7", isDirectory: true)
    private let t7Disk = DiskIdentity(
        bsdName: "disk4s1", wholeDiskBSDName: "disk4",
        volumeURL: URL(fileURLWithPath: "/Volumes/T7", isDirectory: true))

    // MARK: Fixtures

    private func makeMonitor(
        center: NotificationCenter = NotificationCenter(),
        source: FakeApprovalSource = FakeApprovalSource(),
        details: @escaping @Sendable (URL) -> VolumeDetails = { _ in VolumeDetails(bsdName: nil, uuid: nil) }
    ) -> (monitor: VolumeMonitor, center: NotificationCenter, source: FakeApprovalSource) {
        let monitor = VolumeMonitor(notificationCenter: center, approvalSource: source, readDetails: details)
        return (monitor, center, source)
    }

    private func post(
        _ center: NotificationCenter, _ name: Notification.Name, url: URL?, name volumeName: String? = nil,
        oldURL: URL? = nil, oldName: String? = nil
    ) {
        var info: [AnyHashable: Any] = [:]
        if let url { info[NSWorkspace.volumeURLUserInfoKey] = url }
        if let volumeName { info[NSWorkspace.localizedVolumeNameUserInfoKey] = volumeName }
        if let oldURL { info[NSWorkspace.oldVolumeURLUserInfoKey] = oldURL }
        if let oldName { info[NSWorkspace.oldLocalizedVolumeNameUserInfoKey] = oldName }
        center.post(name: name, object: nil, userInfo: info)
    }

    /// Takes `count` events, or whatever arrived before the timeout.
    private func collect(
        _ count: Int, from stream: AsyncStream<VolumeEvent>, timeout: TimeInterval = 3
    ) async -> [VolumeEvent] {
        await withTaskGroup(of: [VolumeEvent]?.self) { group in
            group.addTask {
                var events: [VolumeEvent] = []
                for await event in stream {
                    events.append(event)
                    if events.count == count { return events }
                }
                return events
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first ?? []
        }
    }

    /// True when nothing arrives within `window`.
    private func staysQuiet(_ stream: AsyncStream<VolumeEvent>, for window: TimeInterval = 0.3) async -> Bool {
        await collect(1, from: stream, timeout: window).isEmpty
    }

    private func elapsed(_ body: () -> Void) -> TimeInterval {
        let start = DispatchTime.now().uptimeNanoseconds
        body()
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
    }

    // MARK: Mount events

    func testMountNotificationYieldsMountedWithURLAndName() async {
        let (monitor, center, _) = makeMonitor()
        let stream = monitor.events()
        monitor.start()
        post(center, NSWorkspace.didMountNotification, url: t7, name: "T7")
        let events = await collect(1, from: stream)
        XCTAssertEqual(events, [.mounted(VolumeDescriptor(url: t7, name: "T7", bsdName: nil, uuid: nil))])
    }

    func testWillUnmountNotificationYieldsWillUnmount() async {
        let (monitor, center, _) = makeMonitor()
        let stream = monitor.events()
        monitor.start()
        post(center, NSWorkspace.willUnmountNotification, url: t7, name: "T7")
        let events = await collect(1, from: stream)
        XCTAssertEqual(events, [.willUnmount(VolumeDescriptor(url: t7, name: "T7", bsdName: nil, uuid: nil))])
    }

    func testDidUnmountNotificationYieldsDidUnmount() async {
        let (monitor, center, _) = makeMonitor()
        let stream = monitor.events()
        monitor.start()
        post(center, NSWorkspace.didUnmountNotification, url: t7, name: "T7")
        let events = await collect(1, from: stream)
        XCTAssertEqual(events, [.didUnmount(VolumeDescriptor(url: t7, name: "T7", bsdName: nil, uuid: nil))])
    }

    func testRenameNotificationYieldsTheOldAndTheNewVolume() async {
        let (monitor, center, _) = makeMonitor()
        let stream = monitor.events()
        monitor.start()
        let renamed = URL(fileURLWithPath: "/Volumes/T7 Models", isDirectory: true)
        post(center, NSWorkspace.didRenameVolumeNotification,
             url: renamed, name: "T7 Models", oldURL: t7, oldName: "T7")
        let events = await collect(1, from: stream)
        XCTAssertEqual(events, [.renamed(
            old: VolumeDescriptor(url: t7, name: "T7", bsdName: nil, uuid: nil),
            new: VolumeDescriptor(url: renamed, name: "T7 Models", bsdName: nil, uuid: nil))])
    }

    func testEventsArriveInTheOrderTheNotificationsWerePosted() async {
        let (monitor, center, _) = makeMonitor()
        let stream = monitor.events()
        monitor.start()
        let urls = (0..<25).map { URL(fileURLWithPath: "/Volumes/V\($0)", isDirectory: true) }
        for url in urls { post(center, NSWorkspace.didMountNotification, url: url, name: nil) }
        post(center, NSWorkspace.willUnmountNotification, url: urls[0], name: nil)
        post(center, NSWorkspace.didUnmountNotification, url: urls[0], name: nil)
        let events = await collect(urls.count + 2, from: stream)
        XCTAssertEqual(events.count, urls.count + 2)
        let mounted = events.compactMap { event -> URL? in
            if case .mounted(let volume) = event { return volume.url } else { return nil }
        }
        XCTAssertEqual(mounted, urls)
        guard events.count == urls.count + 2 else { return }
        XCTAssertEqual(events[urls.count].volume?.url, urls[0])
        if case .willUnmount = events[urls.count] {} else { XCTFail("expected willUnmount, got \(events[urls.count])") }
        if case .didUnmount = events[urls.count + 1] {} else { XCTFail("expected didUnmount, got \(events[urls.count + 1])") }
    }

    func testDetailsFromTheReaderRideOnTheEventsAndSurviveToDidUnmount() async {
        // After the unmount the volume is gone and the system can say nothing,
        // so the monitor must hand back what it learned while it was mounted.
        let readable = LockedBox(true)
        let (monitor, center, _) = makeMonitor(details: { _ in
            readable.value
                ? VolumeDetails(bsdName: "disk4s1", uuid: "UUID-T7")
                : VolumeDetails(bsdName: nil, uuid: nil)
        })
        let mounted = monitor.events()
        monitor.start()
        post(center, NSWorkspace.didMountNotification, url: t7, name: "T7")
        post(center, NSWorkspace.willUnmountNotification, url: t7, name: "T7")
        let first = await collect(2, from: mounted)
        // The volume is unreadable from here on, as it is once the OS has unmounted it.
        readable.value = false
        let unmounted = monitor.events()
        post(center, NSWorkspace.didUnmountNotification, url: t7, name: "T7")
        let second = await collect(1, from: unmounted)
        let expected = VolumeDescriptor(url: t7, name: "T7", bsdName: "disk4s1", uuid: "UUID-T7")
        XCTAssertEqual(first + second, [.mounted(expected), .willUnmount(expected), .didUnmount(expected)])
    }

    func testANotificationWithoutAVolumeURLIsIgnored() async {
        let (monitor, center, _) = makeMonitor()
        let stream = monitor.events()
        monitor.start()
        post(center, NSWorkspace.didMountNotification, url: nil, name: "Nameless")
        center.post(name: NSWorkspace.didUnmountNotification, object: nil, userInfo: nil)
        post(center, NSWorkspace.didMountNotification, url: t7, name: "T7")
        let events = await collect(1, from: stream)
        XCTAssertEqual(events, [.mounted(VolumeDescriptor(url: t7, name: "T7", bsdName: nil, uuid: nil))])
    }

    func testEveryConsumerSeesEveryEvent() async {
        let (monitor, center, _) = makeMonitor()
        let first = monitor.events()
        let second = monitor.events()
        monitor.start()
        post(center, NSWorkspace.didMountNotification, url: t7, name: "T7")
        async let a = collect(1, from: first)
        async let b = collect(1, from: second)
        let (left, right) = await (a, b)
        XCTAssertEqual(left.count, 1)
        XCTAssertEqual(left, right)
    }

    // MARK: Unmount approval

    func testTheDelegateIsAskedExactlyOnceWithTheDisksIdentity() {
        let (monitor, _, source) = makeMonitor()
        let delegate = ScriptedDelegate { _ in nil }
        monitor.delegate = delegate
        monitor.start()
        _ = source.requestUnmount(t7Disk)
        XCTAssertEqual(delegate.asked, [t7Disk])
    }

    func testApprovesWhenTheDelegateHasNoObjection() {
        let (monitor, _, source) = makeMonitor()
        let delegate = ScriptedDelegate { _ in nil }
        monitor.delegate = delegate
        monitor.start()
        XCTAssertNil(source.requestUnmount(t7Disk))
    }

    func testADissentIsReturnedAsADissent() {
        let (monitor, _, source) = makeMonitor()
        let delegate = ScriptedDelegate { _ in UnmountDissent(reason: "TurboSpark is unloading a model") }
        monitor.delegate = delegate
        monitor.start()
        XCTAssertEqual(source.requestUnmount(t7Disk), UnmountDissent(reason: "TurboSpark is unloading a model"))
        XCTAssertEqual(delegate.asked.count, 1)
    }

    func testASlowDelegateIsApprovedAtAboutOneHundredMilliseconds() {
        let (monitor, _, source) = makeMonitor()
        // It would dissent if it ever got to answer: the timeout must win.
        let delegate = ScriptedDelegate { _ in
            Thread.sleep(forTimeInterval: 0.5)
            return UnmountDissent(reason: "too late")
        }
        monitor.delegate = delegate
        monitor.start()
        var answer: UnmountDissent? = UnmountDissent(reason: "sentinel")
        let seconds = elapsed { answer = source.requestUnmount(t7Disk) }
        XCTAssertNil(answer, "a delegate that misses the deadline must be approved")
        XCTAssertLessThan(seconds, 0.4)
        XCTAssertGreaterThanOrEqual(seconds, 0.09, "it should have waited for the delegate first")
        XCTAssertEqual(delegate.asked.count, 1)
    }

    func testADelegateThatNeverAnswersIsApproved() {
        let (monitor, _, source) = makeMonitor()
        let gate = DispatchSemaphore(value: 0)
        // Let the stuck worker go when the test ends, so no thread is leaked.
        addTeardownBlock { gate.signal() }
        let delegate = ScriptedDelegate { _ in
            gate.wait()
            return UnmountDissent(reason: "never")
        }
        monitor.delegate = delegate
        monitor.start()
        var answer: UnmountDissent? = UnmountDissent(reason: "sentinel")
        let seconds = elapsed { answer = source.requestUnmount(t7Disk) }
        XCTAssertNil(answer)
        XCTAssertLessThan(seconds, 0.4)
    }

    func testADelegateThatNeedsTheBlockedMainThreadDoesNotDeadlockTheUnmount() {
        let (monitor, _, source) = makeMonitor()
        // The worst case for a delegate that hops to the main actor: main is
        // busy waiting for this very answer. The wait must give up, not hang.
        let delegate = ScriptedDelegate { _ in
            DispatchQueue.main.sync {}
            return UnmountDissent(reason: "from main")
        }
        monitor.delegate = delegate
        monitor.start()
        let finished = expectation(description: "approval answered while main was occupied")
        let box = LockedBox<UnmountDissent?>(UnmountDissent(reason: "sentinel"))
        let seconds = LockedBox<TimeInterval>(0)
        DispatchQueue.main.async {
            seconds.value = self.elapsed { box.value = source.requestUnmount(self.t7Disk) }
            finished.fulfill()
        }
        wait(for: [finished], timeout: 5)
        XCTAssertNil(box.value)
        XCTAssertLessThan(seconds.value, 0.4)
    }

    func testWithoutADelegateEveryUnmountIsApproved() {
        let (monitor, _, source) = makeMonitor()
        monitor.start()
        XCTAssertNil(source.requestUnmount(t7Disk))
    }

    func testTheDelegateIsNotKeptAlive() {
        let (monitor, _, source) = makeMonitor()
        weak var weakDelegate: ScriptedDelegate?
        autoreleasepool {
            let delegate = ScriptedDelegate { _ in UnmountDissent(reason: "busy") }
            weakDelegate = delegate
            monitor.delegate = delegate
        }
        monitor.start()
        XCTAssertNil(weakDelegate, "the monitor must hold its delegate weakly (the coordinator owns the monitor)")
        XCTAssertNil(source.requestUnmount(t7Disk), "a gone delegate means approve")
    }

    // MARK: Lifetime

    func testStartRegistersOnceAndStopUnregisters() {
        let (monitor, _, source) = makeMonitor()
        monitor.start()
        monitor.start()
        XCTAssertEqual(source.startCount, 1)
        XCTAssertTrue(source.isRegistered)
        monitor.stop()
        monitor.stop()
        XCTAssertEqual(source.stopCount, 1)
        XCTAssertFalse(source.isRegistered)
    }

    func testStopAndDeinitRemoveEveryObserverTheMonitorAdded() {
        // The generation guard would hide a leaked observer from the event
        // tests, so count the registrations on the center itself.
        let center = CountingCenter()
        var monitor: VolumeMonitor? = VolumeMonitor(notificationCenter: center, approvalSource: FakeApprovalSource())
        monitor?.start()
        XCTAssertEqual(center.addedCount, 4)
        XCTAssertEqual(center.removedCount, 0)
        monitor?.stop()
        XCTAssertEqual(center.removedCount, 4)

        monitor?.start()
        XCTAssertEqual(center.addedCount, 8)
        monitor = nil
        XCTAssertEqual(center.removedCount, 8, "deinit must remove the observers of a running monitor")
    }

    func testStopMakesLaterNotificationsProduceNothing() async {
        let (monitor, center, _) = makeMonitor()
        let stream = monitor.events()
        monitor.start()
        monitor.stop()
        post(center, NSWorkspace.didMountNotification, url: t7, name: "T7")
        post(center, NSWorkspace.didUnmountNotification, url: t7, name: "T7")
        let quiet = await staysQuiet(stream)
        XCTAssertTrue(quiet)
    }

    func testAfterStopTheDelegateIsNotAskedAndTheUnmountIsApproved() {
        let (monitor, _, source) = makeMonitor()
        let delegate = ScriptedDelegate { _ in UnmountDissent(reason: "busy") }
        monitor.delegate = delegate
        monitor.start()
        // A callback that was already in flight when stop ran still reaches the
        // handler; it must approve rather than ask a stopped monitor's delegate.
        let handler = source.lastHandler
        XCTAssertNotNil(handler)
        monitor.stop()
        XCTAssertNil(handler?(t7Disk))
        XCTAssertTrue(delegate.asked.isEmpty)
    }

    func testStartStopStartDeliversEventsAgain() async {
        let (monitor, center, source) = makeMonitor()
        let firstStream = monitor.events()
        monitor.start()
        post(center, NSWorkspace.didMountNotification, url: t7, name: "first")
        let first = await collect(1, from: firstStream)
        XCTAssertEqual(first.count, 1)
        monitor.stop()
        monitor.start()
        let secondStream = monitor.events()
        post(center, NSWorkspace.didMountNotification, url: t7, name: "second")
        let second = await collect(1, from: secondStream)
        XCTAssertEqual(second, [.mounted(VolumeDescriptor(url: t7, name: "second", bsdName: nil, uuid: nil))])
        XCTAssertEqual(source.startCount, 2)
        XCTAssertEqual(source.stopCount, 1)
        XCTAssertTrue(source.isRegistered)
    }

    func testTheMonitorDeallocatesAndUnregistersWhileStarted() {
        let source = FakeApprovalSource()
        weak var weakMonitor: VolumeMonitor?
        autoreleasepool {
            let center = NotificationCenter()
            let monitor = VolumeMonitor(notificationCenter: center, approvalSource: source)
            let delegate = ScriptedDelegate { _ in nil }
            monitor.delegate = delegate
            weakMonitor = monitor
            monitor.start()
            XCTAssertTrue(source.isRegistered)
        }
        XCTAssertNil(weakMonitor, "a started monitor must not be kept alive by its observers or its DA handler")
        XCTAssertEqual(source.stopCount, 1, "deinit must unregister the approval callback")
        XCTAssertFalse(source.isRegistered)
    }

    func testEventStreamsFinishWhenTheMonitorGoesAway() async {
        var monitor: VolumeMonitor? = VolumeMonitor(notificationCenter: NotificationCenter(), approvalSource: FakeApprovalSource())
        let stream = monitor!.events()
        monitor = nil
        var count = 0
        for await _ in stream { count += 1 }
        XCTAssertEqual(count, 0)
    }

    // MARK: Real DiskArbitration adapter

    func testTheRealSourceCreatesStartsAndStopsWithoutCrashing() {
        let source = DiskArbitrationApprovalSource()
        source.stop()
        source.start { _ in nil }
        source.start { _ in nil }
        source.stop()
        source.stop()
        source.start { _ in nil }
        source.stop()
    }

    func testARealMonitorStartsAndStops() {
        let monitor = VolumeMonitor()
        monitor.start()
        monitor.stop()
        monitor.start()
        monitor.stop()
    }

    func testTheRealSourceReleasesItsContextOnStop() {
        let source = DiskArbitrationApprovalSource()
        weak var weakContext: DiskArbitrationApprovalSource.Context?
        source.start { _ in nil }
        weakContext = source.activeContext
        XCTAssertNotNil(weakContext)
        source.stop()
        XCTAssertNil(weakContext, "stop must release the retained context pointer")
    }

    private func rootDisk() throws -> DADisk {
        let session = try XCTUnwrap(DASessionCreate(kCFAllocatorDefault))
        return try XCTUnwrap(DADiskCreateFromVolumePath(
            kCFAllocatorDefault, session, URL(fileURLWithPath: "/") as CFURL))
    }

    func testTheCallbackEntryAsksTheHandlerAndBuildsABusyDissenter() throws {
        let disk = try rootDisk()
        let seen = LockedBox<DiskIdentity?>(nil)
        let context = DiskArbitrationApprovalSource.Context { identity in
            seen.value = identity
            return UnmountDissent(reason: "TurboSpark is unloading a model")
        }
        let pointer = Unmanaged.passRetained(context).toOpaque()
        defer { Unmanaged<DiskArbitrationApprovalSource.Context>.fromOpaque(pointer).release() }

        let result = DiskArbitrationApprovalSource.unmountApprovalCallback(disk, pointer)
        let dissenter = try XCTUnwrap(result?.takeRetainedValue())
        XCTAssertEqual(DADissenterGetStatus(dissenter), DAReturn(kDAReturnBusy))
        XCTAssertEqual(DADissenterGetStatusString(dissenter) as String?, "TurboSpark is unloading a model")

        let identity = try XCTUnwrap(seen.value)
        XCTAssertEqual(identity.bsdName, DADiskGetBSDName(disk).map { String(cString: $0) })
        XCTAssertEqual(identity.volumeURL?.path, "/")
    }

    func testADiskIdentityNamesTheWholeDiskItBelongsTo() throws {
        // The whole disk's name is a C string owned by a temporary DADisk. It
        // once read back as garbage because the disk was released too early, so
        // check the shape and repeat the read.
        let disk = try rootDisk()
        for _ in 0..<200 {
            let identity = DiskIdentity(disk: disk)
            let bsd = try XCTUnwrap(identity.bsdName)
            let whole = try XCTUnwrap(identity.wholeDiskBSDName)
            XCTAssertNotNil(whole.range(of: #"^disk[0-9]+$"#, options: .regularExpression), "got \(whole)")
            XCTAssertTrue(bsd.hasPrefix(whole), "\(bsd) is not on \(whole)")
        }
    }

    func testTheCallbackEntryApprovesWithAnApprovingHandlerAndAfterDeactivation() throws {
        let disk = try rootDisk()
        let calls = LockedBox<Int>(0)
        let context = DiskArbitrationApprovalSource.Context { _ in calls.value += 1; return nil }
        let pointer = Unmanaged.passRetained(context).toOpaque()
        defer { Unmanaged<DiskArbitrationApprovalSource.Context>.fromOpaque(pointer).release() }

        XCTAssertNil(DiskArbitrationApprovalSource.unmountApprovalCallback(disk, pointer))
        XCTAssertEqual(calls.value, 1)

        context.deactivate()
        XCTAssertNil(DiskArbitrationApprovalSource.unmountApprovalCallback(disk, pointer))
        XCTAssertEqual(calls.value, 1, "a deactivated context must not run the handler")
        XCTAssertNil(DiskArbitrationApprovalSource.unmountApprovalCallback(disk, nil))
    }

    // MARK: Volume details

    func testTheDefaultReaderFindsTheRootVolumesBSDName() {
        let details = VolumeMonitor.defaultReadDetails(URL(fileURLWithPath: "/"))
        XCTAssertTrue(details.bsdName?.hasPrefix("disk") ?? false, "got \(String(describing: details.bsdName))")
        XCTAssertNotNil(details.uuid)
    }

    func testTheDefaultReaderGivesNothingForAVolumeThatIsGone() {
        let gone = URL(fileURLWithPath: "/Volumes/TurboSparkNoSuchVolume-\(UUID().uuidString)")
        let details = VolumeMonitor.defaultReadDetails(gone)
        XCTAssertNil(details.bsdName)
        XCTAssertNil(details.uuid)
    }
}

// MARK: Test helpers

/// A notification center that counts block observers added and removed.
private final class CountingCenter: NotificationCenter, @unchecked Sendable {
    private let lock = NSLock()
    private var added = 0
    private var removed = 0
    var addedCount: Int { lock.lock(); defer { lock.unlock() }; return added }
    var removedCount: Int { lock.lock(); defer { lock.unlock() }; return removed }

    override func addObserver(
        forName name: NSNotification.Name?, object obj: Any?, queue: OperationQueue?,
        using block: @escaping @Sendable (Notification) -> Void
    ) -> any NSObjectProtocol {
        lock.lock(); added += 1; lock.unlock()
        return super.addObserver(forName: name, object: obj, queue: queue, using: block)
    }

    override func removeObserver(_ observer: Any) {
        lock.lock(); removed += 1; lock.unlock()
        super.removeObserver(observer)
    }
}

private final class LockedBox<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value
    init(_ value: Value) { stored = value }
    var value: Value {
        get { lock.lock(); defer { lock.unlock() }; return stored }
        set { lock.lock(); stored = newValue; lock.unlock() }
    }
}

private extension VolumeEvent {
    var volume: VolumeDescriptor? {
        switch self {
        case .mounted(let volume), .willUnmount(let volume), .didUnmount(let volume): return volume
        case .renamed(_, let new): return new
        }
    }
}
