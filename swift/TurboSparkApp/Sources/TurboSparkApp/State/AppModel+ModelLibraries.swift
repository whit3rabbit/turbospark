import Combine
import Foundation
import TurboSpark

/// Where a new install goes when the user has not chosen (requirement 2.2).
struct DefaultInstallDestination: Equatable {
    /// `ModelLibrary.id` of the destination: the default library's, or
    /// `internal` for This Mac.
    let libraryId: String
    /// The destination library as listed. Nil only before the first listing
    /// has landed, when the destination is This Mac by default.
    let library: ModelLibrary?
    /// The default library is set but its drive is not connected, so the
    /// destination fell back to This Mac and the picker must say so.
    let fallbackBecauseNotConnected: Bool
    /// The library that is not connected, for that message. Raw and
    /// untrusted: show it through `UntrustedText.display`.
    let notConnectedLibraryName: String?
}

/// Libraries as the app sees them: which are connected, how fast their links
/// are, how much room they have, which models are loadable, and which newly
/// connected library to offer (design.md "LibraryState").
///
/// **IT NEVER SHOUTS ABOUT A DRIVE.** An unplugged library, a listing that
/// failed and a registry write that failed are state and a log line, not an
/// alert (requirements 3.3 and 3.4). Nothing here holds a reference to
/// `AppModel`, so there is no `error` or toast to set. An action the user asked
/// for (`addLibrary`, `accept`) throws to its caller instead, because that
/// refusal is an answer to a question.
///
/// **EVERY ENGINE CALL IS OFF THE MAIN ACTOR.** A listing reads the mount table,
/// probes a drive and may write the registry. `refresh()` runs the service and
/// the probe in a detached task and applies the result on the main actor.
///
/// **TWO LISTS, ON PURPOSE.** `AppModel.installed` is still the available-only
/// list every picker and gate reads, fed by `refreshModels()`. `allInstalled`
/// here also holds rows whose library is not connected or whose folder is gone,
/// and `loadableInstalled` is its available subset. Re-pointing the views is
/// task 6.x; until then the existing UI cannot see an offline row. A view that
/// shows library state observes `model.libraryState` directly: `AppModel` does
/// not forward its changes, which would redraw every screen on each mount.
@MainActor
final class LibraryState: ObservableObject {
    typealias Sleep = @Sendable (Duration) async throws -> Void
    typealias FreeSpaceReader = @Sendable (URL) -> Int64?

    /// Requirements 4.1 and 4.6 allow five seconds end to end; a burst of
    /// mount events (a hub with three drives) collapses into one refresh.
    static let refreshDebounce: Duration = .milliseconds(500)
    static let internalLibraryId = "internal"

    // MARK: Published state

    /// Every library, This Mac first, as the engine last listed them.
    @Published private(set) var libraries: [ModelLibrary] = []
    /// Link kind and rate per AVAILABLE library, by library id.
    @Published private(set) var connections: [String: DriveConnection] = [:]
    /// Free bytes per AVAILABLE library, by library id
    /// (`volumeAvailableCapacityForImportantUsageKey`). A drive whose capacity
    /// cannot be read has no entry, never a zero.
    @Published private(set) var freeBytes: [String: Int64] = [:]
    /// Decimal MB/s from the speed test, per library. An entry goes when its
    /// volume unmounts (requirement 5.6).
    @Published private(set) var measuredSpeeds: [String: Double] = [:]
    /// Every text row, including `.notConnected` and `.missing` ones.
    @Published private(set) var allInstalled: [InstalledModel] = []
    /// The offer on screen and the ones in line behind it.
    @Published private(set) var offers = LibraryOfferQueue()
    /// A store a past Move Store moved to a drive that is not connected, whose
    /// conversion into a library is waiting for it (requirement 9.3). Set and
    /// cleared by `MoveStoreMigration` only.
    @Published private(set) var pendingMovedStore: PendingMovedStore?

    /// The id of the placeholder row `displayLibraries` appends for
    /// `pendingMovedStore`. A real library id is a v4 UUID, which this can
    /// never equal, and the placeholder has no registry entry.
    static let pendingMovedStoreId = "pending-moved-store"

    /// The libraries to LIST, with a not-connected placeholder named after the
    /// folder at the end while a moved store waits for its drive.
    ///
    /// **THE PLACEHOLDER IS A ROW, NOT A LIBRARY.** It is never in `libraries`,
    /// so `destinationDefault`, `library(for:)`, `library(containing:)`, the
    /// offers and every install picker (which read `libraries`) cannot see it.
    /// It has no root, is not internal, never the default, and `remove` must
    /// not be offered for it (`isPendingMovedStore`): the engine has no entry
    /// to remove.
    var displayLibraries: [ModelLibrary] {
        guard let pending = pendingMovedStore else { return libraries }
        return libraries + [
            ModelLibrary(
                id: Self.pendingMovedStoreId, name: pending.name, root: nil, available: false)
        ]
    }

    /// Whether `library` is the placeholder row rather than a listed library.
    static func isPendingMovedStore(_ library: ModelLibrary) -> Bool {
        library.id == pendingMovedStoreId
    }

    /// The offer to add a library found on a drive that just connected.
    var pendingOffer: LibraryOffer? { offers.pending }
    var queuedOffers: [LibraryOffer] { offers.queued }

    /// Rows that can be opened now. Pickers and load gates use this, never
    /// `allInstalled`.
    var loadableInstalled: [InstalledModel] {
        allInstalled.filter { $0.availability == .available }
    }

    /// Why the last refresh was partial, escaped for a log. For diagnostics:
    /// it is deliberately not shown anywhere.
    private(set) var lastRefreshFailure: String?
    /// Events fully handled so far. A test seam for waiting on the stream.
    private(set) var handledEventCount = 0

    // MARK: Collaborators

    /// The engine seam. Later tasks (installs, moves, import) call it off the
    /// main actor, then `await refresh()`.
    let service: ModelLibraryService
    let probe: DriveLinkProbing
    private let dismissalsURL: URL
    private let debounce: Duration
    private let sleep: Sleep
    private let freeSpace: FreeSpaceReader

    // MARK: Work in flight

    private var eventTask: Task<Void, Never>?
    private var debounceTask: Task<Void, Never>?
    private var refreshRunner: Task<Void, Never>?
    private var refreshRequested = false
    private var discoveryTasks: [UUID: Task<Void, Never>] = [:]
    /// True once a listing has succeeded. An offer needs it: without the list a
    /// library the user already added cannot be told from a new one.
    private var librariesLoaded = false

    private static let dismissalsLabel: String.LocalizationValue = "Library offer dismissals"
    /// Marker ids the user dismissed. Read on first use, so an `AppModel` built
    /// in a test never touches the file unless an offer is considered.
    private lazy var dismissedIds: Set<String> = Set(
        AppJSONStore.load([String].self, from: dismissalsURL, label: LibraryState.dismissalsLabel) ?? [])

    init(
        service: ModelLibraryService = LiveModelLibraryService(),
        probe: DriveLinkProbing = DriveLinkProbe(),
        dismissalsURL: URL = AppStorageRoot.file("library_offer_dismissals.json"),
        debounce: Duration = LibraryState.refreshDebounce,
        sleep: @escaping Sleep = { try await Task.sleep(for: $0) },
        freeSpace: @escaping FreeSpaceReader = LibraryState.availableCapacity
    ) {
        self.service = service
        self.probe = probe
        self.dismissalsURL = dismissalsURL
        self.debounce = debounce
        self.sleep = sleep
        self.freeSpace = freeSpace
    }

    // MARK: - Lifecycle

    var isRunning: Bool { eventTask != nil }

    /// Starts refreshing: once now, then on each volume event.
    ///
    /// **ONE `for await` PER STREAM.** An `AsyncStream` ends when its consumer
    /// leaves the loop, and each call to `VolumeMonitor.events()` is a fresh
    /// stream, so this takes the stream it will drain and a second call is
    /// ignored (it would only strand a stream nobody reads). It never stops the
    /// monitor: `VolumeMonitor.stop()` belongs to app termination alone, because
    /// removing the workspace observers makes the next unmount of ANY volume on
    /// the machine wait about ten seconds.
    func start(events: AsyncStream<VolumeEvent>) {
        guard eventTask == nil else { return }
        eventTask = Task { [weak self] in
            for await event in events {
                guard let self else { return }
                await self.handle(event)
            }
        }
        // The launch refresh does not wait for the debounce.
        Task { [weak self] in await self?.refresh() }
    }

    /// Stops consuming events and cancels the pending timer and discoveries.
    /// The monitor itself is left running.
    func stop() {
        eventTask?.cancel()
        eventTask = nil
        debounceTask?.cancel()
        debounceTask = nil
        for task in discoveryTasks.values { task.cancel() }
        discoveryTasks.removeAll()
    }

    /// Settles discoveries and refreshes that are running, not a debounce
    /// timer that has not fired.
    func waitForPendingWork() async {
        while true {
            let discoveries = Array(discoveryTasks.values)
            let runner = refreshRunner
            if discoveries.isEmpty, runner == nil { return }
            for task in discoveries { await task.value }
            if let runner { await runner.value }
        }
    }

    /// Waits for the refresh the debounce timer started, after it fired.
    func finishScheduledRefresh() async {
        await debounceTask?.value
        await refreshRunner?.value
    }

    // MARK: - Events

    /// Applies one volume event.
    ///
    /// `willUnmount` is ignored: it fires on EVERY unmount attempt, including
    /// one the approval callback dissents from, so the volume may stay
    /// mounted. Only `didUnmount` means a drive is gone.
    func handle(_ event: VolumeEvent) async {
        defer { handledEventCount += 1 }
        switch event {
        case .mounted(let volume):
            scheduleRefresh()
            discoverOffer(on: volume)

        case .willUnmount:
            break

        case .didUnmount(let volume):
            // The speed goes first and without waiting: it was measured on a
            // drive that is no longer there, and nothing may read it in the
            // meantime.
            dropMeasuredSpeeds(on: volume)
            await probe.invalidate(volume: volume.url)
            scheduleRefresh()

        case .renamed(let old, let new):
            // The probe caches by mount path until told otherwise, and both
            // paths may be cached. The speed belongs to the old name's drive
            // reading; the refresh re-resolves the library at its new root.
            dropMeasuredSpeeds(on: old)
            await probe.invalidate(volume: old.url)
            await probe.invalidate(volume: new.url)
            scheduleRefresh()
        }
    }

    /// Debounced trailing-edge refresh: each event restarts the timer.
    private func scheduleRefresh() {
        debounceTask?.cancel()
        let sleep = self.sleep
        let window = debounce
        debounceTask = Task { [weak self] in
            do { try await sleep(window) } catch { return }
            guard let self, !Task.isCancelled else { return }
            await self.refresh()
        }
    }

    // MARK: - Refresh

    /// Reloads libraries, availability, links, free space and the installed
    /// rows, off the main actor. Never throws and never raises an app error:
    /// a failure keeps what was known and is logged.
    ///
    /// Requests that arrive while a refresh runs fold into one more run after
    /// it, so the state a late event asked for is never skipped.
    func refresh() async {
        refreshRequested = true
        if refreshRunner == nil {
            refreshRunner = Task { [weak self] in await self?.drainRefreshRequests() }
        }
        await refreshRunner?.value
    }

    private func drainRefreshRequests() async {
        while refreshRequested {
            refreshRequested = false
            let snapshot = await Self.load(service: service, probe: probe, freeSpace: freeSpace)
            apply(snapshot)
        }
        refreshRunner = nil
    }

    /// What one refresh read. A nil part failed and leaves the old value.
    private struct Snapshot: Sendable {
        var libraries: [ModelLibrary]?
        var installed: [InstalledModel]?
        var connections: [String: DriveConnection] = [:]
        var freeBytes: [String: Int64] = [:]
        var failures: [String] = []
    }

    private nonisolated static func load(
        service: ModelLibraryService, probe: DriveLinkProbing, freeSpace: @escaping FreeSpaceReader
    ) async -> Snapshot {
        // The engine blocks (mount table, drive probes, an `installed.json`
        // write), and so does reading a drive's capacity, so none of it runs on
        // the caller's executor.
        var snapshot = await Task.detached(priority: .utility) { () -> Snapshot in
            var read = Snapshot()
            do {
                read.libraries = try service.libraries()
            } catch {
                read.failures.append("libraries: \(UntrustedText.display("\(error)", limit: 300))")
            }
            do {
                read.installed = try service.installedWithAvailability()
            } catch {
                read.failures.append("installed: \(UntrustedText.display("\(error)", limit: 300))")
            }
            for library in read.libraries ?? [] where library.available {
                guard let root = library.root else { continue }
                if let free = freeSpace(URL(fileURLWithPath: root, isDirectory: true)) {
                    read.freeBytes[library.id] = free
                }
            }
            return read
        }.value

        // The probe is an actor and answers from a cache after the first read.
        for library in snapshot.libraries ?? [] where library.available {
            guard let root = library.root else { continue }
            snapshot.connections[library.id] = await probe.connection(
                for: URL(fileURLWithPath: root, isDirectory: true))
        }
        return snapshot
    }

    private func apply(_ snapshot: Snapshot) {
        if snapshot.failures.isEmpty {
            lastRefreshFailure = nil
        } else {
            let line = snapshot.failures.joined(separator: "; ")
            lastRefreshFailure = line
            FileHandle.standardError.write(Data("TurboSpark: model libraries refresh was partial: \(line)\n".utf8))
        }

        // Assigned only when different: a published write redraws every
        // observer, and most refreshes change nothing.
        if let listed = snapshot.libraries {
            if libraries != listed { libraries = listed }
            librariesLoaded = true
            if connections != snapshot.connections { connections = snapshot.connections }
            if freeBytes != snapshot.freeBytes { freeBytes = snapshot.freeBytes }

            // A speed is only meaningful while its drive is there. The unmount
            // event drops it at once; this catches a drive that left while the
            // app was not looking.
            let connected = Set(listed.filter { $0.available && !$0.isInternal }.map(\.id))
            let kept = measuredSpeeds.filter { connected.contains($0.key) }
            if kept.count != measuredSpeeds.count { measuredSpeeds = kept }

            // A library added by other means no longer needs its offer.
            let listedIds = Set(listed.map(\.id))
            offers.prune { listedIds.contains($0.markerId) }
        }
        if let rows = snapshot.installed, rows != allInstalled { allInstalled = rows }
    }

    /// The capacity the system would let the user use, which for a removable
    /// drive is what is free. Nil when it cannot be read.
    nonisolated static func availableCapacity(at url: URL) -> Int64? {
        (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage
    }

    // MARK: - Lookups

    /// The library a row belongs to, listed or not connected. Nil for a custom
    /// folder (`pull --out`) and for a library that is no longer listed.
    func library(for model: InstalledModel) -> ModelLibrary? {
        switch model.locationKind {
        case .thisMac:
            return libraries.first { $0.isInternal }
        case .library:
            guard let id = model.library else { return nil }
            return libraries.first { $0.id == id }
        case .custom, .unknown:
            return nil
        }
    }

    /// The connected library whose folder holds `path`. An offline library has
    /// no root, so it never matches.
    func library(containing path: String) -> ModelLibrary? {
        let target = Self.normalized(path)
        return libraries
            .compactMap { library -> (ModelLibrary, String)? in
                guard let root = library.root else { return nil }
                let normalizedRoot = Self.normalized(root)
                return Self.path(target, isWithin: normalizedRoot) ? (library, normalizedRoot) : nil
            }
            .max { $0.1.count < $1.1.count }?
            .0
    }

    func connection(for library: ModelLibrary) -> DriveConnection? {
        connections[library.id]
    }

    func measuredSpeed(for libraryId: String) -> Double? {
        measuredSpeeds[libraryId]
    }

    /// Keeps a speed test's result for a connected external library. A figure
    /// for an unknown or unplugged library (the drive left while the test ran),
    /// or one that is not a positive number, is ignored.
    func recordMeasuredSpeed(_ mbps: Double, for libraryId: String) {
        guard mbps.isFinite, mbps > 0 else { return }
        guard libraries.contains(where: { $0.id == libraryId && $0.available && !$0.isInternal }) else {
            return
        }
        measuredSpeeds[libraryId] = mbps
    }

    /// The external libraries on `volume`, matched by path and, when both are
    /// known, by the volume UUID the link was probed on.
    private func libraries(on volume: VolumeDescriptor) -> [ModelLibrary] {
        let mount = Self.normalized(volume.url.path)
        return libraries.filter { library in
            guard !library.isInternal else { return false }
            if let uuid = volume.uuid, let linked = connections[library.id]?.volumeUUID, uuid == linked {
                return true
            }
            guard let root = library.root else { return false }
            return Self.path(Self.normalized(root), isWithin: mount)
        }
    }

    private func dropMeasuredSpeeds(on volume: VolumeDescriptor) {
        let gone = Set(libraries(on: volume).map(\.id))
        guard !gone.isEmpty else { return }
        let kept = measuredSpeeds.filter { !gone.contains($0.key) }
        if kept.count != measuredSpeeds.count { measuredSpeeds = kept }
    }

    private nonisolated static func normalized(_ path: String) -> String {
        URL(fileURLWithPath: path).standardizedFileURL.path
    }

    /// Containment on path components: `/Volumes/T70/x` is not inside
    /// `/Volumes/T7`.
    private nonisolated static func path(_ path: String, isWithin container: String) -> Bool {
        if path == container { return true }
        let prefix = container.hasSuffix("/") ? container : container + "/"
        return path.hasPrefix(prefix)
    }

    // MARK: - Default destination (requirement 2.2)

    /// The default library when it is connected, otherwise This Mac, flagged
    /// when the default was set but its drive is away. No default set, or This
    /// Mac as the default, is not a fallback.
    var destinationDefault: DefaultInstallDestination {
        let thisMac = libraries.first { $0.isInternal }
        guard let preferred = libraries.first(where: { $0.isDefault }) else {
            return DefaultInstallDestination(
                libraryId: Self.internalLibraryId, library: thisMac,
                fallbackBecauseNotConnected: false, notConnectedLibraryName: nil)
        }
        if preferred.isInternal || preferred.available {
            return DefaultInstallDestination(
                libraryId: preferred.id, library: preferred,
                fallbackBecauseNotConnected: false, notConnectedLibraryName: nil)
        }
        return DefaultInstallDestination(
            libraryId: Self.internalLibraryId, library: thisMac,
            fallbackBecauseNotConnected: true, notConnectedLibraryName: preferred.name)
    }

    /// Lists, or stops listing, the not-connected placeholder for a moved store.
    /// Only `MoveStoreMigration` calls this.
    func setPendingMovedStore(_ pending: PendingMovedStore?) {
        if pendingMovedStore != pending { pendingMovedStore = pending }
    }

    // MARK: - Changing libraries

    /// Adds the library at `path` through the engine (off the main actor) and
    /// refreshes. Throws the engine's refusal: it answers a user's action, so
    /// it is the caller's to show, not state.
    @discardableResult
    func addLibrary(path: String, name: String? = nil) async throws -> AddedLibrary {
        let service = self.service
        let added = try await Task.detached(priority: .userInitiated) {
            try service.add(path: path, name: name)
        }.value
        await refresh()
        return added
    }

    /// Stops listing the library `id` and its models through the engine (off
    /// the main actor) and refreshes. Never touches a file. Throws the engine's
    /// refusal; the list is refreshed either way, because a refusal such as
    /// `unknown_library` means another window already changed it.
    func removeLibrary(id: String) async throws {
        let service = self.service
        do {
            try await Task.detached(priority: .userInitiated) { try service.remove(id: id) }.value
        } catch {
            await refresh()
            throw error
        }
        await refresh()
    }

    /// Makes `id` (or `internal` for This Mac) the library new downloads go to.
    /// Same threading and refresh rules as `removeLibrary`.
    func setDefaultLibrary(id: String) async throws {
        let service = self.service
        do {
            try await Task.detached(priority: .userInitiated) { try service.setDefault(id: id) }.value
        } catch {
            await refresh()
            throw error
        }
        await refresh()
    }

    // MARK: - Offers (requirements 4.3, 4.4)

    /// Looks for a library on a newly mounted volume, off the main actor, and
    /// offers it if it is new. Several mounts at once run side by side; the
    /// queue keeps them to one offer on screen.
    private func discoverOffer(on volume: VolumeDescriptor) {
        let id = UUID()
        discoveryTasks[id] = Task { [weak self] in
            await self?.runDiscovery(on: volume)
            self?.discoveryTasks.removeValue(forKey: id)
        }
    }

    private func runDiscovery(on volume: VolumeDescriptor) async {
        let mount = Self.normalized(volume.url.path)
        // The drive of a listed, connected library has nothing new to offer,
        // and this saves a read.
        let isListedDrive = libraries.contains { library in
            guard !library.isInternal, let root = library.root else { return false }
            return Self.path(Self.normalized(root), isWithin: mount)
        }
        if isListedDrive { return }

        // A network share is never a library (requirement 1.4), and reading a
        // marker over one could hang.
        guard await probe.connection(for: volume.url).isLocal else { return }

        if !librariesLoaded { await refresh() }
        guard librariesLoaded, !Task.isCancelled else { return }

        let service = self.service
        let marker: LibraryMarkerSummary?
        do {
            marker = try await Task.detached(priority: .utility) {
                try service.discover(volumeRoot: mount)
            }.value
        } catch {
            // No marker is readable here. That is not something to tell the
            // user about a drive they just plugged in.
            FileHandle.standardError.write(
                Data("TurboSpark: library discovery failed: \(UntrustedText.display("\(error)", limit: 300))\n".utf8))
            return
        }
        guard let marker, !Task.isCancelled else { return }
        consider(marker, volumeRoot: mount)
    }

    /// Publishes an offer for `marker` unless the library is listed, was
    /// dismissed, or is already waiting. The listing is the authority, not the
    /// mount path: a known library on a drive renamed since has the same id.
    private func consider(_ marker: LibraryMarkerSummary, volumeRoot: String) {
        guard !libraries.contains(where: { $0.id == marker.id }) else { return }
        guard !dismissedIds.contains(marker.id) else { return }
        offers.add(
            LibraryOffer(
                volumeRoot: volumeRoot, markerId: marker.id, name: marker.name,
                modelCount: marker.modelCount, readOnly: marker.readOnly))
    }

    /// The user does not want this library offered. Remembered by marker id in
    /// `library_offer_dismissals.json`, so it survives a relaunch and a rename
    /// of the drive. The library can still be added from settings.
    func dismiss(_ offer: LibraryOffer) {
        dismissedIds.insert(offer.markerId)
        AppJSONStore.save(dismissedIds.sorted(), to: dismissalsURL, label: Self.dismissalsLabel)
        offers.remove(markerId: offer.markerId)
    }

    /// Adds the offered library and refreshes. A refusal is thrown and the
    /// offer stays, so the user can try again or dismiss it.
    func accept(_ offer: LibraryOffer) async throws {
        try await addLibrary(path: offer.volumeRoot)
        offers.remove(markerId: offer.markerId)
    }
}

// MARK: - Process-wide volume monitor

/// The one workspace volume monitor of the process.
///
/// A vault lock discards the `AppModel` and a later unlock builds another, so
/// the monitor cannot belong to a model: stopping it would be the removal of
/// workspace observers in a long-lived process, which makes the next unmount of
/// any volume on the machine wait about ten seconds. It is created on first use,
/// handed out as fresh event streams, and stopped only at app termination.
/// Other consumers (the eject coordinator, task 5.11) take their own stream
/// from `events()` and register as the monitor's approval delegate.
enum LibraryVolumeMonitor {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var monitor: VolumeMonitor?

    /// The shared monitor, created and started on first use.
    static var shared: VolumeMonitor {
        lock.lock(); defer { lock.unlock() }
        if let monitor { return monitor }
        let created = VolumeMonitor()
        created.start()
        monitor = created
        return created
    }

    /// A fresh stream of volume events.
    static func events() -> AsyncStream<VolumeEvent> {
        shared.events()
    }

    /// Stops the monitor. Call only from `applicationWillTerminate`; a monitor
    /// that was never started is not created just to be stopped.
    static func stopAtTermination() {
        lock.lock(); defer { lock.unlock() }
        monitor?.stop()
    }
}

extension AppModel {
    /// Starts library monitoring: one refresh now and one on each volume
    /// event. Called when the window appears; calling it again does nothing.
    /// Tests do not call it, so no `AppModel` built under XCTest registers
    /// with the workspace.
    ///
    /// `events` gives a fresh stream per call (each consumer drains its own);
    /// a test passes a stream it controls.
    ///
    /// It also starts the eject coordinator (tasks 5.11 and 5.12): its own
    /// event stream, and its registration as the monitor's approval delegate.
    /// `approvalHost` is the monitor in the app and a fake in a test, so a test
    /// never registers with DiskArbitration. A vault lock builds a new model,
    /// which registers its own coordinator; `shutdown()` of the old one removes
    /// only its own registration and never stops the monitor.
    func startLibraryMonitoring(
        events: () -> AsyncStream<VolumeEvent> = { LibraryVolumeMonitor.events() },
        approvalHost: () -> UnmountApprovalHosting = { LibraryVolumeMonitor.shared },
        ejector: DiskEjecting = DiskArbitrationEjector(),
        volumeLocator: ModelVolumeLocating = LiveModelVolumeLocator(),
        loadedModels: (any LoadedModelSource)? = nil,
        offerLedger: LibraryOfferAnnouncementLedger = .process
    ) {
        guard !libraryState.isRunning else { return }
        // Before the first event, so an offer found at launch is announced.
        startLibraryOfferAnnouncements(ledger: offerLedger)
        // Before the first listing lands, so the pickers' list follows it too: a
        // successful Add (which does not reload models) and every mount, unmount
        // and rename reach `installed` through this (task 6.8).
        startInstalledListFollowing()
        libraryState.start(events: events())
        startMoveStoreMigration(events: events())
        startEjectCoordination(
            events: events(), host: approvalHost(), ejector: ejector, locator: volumeLocator,
            source: loadedModels)
    }

    /// Posts ONE info toast for each add-library offer, when it becomes the offer
    /// on screen (task 6.6, requirement 4.3). The ledger keeps an id to one
    /// toast per process; calling this again does nothing.
    ///
    /// The toast is the only place an offer speaks outside its banner, and it is
    /// never the app error: an offer is not a failure.
    func startLibraryOfferAnnouncements(ledger: LibraryOfferAnnouncementLedger = .process) {
        guard libraryOfferAnnouncer == nil else { return }
        let announcer = LibraryOfferAnnouncer(libraryState: libraryState, ledger: ledger) { [weak self] sentence in
            // Two lines of text: long enough to read, and VoiceOver hears the
            // whole sentence either way (`showToast` announces it).
            self?.showToast(sentence, style: .info, duration: Self.libraryOfferToastSeconds)
        }
        announcer.start()
        libraryOfferAnnouncer = announcer
    }

    /// How long the offer's toast stays up.
    static let libraryOfferToastSeconds: TimeInterval = 8

    /// Converts a store a past Move Store recorded into a library: once now,
    /// and again on each mount while it is pending. It takes its own stream of
    /// the process-wide monitor, drains it with one `for await`, and never stops
    /// the monitor.
    ///
    /// The seams are parameters so a test can run it against a fake record and
    /// engine without registering with the workspace.
    func startMoveStoreMigration(
        events: AsyncStream<VolumeEvent>,
        record: MovedStoreRecord = LiveMovedStoreRecord(),
        pendingURL: URL = MoveStoreMigration.defaultPendingURL,
        homeRoot: @escaping @Sendable () -> String = { ModelStorageManager.defaultTurboSparkStoreRoot }
    ) {
        guard moveStoreMigration == nil else { return }
        let migration = MoveStoreMigration(
            service: libraryState.service, libraryState: libraryState, record: record,
            pendingURL: pendingURL, homeRoot: homeRoot,
            // The picker list was read at launch, before the import listed the
            // moved store's models.
            onImported: { [weak self] in self?.refreshModels() })
        moveStoreMigration = migration
        migration.start(events: events)
    }
}
