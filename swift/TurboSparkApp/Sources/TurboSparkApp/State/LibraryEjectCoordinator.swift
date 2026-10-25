import Foundation

/// How an unmount-and-eject ended.
enum EjectResult: Equatable, Sendable {
    case ejected
    /// The system's own words for why the drive stayed (a system string, not
    /// ours: it is shown as it came).
    case failed(String)
}

/// Unmounts a disk and ejects it. The live one is DiskArbitration; tests use a
/// fake so no real disk is ever touched.
protocol DiskEjecting: Sendable {
    func unmountAndEject(_ disk: DiskIdentity) async -> EjectResult
}

/// Finds the volume a model path lives on. Blocking (it reads the disk), so the
/// coordinator calls it only off the main actor.
protocol ModelVolumeLocating: Sendable {
    func locate(path: String) -> ModelVolume
}

/// The sessions that hold models, as the coordinator sees them.
///
/// The live implementation is over `AppModel` and holds it weakly. Every
/// member runs on the main actor, so nothing here may be called from the
/// approval callback: the coordinator reads this to FILL its snapshot and to
/// act on a cycle, never to decide.
@MainActor
protocol LoadedModelSource: AnyObject {
    /// Every model a chat, server or audio session holds open right now. A
    /// model something still keeps mapped is listed even when the app already
    /// dropped its own reference: this is the truth the coordinator checks
    /// after an unload.
    func loadedModels() -> [LoadedModel]
    /// Stops generation that uses `models` (requirement 6.2), and the work
    /// that keeps those sessions alive. Returns once it has stopped, or has
    /// stopped being waited for.
    func stopGeneration(in category: LoadedModelCategory, models: [LoadedModel]) async
    /// Asks the sessions that hold `models` to let go, with the app's own
    /// unload path, for a bounded time. A request, not a result: the
    /// coordinator reads `loadedModels()` afterwards and tells the user only
    /// what is gone.
    func unload(_ category: LoadedModelCategory, models: [LoadedModel]) async
    /// Brackets one whole release (the stop step, the unload and the check
    /// that follows). The coordinator balances every `beginRelease` with an
    /// `endRelease` on every exit, so a source can hold state for exactly the
    /// span in which sessions are being taken down. `models` is the same list
    /// both times.
    func beginRelease(of models: [LoadedModel])
    func endRelease(of models: [LoadedModel])
}

extension LoadedModelSource {
    func beginRelease(of models: [LoadedModel]) {}
    func endRelease(of models: [LoadedModel]) {}
}

/// What the coordinator wants the user told. The words, in the user's
/// language, are the app's to write (`EjectNoticeText`).
enum EjectNotice: Equatable, Sendable {
    case ejected(models: [String])
    case ejectFailed(models: [String], reason: String)
    /// The drive left without an eject (requirement 6.3).
    case removedWithoutEject(models: [String])
    /// A session still holds these models after the unload was attempted, so
    /// the drive was not ejected. Names ONLY the models still held.
    case unloadBlocked(models: [String])
}

/// Anything that can host the unmount-approval delegate: the process-wide
/// `VolumeMonitor`, or a fake in tests.
protocol UnmountApprovalHosting: AnyObject {
    var delegate: UnmountApprovalDelegate? { get set }
}

extension VolumeMonitor: UnmountApprovalHosting {}

/// Lets a drive with loaded models eject in one action (requirements 6.1 to
/// 6.3), and ends the sessions of a drive that was pulled instead (6.3).
///
/// **TWO HALVES, ON TWO THREADS.**
/// - The DECISION is `unmountRequested(for:)`. DiskArbitration waits at most
///   100 ms for it, on a worker thread, so it reads only `decisions`, a
///   lock-protected snapshot this coordinator keeps current. It never touches
///   the main actor, the engine or a disk, and it returns at once.
/// - The WORK is `run(_:)`, on the main actor, after the dissent has gone out:
///   stop generation, unload each session, ASK the sessions what they still
///   hold, and only when nothing is held issue the unmount and eject ONCE and
///   say so. A model that is still held ends the request with a failure
///   notice and no eject, and the user is never told "unloaded" for it. The
///   re-issue asks the approval callback again; the cycle is marked so that
///   second request is approved.
///
/// **THE SNAPSHOT IS REFRESHED, NOT CONSULTED LIVE.** `scheduleRefresh()` runs
/// whenever the loaded set changes (the app model's publishers and the audio
/// workspace call it) and on mount events. A model loaded a few milliseconds
/// before an eject may not be in it yet; the eject then proceeds as it did
/// before this feature existed.
///
/// **A SURPRISE REMOVAL TOUCHES NO RECORD.** `didUnmount` ends sessions and
/// shows a message; it never writes the registry, a model row or the library
/// list (requirement 6.4). `LibraryState` marks the library not connected on
/// its own, from the same event.
///
/// The monitor holds its delegate weakly, so the app model keeps this strongly.
/// `stop()` makes it approve everything, which is what a replaced coordinator
/// (a vault lock builds a new model) must do.
final class LibraryEjectCoordinator: UnmountApprovalDelegate, @unchecked Sendable {
    /// The status DiskArbitration carries with the busy dissent. A system
    /// string, not UI.
    static let dissentReason = "TurboSpark is unloading models from this drive"

    let decisions = EjectDecisionState()
    private let source: any LoadedModelSource
    private let ejector: DiskEjecting
    private let locator: ModelVolumeLocating
    private let present: @MainActor (EjectNotice) -> Void
    private var eventTask: Task<Void, Never>?
    /// Events fully handled so far. A test seam for waiting on the stream.
    private(set) var handledEventCount = 0

    @MainActor
    init(
        source: any LoadedModelSource, ejector: DiskEjecting, locator: ModelVolumeLocating,
        present: @escaping @MainActor (EjectNotice) -> Void
    ) {
        self.source = source
        self.ejector = ejector
        self.locator = locator
        self.present = present
    }

    // MARK: - The decision (worker thread, 100 ms)

    func unmountRequested(for disk: DiskIdentity) -> UnmountDissent? {
        switch decisions.decide(disk) {
        case .approve:
            return nil
        case .dissent:
            return UnmountDissent(reason: Self.dissentReason)
        case .dissentAndStart(let cycle):
            // The dissent is returned first; the work follows on the main
            // actor. The state, not `self`, is captured for the bookkeeping so
            // a coordinator that goes away mid-flight still balances it.
            let decisions = self.decisions
            decisions.runScheduled()
            Task { @MainActor [weak self] in
                guard let self else {
                    decisions.finish(cycle.id)
                    decisions.runEnded()
                    return
                }
                await self.run(cycle)
            }
            return UnmountDissent(reason: Self.dissentReason)
        }
    }

    // MARK: - The snapshot

    /// Re-reads the loaded set and where each model lives, then replaces the
    /// snapshot. The disk is read off the main actor.
    @MainActor
    func refreshSnapshot() async {
        guard !decisions.isStopped else { return }
        let ticket = decisions.nextTicket()
        let models = source.loadedModels()
        let locator = self.locator
        let entries = await Task.detached(priority: .utility) {
            models.map { EjectDecisionState.Entry(model: $0, volume: locator.locate(path: $0.path)) }
        }.value
        decisions.apply(entries, ticket: ticket)
    }

    /// Asks for a refresh without waiting for it. The loaded set changes inside
    /// a property's setter (a Combine publisher fires before the value is
    /// stored), so the read is deferred to a main-actor task, which runs after
    /// the setter has finished. Callable from any context.
    func scheduleRefresh() {
        Task { @MainActor [weak self] in await self?.refreshSnapshot() }
    }

    // MARK: - Events (requirement 6.3)

    /// Drains `events` with one `for await`. A second call is ignored: each
    /// `VolumeMonitor.events()` call is its own stream, and one that nobody
    /// reads is only a stranded buffer. Never stops the monitor.
    @MainActor
    func start(events: AsyncStream<VolumeEvent>) {
        guard eventTask == nil, !decisions.isStopped else { return }
        eventTask = Task { [weak self] in
            for await event in events {
                guard let self else { return }
                await self.handle(event)
            }
        }
    }

    /// `willUnmount` fires on EVERY unmount attempt, including one this
    /// coordinator dissents from, so the volume may stay mounted: only
    /// `didUnmount` means a drive is gone.
    @MainActor
    func handle(_ event: VolumeEvent) async {
        defer { handledEventCount += 1 }
        switch event {
        case .willUnmount:
            break
        case .mounted, .renamed:
            await refreshSnapshot()
        case .didUnmount(let volume):
            await volumeRemoved(volume)
        }
    }

    /// A drive left. If a session still holds a model from it, the files are
    /// already gone: stop what is running, drop the sessions (references only,
    /// nothing reads the drive), and advise the user to eject next time.
    @MainActor
    private func volumeRemoved(_ volume: VolumeDescriptor) async {
        guard !decisions.isStopped else { return }
        switch decisions.noteVolumeGone(volume) {
        case .ownEject, .duringUnload:
            // This coordinator's own cycle owns the notice.
            return
        case .unrelated:
            break
        }
        // The live set, not the snapshot: a model this coordinator already
        // unloaded (its own eject, or an earlier event) is not held any more,
        // which is what makes a repeated event a no-op.
        let hit = source.loadedModels().filter { decisions.model($0, wasOn: volume) }
        guard !hit.isEmpty else { return }
        guard let result = await release(hit) else { return }
        // Only what is really gone is advised as torn down. A model a session
        // still holds is not named: saying so would be false.
        if !result.gone.isEmpty {
            present(.removedWithoutEject(models: Self.names(of: result.gone)))
        }
        await refreshSnapshot()
    }

    // MARK: - The work

    /// What the sessions hold after they were asked to let go.
    private struct Released {
        /// Models no session holds any more.
        let gone: [LoadedModel]
        /// Models a session still holds, which keeps their files open.
        let held: [LoadedModel]
    }

    /// Stops generation on every category first, then unloads each, so no
    /// model is released while it is still producing tokens (requirement 6.2).
    /// Then ASKS the sessions what they hold: an unload is a request, not a
    /// result (a background task, or a request the engine is serving, can
    /// keep a model mapped), and the user is told only what is true.
    /// Returns nil when the coordinator was stopped part way, and nothing
    /// further should happen.
    @MainActor
    private func release(_ models: [LoadedModel]) async -> Released? {
        // Bracketed for every exit, early ones included: the source holds
        // state (the chat may not start a turn) for exactly this span.
        source.beginRelease(of: models)
        defer { source.endRelease(of: models) }
        let categories = Set(models.map(\.category)).sorted()
        for category in categories {
            guard !decisions.isStopped else { return nil }
            await source.stopGeneration(in: category, models: models.filter { $0.category == category })
        }
        for category in categories {
            guard !decisions.isStopped else { return nil }
            await source.unload(category, models: models.filter { $0.category == category })
        }
        guard !decisions.isStopped else { return nil }
        // The live set, not the snapshot: the snapshot is what the sessions
        // held when it was last refreshed.
        let live = Set(source.loadedModels().map(Self.identity))
        return Released(
            gone: models.filter { !live.contains(Self.identity($0)) },
            held: models.filter { live.contains(Self.identity($0)) })
    }

    /// A model is the same model whatever name it is shown under.
    private static func identity(_ model: LoadedModel) -> String {
        "\(model.category.rawValue):" + EjectDecisionState.normalized(model.path)
    }

    /// One user request: release the sessions, unmount and eject once, say so.
    /// A model that is still held ends the request without an eject: the drive
    /// would only report busy, and "unloaded" would be untrue.
    @MainActor
    private func run(_ cycle: EjectDecisionState.Cycle) async {
        defer {
            decisions.finish(cycle.id)
            decisions.runEnded()
        }
        guard let released = await release(cycle.models) else { return }
        let gone = Self.names(of: released.gone)
        switch decisions.beginReissue(cycle.id, unloaded: released.gone, stillHeld: !released.held.isEmpty) {
        case .abandoned:
            return
        case .volumeGone:
            if !released.gone.isEmpty { present(.removedWithoutEject(models: gone)) }
            await refreshSnapshot()
            return
        case .held:
            present(.unloadBlocked(models: Self.names(of: released.held)))
            await refreshSnapshot()
            return
        case .proceed:
            break
        }
        let result = await ejector.unmountAndEject(cycle.disk)
        guard !decisions.isStopped else { return }
        switch result {
        case .ejected:
            present(.ejected(models: gone))
        case .failed(let reason):
            present(.ejectFailed(models: gone, reason: reason))
        }
        await refreshSnapshot()
    }

    /// Each distinct name once, in the order the models were found.
    private static func names(of models: [LoadedModel]) -> [String] {
        var seen = Set<String>()
        return models.map(\.name).filter { seen.insert($0).inserted }
    }

    // MARK: - Lifecycle

    /// Ends the event loop and makes every later request approve. The monitor
    /// itself keeps running: stopping it makes the next unmount of any volume
    /// wait about ten seconds.
    @MainActor
    func stop() {
        eventTask?.cancel()
        eventTask = nil
        decisions.stop()
    }

    /// Waits until every cycle body has finished. A test seam.
    func waitForIdle(timeout: TimeInterval = 5) async {
        let deadline = Date().addingTimeInterval(timeout)
        while decisions.pendingRunCount > 0, Date() < deadline {
            try? await Task.sleep(nanoseconds: 2_000_000)
        }
    }
}
