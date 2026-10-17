import Foundation

/// The session kinds that can keep a model's files open (requirement 6.1).
enum LoadedModelCategory: Int, CaseIterable, Comparable, Sendable {
    case chat
    case server
    case audio

    static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
}

/// One model a session holds open right now.
struct LoadedModel: Equatable, Hashable, Sendable {
    let category: LoadedModelCategory
    /// What the toast calls it. Raw and untrusted: show it through
    /// `UntrustedText.display`.
    let name: String
    /// The path the session was opened with.
    let path: String
}

/// Where a loaded model's files live, read OFF the main actor when the
/// snapshot is refreshed, so the approval callback never reads a disk.
struct ModelVolume: Equatable, Sendable {
    /// The model path with links resolved.
    let realPath: String
    /// The volume's mount point ("/Volumes/T7"). Nil when it could not be read.
    let mountPoint: String?
    /// The volume's own device ("disk5s1"). Nil when it could not be read.
    let bsdName: String?
    /// Whole disks the volume sits on: for an APFS volume the synthesized
    /// container ("disk5") and, when it could be resolved, the physical store's
    /// disk ("disk4"). Used for a request that names a whole disk, and to find
    /// the models of sibling volumes of one drive.
    let diskNames: Set<String>

    static func unresolved(_ path: String) -> ModelVolume {
        ModelVolume(realPath: path, mountPoint: nil, bsdName: nil, diskNames: [])
    }
}

/// Everything the approval callback is allowed to know, behind one lock.
///
/// **THE CALLBACK READS THIS AND NOTHING ELSE.** `decide` takes the lock, looks
/// at an array and a dictionary, and returns: it never touches the main actor,
/// the engine or the disk, because DiskArbitration waits at most 100 ms for the
/// answer and a late one approves.
///
/// **MATCHING RULES.** A loaded model backs the requested disk when
///  1. its path (or its link-resolved path) lies under the request's volume
///     path, compared on path components (`/Volumes/T70` is not inside
///     `/Volumes/T7`), or
///  2. its volume's device name equals the request's (a renamed volume), or
///  3. the request names a WHOLE disk and no volume, and that disk is one of
///     the disks the model's volume sits on.
/// The whole-disk name of a VOLUME request never decides alone: an APFS
/// container reports the same synthesized disk for every volume in it, and
/// unmounting a sibling volume does not touch the model.
final class EjectDecisionState: @unchecked Sendable {
    struct Entry: Equatable, Sendable {
        let model: LoadedModel
        let volume: ModelVolume
    }

    /// One user request being served: from the dissent until the re-issued
    /// unmount has finished.
    struct Cycle: Equatable, Sendable {
        let id: UUID
        /// The request that started it; the ejector unmounts this disk.
        let disk: DiskIdentity
        /// Every model on the drive, across sessions.
        let models: [LoadedModel]
        /// Disk and volume names that belong to this cycle's drive.
        let names: Set<String>
        let mountPoints: [String]
        /// True once the coordinator itself is unmounting. An unmount request
        /// that arrives now is the coordinator's own and must be approved.
        var reissuing = false
        /// The drive was pulled before the re-issue: there is nothing to eject.
        var volumeGone = false
    }

    enum Decision: Equatable, Sendable {
        case approve
        /// Dissent for a request an earlier dissent already started a cycle for.
        case dissent
        case dissentAndStart(Cycle)
    }

    /// Where a `didUnmount` stands against the cycles in flight.
    enum Removal: Equatable, Sendable {
        case unrelated
        /// The coordinator's own unmount completing; it presents its own notice.
        case ownEject
        /// The drive left before the coordinator unmounted it.
        case duringUnload
    }

    enum Reissue: Equatable, Sendable {
        case proceed
        case volumeGone
        /// A session still holds a model of this drive: an eject would only
        /// report busy, so the cycle ends without one.
        case held
        /// The coordinator was stopped: do nothing more.
        case abandoned
    }

    /// An entry with its paths and disk names normalized once, at refresh time.
    private struct Stored {
        let entry: Entry
        let path: String
        let realPath: String
        let mountPoint: String?
        let names: Set<String>
    }

    private struct Request {
        let volumePath: String?
        let bsd: String?
        let whole: String?

        init(_ disk: DiskIdentity) {
            volumePath = disk.volumeURL.map { EjectDecisionState.normalized($0.path) }
            bsd = disk.bsdName
            whole = disk.wholeDiskBSDName
        }

        var names: Set<String> { Set([bsd, whole].compactMap { $0 }) }
        /// A disk object with no volume: the physical store or an unmounted disk.
        var isWholeDisk: Bool { volumePath == nil && bsd != nil && bsd == whole }
        var isInternal: Bool { volumePath.map(EjectDecisionState.isInternalMount) ?? false }
    }

    private let lock = NSLock()
    private var stored: [Stored] = []
    private var issuedTickets = 0
    private var appliedTicket = 0
    private var cycles: [UUID: Cycle] = [:]
    private var pendingRuns = 0
    private var stopped = false

    // MARK: Paths

    /// Purely lexical: no empty, `.` or `..` components and no trailing slash.
    /// It never touches the file system, because the approval callback runs
    /// this on every request and a stat on a hung volume would hold it past
    /// its 100 ms (`standardizedFileURL` stats for a `/private` prefix).
    static func normalized(_ path: String) -> String {
        var parts: [Substring] = []
        for part in path.split(separator: "/", omittingEmptySubsequences: true) {
            if part == "." { continue }
            if part == ".." {
                if !parts.isEmpty { parts.removeLast() }
                continue
            }
            parts.append(part)
        }
        return "/" + parts.joined(separator: "/")
    }

    /// Containment on path components: `/Volumes/T70/x` is not inside
    /// `/Volumes/T7`.
    static func path(_ path: String, isWithin container: String) -> Bool {
        if path == container { return true }
        let prefix = container.hasSuffix("/") ? container : container + "/"
        return path.hasPrefix(prefix)
    }

    /// The volumes that make up This Mac's own disk. A request for one of them
    /// is never dissented from, whatever is loaded: a model in the home folder
    /// is "within" the root volume by prefix, and shutdown unmounts these.
    static func isInternalMount(_ path: String) -> Bool {
        path == "/" || path == "/System/Volumes" || path.hasPrefix("/System/Volumes/")
    }

    // MARK: Snapshot

    /// Takes a place in line before the models are read, so a slow refresh
    /// that started earlier cannot overwrite a newer one.
    func nextTicket() -> Int {
        lock.lock(); defer { lock.unlock() }
        issuedTickets += 1
        return issuedTickets
    }

    /// Replaces the snapshot, unless a newer one is already in. Models on This
    /// Mac's disk are dropped here: they can never match.
    func apply(_ fresh: [Entry], ticket: Int) {
        let rows = fresh.compactMap { entry -> Stored? in
            if let mount = entry.volume.mountPoint, Self.isInternalMount(Self.normalized(mount)) { return nil }
            var names = entry.volume.diskNames
            if let bsd = entry.volume.bsdName { names.insert(bsd) }
            return Stored(
                entry: entry, path: Self.normalized(entry.model.path),
                realPath: Self.normalized(entry.volume.realPath),
                mountPoint: entry.volume.mountPoint.map(Self.normalized), names: names)
        }
        lock.lock(); defer { lock.unlock() }
        guard !stopped, ticket > appliedTicket else { return }
        appliedTicket = ticket
        stored = rows
    }

    var snapshot: [Entry] {
        lock.lock(); defer { lock.unlock() }
        return stored.map(\.entry)
    }

    // MARK: Decisions

    private func matches(_ row: Stored, _ request: Request) -> Bool {
        if let volumePath = request.volumePath {
            if Self.path(row.path, isWithin: volumePath) || Self.path(row.realPath, isWithin: volumePath) {
                return true
            }
            if let mount = row.mountPoint, mount == volumePath { return true }
        }
        if let bsd = request.bsd, let own = row.entry.volume.bsdName, bsd == own { return true }
        if request.isWholeDisk, let bsd = request.bsd, row.entry.volume.diskNames.contains(bsd) { return true }
        return false
    }

    private func covers(_ cycle: Cycle, path: String?, names: Set<String>) -> Bool {
        if let path, cycle.mountPoints.contains(where: { Self.path(path, isWithin: $0) }) { return true }
        return !names.isDisjoint(with: cycle.names)
    }

    /// The approval answer for one unmount request. Pure bookkeeping under a
    /// lock; see the type's note.
    func decide(_ disk: DiskIdentity) -> Decision {
        lock.lock(); defer { lock.unlock() }
        guard !stopped else { return .approve }
        let request = Request(disk)
        if request.isInternal { return .approve }

        // A cycle already serves this drive. Its own re-issue (and the unmount
        // of each of its volumes) is approved, or the coordinator would
        // dissent from itself forever. Before the re-issue, a sibling volume
        // that still holds a model is dissented from as part of the same user
        // request, and one that holds nothing is approved as ever.
        if let cycle = cycles.values.first(where: { covers($0, path: request.volumePath, names: request.names) }) {
            if cycle.reissuing { return .approve }
            return stored.contains { matches($0, request) } ? .dissent : .approve
        }

        let hits = stored.indices.filter { matches(stored[$0], request) }
        guard !hits.isEmpty else { return .approve }

        // Everything on the drive goes in one cycle, including a model on
        // another volume of the same drive: the eject needs all of them gone.
        var names = request.names
        for index in hits { names.formUnion(stored[index].names) }
        let hitSet = Set(hits)
        let members = stored.indices.filter { hitSet.contains($0) || !stored[$0].names.isDisjoint(with: names) }
        var mountPoints: [String] = []
        for path in [request.volumePath] + members.map({ stored[$0].mountPoint }) {
            if let path, !mountPoints.contains(path) { mountPoints.append(path) }
        }
        for index in members { names.formUnion(stored[index].names) }

        let cycle = Cycle(
            id: UUID(), disk: disk, models: members.map { stored[$0].entry.model }, names: names,
            mountPoints: mountPoints)
        cycles[cycle.id] = cycle
        return .dissentAndStart(cycle)
    }

    /// Called once the sessions were asked to let go. Forgets the models that
    /// really are gone (a model a session still holds STAYS in the snapshot,
    /// so the drive keeps being dissented from until it is free) and, when
    /// nothing is held, marks the cycle as the one issuing the unmount, so the
    /// approval request that unmount causes is answered "approve".
    func beginReissue(_ id: UUID, unloaded: [LoadedModel], stillHeld: Bool) -> Reissue {
        lock.lock(); defer { lock.unlock() }
        guard !stopped, var cycle = cycles[id] else { return .abandoned }
        let released = Set(unloaded)
        stored.removeAll { released.contains($0.entry.model) }
        if cycle.volumeGone { return .volumeGone }
        if stillHeld { return .held }
        cycle.reissuing = true
        cycles[id] = cycle
        return .proceed
    }

    func finish(_ id: UUID) {
        lock.lock(); defer { lock.unlock() }
        cycles.removeValue(forKey: id)
    }

    /// Tells the cycles about a `didUnmount`, and says whether one of them
    /// owns it.
    func noteVolumeGone(_ volume: VolumeDescriptor) -> Removal {
        lock.lock(); defer { lock.unlock() }
        let path = Self.normalized(volume.url.path)
        let names = Set([volume.bsdName].compactMap { $0 })
        for (id, var cycle) in cycles where covers(cycle, path: path, names: names) {
            if cycle.reissuing { return .ownEject }
            cycle.volumeGone = true
            cycles[id] = cycle
            return .duringUnload
        }
        return .unrelated
    }

    /// Whether `model` was held open from `volume`: by path, by the path with
    /// links resolved when the snapshot knew it, or by the volume's device.
    func model(_ model: LoadedModel, wasOn volume: VolumeDescriptor) -> Bool {
        lock.lock(); defer { lock.unlock() }
        let path = Self.normalized(volume.url.path)
        if Self.isInternalMount(path) { return false }
        if Self.path(Self.normalized(model.path), isWithin: path) { return true }
        guard let row = stored.first(where: { $0.entry.model.path == model.path }) else { return false }
        if Self.path(row.realPath, isWithin: path) { return true }
        if let bsd = volume.bsdName, bsd == row.entry.volume.bsdName { return true }
        return false
    }

    // MARK: Bookkeeping

    var activeCycleCount: Int {
        lock.lock(); defer { lock.unlock() }
        return cycles.count
    }

    /// Cycle bodies scheduled and not yet finished. Unlike `activeCycleCount`
    /// it survives `stop()`, so a test can wait for an abandoned cycle to end.
    var pendingRunCount: Int {
        lock.lock(); defer { lock.unlock() }
        return pendingRuns
    }

    func runScheduled() {
        lock.lock(); pendingRuns += 1; lock.unlock()
    }

    func runEnded() {
        lock.lock(); pendingRuns -= 1; lock.unlock()
    }

    var isStopped: Bool {
        lock.lock(); defer { lock.unlock() }
        return stopped
    }

    /// After this every request is approved and nothing is remembered.
    func stop() {
        lock.lock(); defer { lock.unlock() }
        stopped = true
        stored = []
        cycles = [:]
    }
}
