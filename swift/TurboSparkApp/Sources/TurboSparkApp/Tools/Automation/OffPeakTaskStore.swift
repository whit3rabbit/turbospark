import Foundation

/// Where an off-peak task came from. A model-origin row carries a marker in
/// the pane, and a turn started by an off-peak task may not queue another
/// one (see `OffPeakTaskStore.enqueue`).
public enum OffPeakOrigin: String, Codable, Sendable {
    case user
    case model
}

/// One task's lifecycle. `admitted` from the design draft is folded into
/// `running`: the claim is the persisted write, so there is no separate
/// admitted state to recover. `skipped` means the target chat is gone.
public enum OffPeakTaskState: String, Codable, Sendable {
    case queued
    case running
    case completed
    case interrupted
    case cancelled
    case skipped

    /// A state that holds a slot against the job cap.
    public var isActive: Bool { self == .queued || self == .running }
}

/// One dispatch attempt. `endedAt` is nil while the run is in flight.
public struct OffPeakRunRecord: Codable, Equatable, Sendable {
    public var startedAt: Date
    public var endedAt: Date?
    /// "completed", "stopped", "approval_needed", "app_quit", "chat_missing",
    /// or "crash_recovered". Stored as data, shown through the catalog.
    public var outcome: String?

    public init(startedAt: Date = Date(), endedAt: Date? = nil, outcome: String? = nil) {
        self.startedAt = startedAt
        self.endedAt = endedAt
        self.outcome = outcome
    }
}

/// A prompt the user (or the model) wants run eventually, while the machine
/// is idle. Delivered into the chat it was queued from.
public struct AppOffPeakTask: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var prompt: String
    public var chatID: UUID
    public var origin: OffPeakOrigin
    public var state: OffPeakTaskState
    public var createdAt: Date
    /// Newest first, capped at `runHistoryLimit`.
    public var runs: [OffPeakRunRecord]

    public static let runHistoryLimit = 10

    public init(
        id: String = UUID().uuidString.prefix(8).lowercased().description,
        prompt: String,
        chatID: UUID,
        origin: OffPeakOrigin,
        state: OffPeakTaskState = .queued,
        createdAt: Date = Date(),
        runs: [OffPeakRunRecord] = []
    ) {
        self.id = id
        self.prompt = prompt
        self.chatID = chatID
        self.origin = origin
        self.state = state
        self.createdAt = createdAt
        self.runs = runs
    }

    /// Tolerant decode: a row written by a newer build must not sink the file.
    /// An unknown state loads as `interrupted`, never as `queued`, so a row
    /// this build cannot read is offered for a deliberate re-run instead of
    /// being dispatched unattended.
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        prompt = try c.decodeIfPresent(String.self, forKey: .prompt) ?? ""
        chatID = try c.decodeIfPresent(UUID.self, forKey: .chatID) ?? UUID()
        origin = (try? c.decodeIfPresent(OffPeakOrigin.self, forKey: .origin)) ?? .user
        state = (try? c.decodeIfPresent(OffPeakTaskState.self, forKey: .state)) ?? .interrupted
        createdAt = try c.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        runs = try c.decodeIfPresent([OffPeakRunRecord].self, forKey: .runs) ?? []
    }
}

/// The persistent queue behind the off-peak tools and the Scheduled Tasks
/// pane's off-peak section.
///
/// **INDEPENDENT OF `CronScheduler`.** Own file (`offpeak_tasks.json`), own
/// lock, own timer. The job cap is an atomic check-and-insert under THIS
/// store's lock, so the pane and the model's tool cannot both pass the check
/// and overshoot it.
///
/// **THE CLAIM IS WRITTEN BEFORE THE RUN.** `claimNext` saves `running`
/// before the prompt is delivered, so a crash leaves a record a relaunch can
/// tell from a task that never started (`queued`). Launch recovery turns
/// every `running` row into `interrupted`; the in-process shutdown path does
/// not run on a crash, so the sweep cannot depend on it.
public final class OffPeakTaskStore: @unchecked Sendable {
    public static let shared = OffPeakTaskStore()

    /// Active (queued or running) tasks allowed at once. Fixed, not a setting.
    public static let defaultJobCap = 20
    /// Terminal rows kept for the pane's history; the oldest are dropped.
    static let terminalRetention = 20
    /// A prompt longer than this is refused rather than silently truncated.
    public static let maxPromptCharacters = 8_000

    private let lock = NSLock()
    private var tasks: [AppOffPeakTask] = []
    private let fileURL: URL
    private let jobCap: Int

    public init(directory: URL = AppStorageRoot.directory, jobCap: Int = OffPeakTaskStore.defaultJobCap) {
        self.fileURL = directory.appendingPathComponent("offpeak_tasks.json")
        self.jobCap = jobCap
        load()
    }

    // MARK: - Persistence

    private let writeLock = NSLock()
    private var snapshotVersion = 0
    private var writtenVersion = 0

    private func load() {
        guard let archive = AppJSONStore.load(
            [AppOffPeakTask].self, from: fileURL, label: "Off-peak tasks")
        else { return }
        lock.lock()
        tasks = archive
        // Crash or force-quit recovery: a persisted claim with no completion.
        for index in tasks.indices where tasks[index].state == .running {
            tasks[index].state = .interrupted
            if let last = tasks[index].runs.indices.first, tasks[index].runs[last].endedAt == nil {
                tasks[index].runs[last].endedAt = Date()
                tasks[index].runs[last].outcome = "crash_recovered"
            }
        }
        lock.unlock()
        save()
    }

    /// Snapshot under `lock`, versioned write under `writeLock`, so a stale
    /// snapshot never overwrites a newer one (same shape as the cron store).
    private func save() {
        lock.lock()
        snapshotVersion += 1
        let version = snapshotVersion
        let snapshot = tasks
        lock.unlock()
        writeLock.lock()
        defer { writeLock.unlock() }
        guard version > writtenVersion else { return }
        writtenVersion = version
        AppJSONStore.save(snapshot, to: fileURL, label: "Off-peak tasks")
    }

    // MARK: - Queries

    /// Queue order: FIFO for queued rows, then history.
    public func list() -> [AppOffPeakTask] {
        lock.lock()
        defer { lock.unlock() }
        return tasks
    }

    public func task(id: String) -> AppOffPeakTask? {
        lock.lock()
        defer { lock.unlock() }
        return tasks.first { $0.id == id }
    }

    public func runningTask() -> AppOffPeakTask? {
        lock.lock()
        defer { lock.unlock() }
        return tasks.first { $0.state == .running }
    }

    public var activeCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return tasks.filter { $0.state.isActive }.count
    }

    // MARK: - Mutations

    /// Appends a queued task. Throws when the cap is full, the prompt is
    /// empty or oversized. A task whose chat has
    /// a RUNNING off-peak task is refused for model-origin callers: a
    /// scheduled turn must not queue more scheduled work unattended.
    @discardableResult
    public func enqueue(prompt: String, chatID: UUID, origin: OffPeakOrigin) throws -> AppOffPeakTask {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw Self.error(1, "Missing 'prompt': say what to run when the machine is idle.")
        }
        guard trimmed.count <= Self.maxPromptCharacters else {
            throw Self.error(2, "The prompt is longer than \(Self.maxPromptCharacters) characters.")
        }
        let task: AppOffPeakTask
        do {
            lock.lock()
            defer { lock.unlock() }
            if origin == .model,
                tasks.contains(where: { $0.state == .running && $0.chatID == chatID })
            {
                throw Self.error(4, "An off-peak task cannot queue another off-peak task.")
            }
            guard tasks.filter({ $0.state.isActive }).count < jobCap else {
                throw Self.error(5, "The off-peak queue is full (\(jobCap) active tasks). Cancel or wait for one to finish.")
            }
            task = AppOffPeakTask(prompt: trimmed, chatID: chatID, origin: origin)
            tasks.append(task)
            pruneTerminalLocked()
        }
        save()
        return task
    }

    /// Cancels a queued task. A running task is not cancellable here: its
    /// generation is stopped in its chat, which ends the run as stopped.
    public func cancel(id: String) throws {
        lock.lock()
        guard let index = tasks.firstIndex(where: { $0.id == id }) else {
            lock.unlock()
            throw Self.error(6, "No off-peak task with id '\(id)'.")
        }
        guard tasks[index].state == .queued else {
            let state = tasks[index].state
            lock.unlock()
            throw Self.error(7, state == .running
                ? "Task '\(id)' is running; stop its generation in its chat instead."
                : "Task '\(id)' is not queued (\(state.rawValue)).")
        }
        tasks[index].state = .cancelled
        lock.unlock()
        save()
    }

    /// Removes a finished row from the history. Active rows stay.
    @discardableResult
    public func remove(id: String) -> Bool {
        lock.lock()
        let before = tasks.count
        tasks.removeAll { $0.id == id && !$0.state.isActive }
        let removed = tasks.count != before
        lock.unlock()
        if removed { save() }
        return removed
    }

    /// Re-runs an interrupted task by putting it at the TAIL of the queue.
    /// Counts against the cap like a fresh enqueue.
    public func requeue(id: String) throws {
        lock.lock()
        guard let index = tasks.firstIndex(where: { $0.id == id }) else {
            lock.unlock()
            throw Self.error(6, "No off-peak task with id '\(id)'.")
        }
        guard tasks[index].state == .interrupted else {
            lock.unlock()
            throw Self.error(8, "Only an interrupted task can be re-run.")
        }
        guard tasks.filter({ $0.state.isActive }).count < jobCap else {
            lock.unlock()
            throw Self.error(5, "The off-peak queue is full (\(jobCap) active tasks).")
        }
        var task = tasks.remove(at: index)
        task.state = .queued
        tasks.append(task)
        lock.unlock()
        save()
    }

    // MARK: - Dispatch bookkeeping (dispatcher surface)

    /// Claims the first queued task (FIFO) that `eligible` accepts and
    /// persists `running` BEFORE returning it. Nil when another task is
    /// already running: one at a time.
    func claimNext(now: Date = Date(), eligible: (AppOffPeakTask) -> Bool) -> AppOffPeakTask? {
        lock.lock()
        guard !tasks.contains(where: { $0.state == .running }),
            let index = tasks.firstIndex(where: { $0.state == .queued && eligible($0) })
        else {
            lock.unlock()
            return nil
        }
        tasks[index].state = .running
        tasks[index].runs.insert(OffPeakRunRecord(startedAt: now), at: 0)
        if tasks[index].runs.count > AppOffPeakTask.runHistoryLimit {
            tasks[index].runs.removeLast(tasks[index].runs.count - AppOffPeakTask.runHistoryLimit)
        }
        let claimed = tasks[index]
        lock.unlock()
        save()
        return claimed
    }

    /// Undoes a claim whose delivery did not start (the chat became busy
    /// between the check and the send): back to `queued`, run entry dropped.
    func releaseClaim(id: String) {
        lock.lock()
        if let index = tasks.firstIndex(where: { $0.id == id && $0.state == .running }) {
            tasks[index].state = .queued
            if !tasks[index].runs.isEmpty { tasks[index].runs.removeFirst() }
        }
        lock.unlock()
        save()
    }

    /// Ends the running task. Closes its newest run entry.
    func finish(id: String, state: OffPeakTaskState, outcome: String, now: Date = Date()) {
        lock.lock()
        if let index = tasks.firstIndex(where: { $0.id == id && $0.state == .running }) {
            tasks[index].state = state
            if !tasks[index].runs.isEmpty {
                tasks[index].runs[0].endedAt = now
                tasks[index].runs[0].outcome = outcome
            }
            pruneTerminalLocked()
        }
        lock.unlock()
        save()
    }

    /// Marks a queued task whose chat no longer exists. Recorded, never
    /// dispatched.
    func skip(id: String, outcome: String, now: Date = Date()) {
        lock.lock()
        if let index = tasks.firstIndex(where: { $0.id == id && $0.state == .queued }) {
            tasks[index].state = .skipped
            tasks[index].runs.insert(
                OffPeakRunRecord(startedAt: now, endedAt: now, outcome: outcome), at: 0)
            pruneTerminalLocked()
        }
        lock.unlock()
        save()
    }

    private func pruneTerminalLocked() {
        let terminalIndices = tasks.indices.filter { !tasks[$0].state.isActive }
        let excess = terminalIndices.count - Self.terminalRetention
        guard excess > 0 else { return }
        // Oldest terminal rows sit earliest in the array.
        let drop = Set(terminalIndices.prefix(excess))
        tasks = tasks.enumerated().filter { !drop.contains($0.offset) }.map(\.element)
    }

    private static func error(_ code: Int, _ message: String) -> NSError {
        NSError(domain: "TurboSparkOffPeak", code: code, userInfo: [
            NSLocalizedDescriptionKey: message
        ])
    }
}
