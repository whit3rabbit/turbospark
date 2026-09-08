import Foundation

/// One backgrounded shell command and everything needed to read or kill it.
///
/// The output buffer is the SAME one-shot design `ProcessExecutor` uses: one
/// dedicated reader thread per pipe draining continuously into a capped
/// buffer (state#66), both pipes merged into a single buffer so the text
/// interleaves in arrival order. A background command has no deadline -- the
/// model chose `run_in_background` precisely because the work outlives a
/// foreground timeout, and only completion or `KillShell` ends it.
final class BackgroundShellRecord: @unchecked Sendable {
    enum State: String {
        case running, completed, failed, killed
    }

    let id: String
    let command: String
    let displayDescription: String?
    let chatID: UUID?
    let startedAt = Date()
    let process: Process
    let outputBuffer: CappedOutputBuffer

    private let lock = NSLock()
    private var _state: State = .running
    private var _exitCode: Int32?

    init(id: String, command: String, description: String?, chatID: UUID?, process: Process, outputBuffer: CappedOutputBuffer) {
        self.id = id
        self.command = command
        self.displayDescription = description
        self.chatID = chatID
        self.process = process
        self.outputBuffer = outputBuffer
    }

    var state: State {
        lock.lock(); defer { lock.unlock() }
        return _state
    }

    var exitCode: Int32? {
        lock.lock(); defer { lock.unlock() }
        return _exitCode
    }

    /// Marks the run over. `kill()` sets `.killed` BEFORE terminating the
    /// process, and the completion handler refuses to overwrite a state that
    /// is no longer `.running` -- so a killed shell reads as killed and not
    /// as a failure with a signal exit code.
    func finalize(state newState: State, exitCode: Int32) {
        lock.lock(); defer { lock.unlock() }
        guard _state == .running else { return }
        _state = newState
        _exitCode = exitCode
    }

    func markKilled() {
        lock.lock(); defer { lock.unlock() }
        guard _state == .running else { return }
        _state = .killed
        _exitCode = nil
    }

    var summaryLine: String {
        let elapsed = Int(Date().timeIntervalSince(startedAt))
        switch state {
        case .running:
            return "[\(id)] still running (\(elapsed)s elapsed)"
        case .completed:
            return "[\(id)] completed (exit 0)"
        case .failed:
            return "[\(id)] failed (exit \(exitCode.map(String.init) ?? "unknown"))"
        case .killed:
            return "[\(id)] killed before completion"
        }
    }
}

/// Value-type snapshot of one running background shell, for the kill UI.
///
/// `BackgroundShellRecord` is a class shared with its reader threads; the
/// strip renders from this copy so SwiftUI never observes a live record and
/// never touches a process handle directly. Public because `AppModel`
/// publishes a `[BackgroundShellSummary]`.
public struct BackgroundShellSummary: Identifiable, Equatable {
    public let id: String
    public let commandHead: String
    public let description: String?
    public let startedAt: Date
    public let chatID: UUID?
    public let elapsedSeconds: Int

    init(
        id: String, commandHead: String, description: String?,
        startedAt: Date, chatID: UUID?, elapsedSeconds: Int
    ) {
        self.id = id
        self.commandHead = commandHead
        self.description = description
        self.startedAt = startedAt
        self.chatID = chatID
        self.elapsedSeconds = elapsedSeconds
    }
}

/// Process-lifetime registry of background shells.
///
/// Records are chat-scoped: an id is only resolvable from the conversation
/// that launched it, so a subagent or an unrelated chat can neither read
/// another turn's output nor kill another turn's process. Finished records
/// are kept for later retrieval and pruned per chat, since the registry is
/// unbounded otherwise and every record pins its output buffer.
final class BackgroundShellManager: @unchecked Sendable {
    static let shared = BackgroundShellManager()

    /// Bound on concurrently RUNNING shells. Output memory is capped per
    /// record, but processes are not: without a ceiling, twenty runaway
    /// `yes` loops are twenty cores this app handed out.
    static let maxRunningShells = 20
    /// Finished records kept per chat for later retrieval; the oldest fall
    /// off the end.
    static let maxFinishedRecordsPerChat = 50

    private let lock = NSLock()
    private var records: [String: BackgroundShellRecord] = [:]
    private var nextID = 1
    /// Set by `AppModel` at init, read under the lock, always dispatched to
    /// the main queue. The kill UI cannot poll the registry from SwiftUI (a
    /// view only re-renders when observed state changes), so every shape
    /// change of the record set pushes instead.
    private var _changeObserver: (() -> Void)?

    // MARK: - Launch

    /// Spawns `command` under zsh and registers it. Returns the record with
    /// its assigned id. Throws when the launch itself fails (bad binary,
    /// missing cwd); a command that later exits nonzero is a `.failed`
    /// record, not a throw -- the model reads the status through
    /// `BashOutput`.
    func launch(
        command: String,
        startDirectory: URL,
        environment: [String: String],
        chatID: UUID?,
        description: String?
    ) throws -> BackgroundShellRecord {
        lock.lock()
        let runningCount = records.values.filter { $0.state == .running }.count
        guard runningCount < Self.maxRunningShells else {
            lock.unlock()
            throw NSError(
                domain: "TurboSparkTool", code: 41,
                userInfo: [NSLocalizedDescriptionKey:
                    "Too many background commands are already running "
                        + "(\(runningCount)). Retrieve or kill one with BashOutput / "
                        + "KillShell before starting another."])
        }
        let id = "bg_\(nextID)"
        nextID += 1
        lock.unlock()

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-c", command]
        process.currentDirectoryURL = startDirectory
        process.environment = environment

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        let stdinPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        process.standardInput = stdinPipe

        let buffer = CappedOutputBuffer(capBytes: ProcessExecutor.defaultOutputCapBytes)
        let record = BackgroundShellRecord(
            id: id, command: command, description: description,
            chatID: chatID, process: process, outputBuffer: buffer)

        // Same one-reader-per-pipe pattern as ProcessExecutor (state#66): a
        // thread that drains to EOF, never a readabilityHandler with its
        // second-reader race. EOF arrives when the child (and any
        // grandchild holding the write end) exits or is killed.
        let readers = DispatchGroup()
        for handle in [stdoutPipe.fileHandleForReading, stderrPipe.fileHandleForReading] {
            readers.enter()
            DispatchQueue.global(qos: .utility).async {
                defer { readers.leave() }
                while true {
                    let chunk = handle.availableData
                    if chunk.isEmpty { break }
                    buffer.append(chunk)
                }
            }
        }

        // The completion handler closes the state only after the readers
        // have drained, so a `completed` status never coexists with unread
        // output. The bounded wait covers a grandchild that inherited the
        // pipe and outlives the shell itself, exactly as ProcessExecutor's
        // post-exit wait does.
        process.terminationHandler = { [weak record] terminatedProcess in
            guard let record else { return }
            _ = readers.wait(timeout: .now() + 2.0)
            let code = terminatedProcess.terminationStatus
            record.finalize(
                state: code == 0 ? .completed : .failed,
                exitCode: code)
            self.pruneFinished()
            self.notifyChanged()
        }

        lock.lock()
        records[id] = record
        lock.unlock()
        notifyChanged()

        // Background commands take no stdin: closing the write end delivers
        // EOF immediately, so a script that reads stdin fails fast instead
        // of blocking forever on a pipe nothing will ever feed.
        try? stdinPipe.fileHandleForWriting.close()

        do {
            try process.run()
            // Child process has inherited/duplicated write descriptors for stdout/stderr;
            // close parent's write handles immediately so EOF is cleanly delivered when child exits.
            try? stdoutPipe.fileHandleForWriting.close()
            try? stderrPipe.fileHandleForWriting.close()
        } catch {
            // The shell never started, so its id must not resolve: a
            // BashOutput against it would otherwise poll forever on a
            // `.running` record whose process does not exist.
            lock.lock()
            records.removeValue(forKey: id)
            lock.unlock()
            // Close pipe handles so reader threads see EOF and exit rather
            // than blocking forever on availableData.
            try? stdoutPipe.fileHandleForWriting.close()
            try? stderrPipe.fileHandleForWriting.close()
            try? stdoutPipe.fileHandleForReading.close()
            try? stderrPipe.fileHandleForReading.close()
            throw error
        }
        return record
    }

    // MARK: - Lookup and output

    /// Resolves an id within one chat's scope. Nil when the id is unknown,
    /// belongs to another chat, or has been pruned -- the caller cannot
    /// distinguish these and reports "unknown id" uniformly, which is also
    /// why the error text carries the ids that DO resolve.
    func record(id: String, chatID: UUID?) -> BackgroundShellRecord? {
        lock.lock(); defer { lock.unlock() }
        guard let record = records[id] else { return nil }
        guard record.chatID == chatID else { return nil }
        return record
    }

    func knownShellIDs(chatID: UUID?) -> [String] {
        lock.lock(); defer { lock.unlock() }
        return records.values
            .filter { $0.chatID == chatID }
            .sorted { $0.startedAt < $1.startedAt }
            .map { $0.id }
    }

    /// A one-shot status snapshot: the summary line plus the compacted
    /// output so far. Over the model cap the full buffer is spilled under
    /// the shell's own id, so repeated polls overwrite ONE file rather
    /// than accumulating one per poll.
    func outputSnapshot(_ record: BackgroundShellRecord) -> String {
        let text = ShellOutputFormatting.compactWithSpill(
            ShellOutputFormatting.stripANSI(record.outputBuffer.text),
            label: record.command, spillName: "shell-\(record.id)")
        if text.isEmpty {
            return record.summaryLine + "\n(No output yet)"
        }
        return record.summaryLine + "\n" + text
    }

    /// Polls until the shell is no longer running or `seconds` elapses, then
    /// returns the snapshot. Cancellation returns the current snapshot
    /// rather than throwing: the shell keeps running either way, and the
    /// model loses nothing by asking again.
    func waitAndOutput(_ record: BackgroundShellRecord, seconds: TimeInterval) async -> String {
        let deadline = Date().addingTimeInterval(seconds)
        while record.state == .running && Date() < deadline {
            do {
                try await Task.sleep(nanoseconds: 250_000_000)
            } catch {
                break
            }
        }
        return outputSnapshot(record)
    }

    // MARK: - Kill and pruning

    /// Terminates a running shell (SIGTERM, bounded wait, SIGKILL) and
    /// marks the record. Returns false when the shell was already finished.
    @discardableResult
    func kill(_ record: BackgroundShellRecord) -> Bool {
        guard record.state == .running else { return false }
        record.markKilled()
        ProcessExecutor.terminateAndReap(record.process)
        notifyChanged()
        return true
    }

    // MARK: - Kill UI and observation

    /// Registers the callback fired (on the main queue) whenever the record
    /// set changes shape: launch, kill, natural completion. Pass nil to
    /// detach; tests leave it unset.
    func setChangeObserver(_ callback: (() -> Void)?) {
        lock.lock()
        _changeObserver = callback
        lock.unlock()
    }

    private func notifyChanged() {
        lock.lock()
        let callback = _changeObserver
        lock.unlock()
        guard let callback else { return }
        DispatchQueue.main.async(execute: callback)
    }

    /// Every record whose process is still running, oldest first. The kill
    /// UI renders from these; kill scope is enforced by the CALLER
    /// (`AppModel` filters by the same visibility rule the strip shows).
    func runningRecords() -> [BackgroundShellRecord] {
        lock.lock(); defer { lock.unlock() }
        return records.values
            .filter { $0.state == .running }
            .sorted { $0.startedAt < $1.startedAt }
    }

    /// Running records for one chat, oldest first: what a chat's deletion
    /// is allowed to end.
    func runningRecords(chatID: UUID) -> [BackgroundShellRecord] {
        lock.lock(); defer { lock.unlock() }
        return records.values
            .filter { $0.state == .running && $0.chatID == chatID }
            .sorted { $0.startedAt < $1.startedAt }
    }

    /// Kills every running shell, ladders in parallel. Each
    /// `terminateAndReap` can block ~3 s on a child that ignores SIGTERM;
    /// the shells are independent processes, so the ladders run
    /// concurrently rather than summing on the caller's thread.
    @discardableResult
    func killAll() -> Int {
        let running = runningRecords()
        for record in running {
            record.markKilled()
            DispatchQueue.global(qos: .userInitiated).async {
                ProcessExecutor.terminateAndReap(record.process)
            }
        }
        if !running.isEmpty { notifyChanged() }
        return running.count
    }

    /// Decisive variant for app shutdown: SIGKILL the whole tree with no
    /// grace period and no waiting. The process is exiting, so SIGTERM's
    /// output-flush grace buys nothing and the ladder's bounded waits would
    /// stall quit by seconds per stubborn child.
    func killAllForShutdown() {
        let running = runningRecords()
        for record in running {
            record.markKilled()
            ProcessExecutor.killTreeNow(record.process.processIdentifier)
        }
    }

    /// Drops the oldest finished records past the per-chat keep count.
    /// Running records are never pruned: they are alive, and their ids must
    /// keep resolving.
    private func pruneFinished() {
        lock.lock(); defer { lock.unlock() }
        let finished = records.values.filter { $0.state != .running }
        let byChat = Dictionary(grouping: finished, by: { $0.chatID })
        var doomed: Set<String> = []
        for (_, group) in byChat {
            guard group.count > Self.maxFinishedRecordsPerChat else { continue }
            let oldest = group.sorted { $0.startedAt < $1.startedAt }
                .prefix(group.count - Self.maxFinishedRecordsPerChat)
            doomed.formUnion(oldest.map { $0.id })
        }
        for id in doomed { records.removeValue(forKey: id) }
    }

    /// Test seam: clears every record. Running shells are killed first so a
    /// test cannot leak a process into the next one.
    func resetForTests() {
        lock.lock()
        let all = Array(records.values)
        records.removeAll()
        lock.unlock()
        for record in all where record.state == .running {
            record.markKilled()
            ProcessExecutor.terminateAndReap(record.process)
        }
    }
}
