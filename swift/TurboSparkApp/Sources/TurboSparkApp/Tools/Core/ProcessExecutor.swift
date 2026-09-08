import Foundation

/// Thread-safe growable byte buffer with a hard cap.
///
/// The pipe readers run on GCD threads of their own, never on the caller's,
/// so this cannot be a plain `var` behind the executor's async function.
final class CappedOutputBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()
    private let capBytes: Int
    private var truncated = false

    init(capBytes: Int) { self.capBytes = capBytes }

    func append(_ chunk: Data) {
        lock.lock(); defer { lock.unlock() }
        guard data.count < capBytes else { truncated = true; return }
        let room = capBytes - data.count
        if chunk.count > room {
            data.append(chunk.prefix(room))
            truncated = true
        } else {
            data.append(chunk)
        }
    }

    var text: String {
        lock.lock(); defer { lock.unlock() }
        var s = String(data: data, encoding: .utf8) ?? String(decoding: data, as: UTF8.self)
        if truncated {
            s += "\n... (output truncated at \(capBytes) bytes)"
        }
        return s
    }
}

/// Shared subprocess execution helper.
///
/// Every one-shot process this app spawns from a model-proposed tool call or
/// a lifecycle hook (`run_command`, hook commands) goes through here so the
/// deadlock class of bug is fixed once rather than three times independently:
/// `Process.waitUntilExit()` followed by `readDataToEndOfFile()` blocks
/// forever the moment a child writes more than the ~64KB pipe buffer before
/// exiting, because nothing is draining the pipe while we wait, and the
/// child then blocks on ITS OWN write() call. Reading stdout/stderr
/// concurrently with the wait, on one dedicated thread per pipe, is what
/// breaks that cycle, and a deadline re-checked on a timer -- never inside a
/// blocking read -- is what makes a timeout actually fire.
enum ProcessExecutor {
    struct Output {
        var stdout: String
        var stderr: String
        var exitCode: Int32
        var timedOut: Bool
        /// Set only when the run asked for merged streams: stdout and stderr
        /// interleaved in arrival order, the shared buffer's text. `stdout`
        /// and `stderr` are both empty in that mode.
        var mergedOutput: String?

        var combinedText: String {
            if let mergedOutput { return mergedOutput }
            return [stdout, stderr].filter { !$0.isEmpty }.joined(separator: "\n")
        }
    }

    static let defaultOutputCapBytes = 1_000_000
    static let defaultTimeoutSeconds: TimeInterval = 120

    /// Runs `executableURL` to completion (or until `timeoutSeconds` elapses),
    /// draining stdout/stderr concurrently so neither the process nor the
    /// caller can block on a full pipe buffer.
    ///
    /// `mergeStreams` appends BOTH pipes into one shared capped buffer, so
    /// the text interleaves in arrival order the way a terminal would show
    /// it, instead of a stdout block followed by a stderr block whose
    /// relative order is unrecoverable. Arrival order is the child's write
    /// order as the kernel delivered it, which is the honest ordering there
    /// is. The shell tool runs merged; hook runners keep the two streams
    /// apart because their verdict JSON parses per stream.
    static func run(
        executableURL: URL,
        arguments: [String],
        currentDirectoryURL: URL? = nil,
        environment: [String: String]? = nil,
        stdin inputData: Data? = nil,
        timeoutSeconds: TimeInterval,
        outputCapBytes: Int = defaultOutputCapBytes,
        mergeStreams: Bool = false
    ) async throws -> Output {
        let process = Process()
        process.executableURL = executableURL
        process.arguments = arguments
        if let currentDirectoryURL { process.currentDirectoryURL = currentDirectoryURL }
        if let environment { process.environment = environment }

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        let stdinPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        process.standardInput = stdinPipe

        let stdoutBuffer = CappedOutputBuffer(capBytes: outputCapBytes)
        let stderrBuffer = mergeStreams ? stdoutBuffer : CappedOutputBuffer(capBytes: outputCapBytes)

        // **ONE READER PER PIPE, AND THE CALLER WAITS FOR IT** (state#66).
        //
        // This used a `readabilityHandler` plus a trailing `availableData`
        // read after the wait. Setting `readabilityHandler = nil` does not
        // wait for a block already dispatched on the handle's private source
        // queue, so the trailing read could run CONCURRENTLY with the
        // handler's own -- two readers on one descriptor. The lock-guarded
        // buffer keeps that from corrupting memory and does nothing for the
        // ORDER: a chunk read by the handler after the trailing read has
        // appended lands out of sequence, and a hook's JSON verdict arriving
        // in two halves parses as neither.
        //
        // A dedicated thread per pipe, looping on `availableData`, has no
        // second reader to race, and cannot deadlock: it drains continuously,
        // so the child never blocks on a full pipe, and it stops at EOF --
        // which arrives when the child exits, including when
        // `terminateAndReap` is what makes it exit. The `DispatchGroup` is
        // what lets the caller know both are done before it reads the text.
        let readers = DispatchGroup()
        for (handle, buffer) in [
            (stdoutPipe.fileHandleForReading, stdoutBuffer),
            (stderrPipe.fileHandleForReading, stderrBuffer),
        ] {
            readers.enter()
            DispatchQueue.global(qos: .utility).async {
                defer { readers.leave() }
                // Chunked, NOT `readDataToEndOfFile()`: that accumulates the
                // whole stream in memory before returning, which would defeat
                // `outputCapBytes` on a runaway command -- the cap can only
                // bound what it is shown a piece at a time. `availableData`
                // blocks until there is data or EOF, and returns empty at EOF.
                while true {
                    let chunk = handle.availableData
                    if chunk.isEmpty { break }
                    buffer.append(chunk)
                }
            }
        }

        try process.run()

        // Feed stdin off the calling task: a full pipe buffer would otherwise
        // block this write until the child drains it, which is exactly the
        // deadlock this helper exists to avoid (the hook runner's payload
        // write did this synchronously before the timeout loop even began).
        if let inputData, !inputData.isEmpty {
            let writeHandle = stdinPipe.fileHandleForWriting
            DispatchQueue.global(qos: .utility).async {
                try? writeHandle.write(contentsOf: inputData)
                try? writeHandle.close()
            }
        } else {
            try? stdinPipe.fileHandleForWriting.close()
        }

        // **THE TICK MUST NOT COLLAPSE UNDER CANCELLATION, AND CANCELLATION
        // MUST KILL THE CHILD.**
        //
        // `try? await Task.sleep` returns IMMEDIATELY once the surrounding
        // task is cancelled, and swallowing that with `try?` leaves the loop
        // condition unchanged -- so a cancelled `run_command` spun this loop
        // at full CPU for the whole remaining deadline (120 s by default)
        // while the child kept running to completion behind it. Both halves
        // were silent: no error, no log, just a busy core and an orphan
        // process. `Task.isCancelled` is checked explicitly and the sleep is
        // one that does not throw.
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        var timedOut = false
        var cancelled = false
        while process.isRunning {
            if Task.isCancelled {
                cancelled = true
                terminateAndReap(process)
                break
            }
            if Date() >= deadline {
                timedOut = true
                terminateAndReap(process)
                break
            }
            await uninterruptibleSleep(nanoseconds: 20_000_000)
        }

        // The readers finish at EOF, which the child's exit delivers. The
        // bound is for the case where it does NOT: a grandchild inheriting
        // the pipe holds the write end open after its parent is reaped, and
        // waiting forever there would park this task for the life of the
        // process. Two seconds is long past when a dead child's buffered
        // output has arrived.
        await waitForReaders(readers, timeoutSeconds: 2.0)

        let exitCode = process.isRunning ? -1 : process.terminationStatus
        if cancelled { throw CancellationError() }
        if mergeStreams {
            return Output(
                stdout: "", stderr: "", exitCode: exitCode, timedOut: timedOut,
                mergedOutput: stdoutBuffer.text)
        }
        return Output(
            stdout: stdoutBuffer.text,
            stderr: stderrBuffer.text,
            exitCode: exitCode,
            timedOut: timedOut,
            mergedOutput: nil
        )
    }

    /// SIGTERM, then a bounded wait, then SIGKILL -- and then the same for
    /// everything the child left running.
    ///
    /// Synchronous on purpose: it runs from the cancellation path, where an
    /// `await` would suspend on an already-cancelled task and hand back
    /// control before the child is dead.
    ///
    /// `internal` rather than `private` since state#61: `McpClientEngine`
    /// spawns stdio children of its own and had only a bare `terminate()`,
    /// so a server that traps or ignores SIGTERM survived every call and
    /// accumulated one orphan per invocation for the life of the app.
    ///
    /// **THE KILL IS A TREE KILL, NOT A PID KILL.** A shell tool command is
    /// `zsh -c <script>`, and zsh does not forward signals: a script that
    /// started a dev server, daemon, or `yes` loop survives the death of
    /// its shell, keeps its pipe write end open, and keeps burning a core
    /// -- which is exactly the thing the user asked to stop. Before the
    /// ladder, the descendant set is snapshotted from the kernel process
    /// table; after the child is gone, every snapshot member still alive is
    /// SIGKILLed. Best-effort by nature (a pid that churned inside the
    /// window could be signalled, and a grandchild spawned DURING the ladder
    /// is invisible to the snapshot), but it closes the common case.
    static func terminateAndReap(_ process: Process) {
        // The snapshot MUST precede the kill: once the direct child dies its
        // children reparent to launchd and the ppid chain that identifies
        // them is gone.
        let descendants = descendantPIDs(of: process.processIdentifier)
        process.terminate()
        let killDeadline = Date().addingTimeInterval(2.0)
        while process.isRunning && Date() < killDeadline {
            usleep(50_000)
        }
        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
            let hardDeadline = Date().addingTimeInterval(1.0)
            while process.isRunning && Date() < hardDeadline {
                usleep(50_000)
            }
        }
        for pid in descendants {
            kill(pid, SIGKILL)
        }
    }

    /// Every living process as `(pid, ppid)`, from one `sysctl` walk of
    /// `kern.proc.all`.
    private static func processTableSnapshot() -> [(pid: pid_t, ppid: pid_t)] {
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_ALL]
        var size = 0
        guard sysctl(&mib, u_int(mib.count), nil, &size, nil, 0) == 0, size > 0 else {
            return []
        }
        // Over-allocate ~12%: processes can appear between the sizing call
        // and the read, and sysctl answers a short buffer with ENOMEM rather
        // than truncating. One generous buffer covers that race.
        let stride = MemoryLayout<kinfo_proc>.stride
        let capacity = max(1, (size + size / 8) / stride)
        let buffer = UnsafeMutablePointer<kinfo_proc>.allocate(capacity: capacity)
        defer { buffer.deallocate() }
        var actual = capacity * stride
        guard sysctl(&mib, u_int(mib.count), buffer, &actual, nil, 0) == 0 else {
            return []
        }
        let count = actual / stride
        return (0..<count).map { i in
            (pid: buffer[i].kp_proc.p_pid, ppid: buffer[i].kp_eproc.e_ppid)
        }
    }

    /// All transitively-descendant pids of `root` at the moment of the call.
    /// Empty on any probe failure: the sweep is an addition to the direct
    /// kill, never a condition of it.
    static func descendantPIDs(of root: pid_t) -> [pid_t] {
        guard root > 0 else { return [] }
        var childrenOf: [pid_t: [pid_t]] = [:]
        for (pid, ppid) in processTableSnapshot() where ppid > 0 {
            childrenOf[ppid, default: []].append(pid)
        }
        var descendants: [pid_t] = []
        var seen: Set<pid_t> = []
        var queue = childrenOf[root] ?? []
        seen.formUnion(queue)
        while let pid = queue.popLast() {
            descendants.append(pid)
            for child in childrenOf[pid] ?? [] where !seen.contains(child) {
                seen.insert(child)
                queue.append(child)
            }
        }
        return descendants
    }

    /// The shutdown variant: SIGKILL the tree immediately, no grace period,
    /// no waiting. Only for process exit -- everywhere else the SIGTERM
    /// ladder in `terminateAndReap` is what gives a child the chance to
    /// flush its output.
    static func killTreeNow(_ pid: pid_t) {
        guard pid > 0 else { return }
        for descendant in descendantPIDs(of: pid) {
            kill(descendant, SIGKILL)
        }
        kill(pid, SIGKILL)
    }

    /// Asynchronously waits for `group` to complete, or until `timeoutSeconds`
    /// elapses, without blocking the cooperative thread pool.
    private static func waitForReaders(_ group: DispatchGroup, timeoutSeconds: Double) async {
        await withCheckedContinuation { continuation in
            let lock = NSLock()
            var resumed = false
            var timeoutWorkItem: DispatchWorkItem?

            let resumeOnce = {
                lock.lock()
                defer { lock.unlock() }
                if !resumed {
                    resumed = true
                    timeoutWorkItem?.cancel()
                    continuation.resume()
                }
            }

            let workItem = DispatchWorkItem {
                resumeOnce()
            }
            timeoutWorkItem = workItem

            group.notify(queue: DispatchQueue.global(qos: .utility)) {
                resumeOnce()
            }
            DispatchQueue.global(qos: .utility).asyncAfter(
                deadline: .now() + timeoutSeconds,
                execute: workItem
            )
        }
    }

    /// A sleep that actually sleeps on a cancelled task.
    ///
    /// `Task.sleep` throws `CancellationError` the moment the task is
    /// cancelled, so it is not usable as the tick of a loop that has cleanup
    /// left to do: the loop stops pacing and spins. The cancellation check is
    /// the loop's own job, above.
    private static func uninterruptibleSleep(nanoseconds: UInt64) async {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).asyncAfter(
                deadline: .now() + .nanoseconds(Int(nanoseconds))
            ) {
                continuation.resume()
            }
        }
    }
}
