import Darwin
import Foundation

/// Sees what the Rust engine writes to standard error.
///
/// The engine reports a number of decisions only by `eprintln!` (a vision
/// auto-resolution, a speculation fallback, an oversize-image clamp), from
/// crates that are shared with the command-line tools and so cannot grow a
/// host-specific logging callback. They all land on file descriptor 2 of this
/// process, which is where this reads them.
///
/// **THIS TEES, IT DOES NOT STEAL.** Each chunk is still written to the
/// original stderr, so a terminal, Xcode console or `log` redirection keeps
/// working; the handler gets a copy, one call per line.
///
/// **PROCESS-WIDE, ONE AT A TIME.** It replaces fd 2 for as long as it is
/// running, which affects every writer in the process, not just the engine.
/// A second `start` while one is running throws rather than nesting, and
/// `stop()` puts the original descriptor back. The handler runs on a private
/// serial queue and must not block it. Calling `stop()` from the handler is
/// safe: it restores stderr immediately and drains buffered output after the
/// current callback returns. Lines carry no level or source: the engine does
/// not emit one.
public final class StderrCapture: @unchecked Sendable {
    public static let shared = StderrCapture()

    private final class Capture {
        let original: Int32
        let readEnd: Int32
        let handler: @Sendable (String) -> Void
        var pending = Data()
        var stopping = false
        var finished = false

        init(original: Int32, readEnd: Int32, handler: @escaping @Sendable (String) -> Void) {
            self.original = original
            self.readEnd = readEnd
            self.handler = handler
        }
    }

    private let lock = NSLock()
    private var capture: Capture?
    private var source: DispatchSourceRead?
    private let queue: DispatchQueue
    private let queueKey = DispatchSpecificKey<UInt8>()

    // Internal construction lets tests isolate the process-wide fd lifecycle.
    init() {
        queue = DispatchQueue(label: "com.turbospark.stderr-capture")
        queue.setSpecific(key: queueKey, value: 1)
    }

    /// Whether a capture is currently running.
    public var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return capture.map { !$0.stopping } ?? false
    }

    /// Starts copying stderr to `handler`, one call per complete line.
    public func start(_ handler: @escaping @Sendable (String) -> Void) throws {
        lock.lock(); defer { lock.unlock() }
        guard capture == nil else {
            throw TurboSparkError(code: .invalidArgument, message: "stderr capture is already running")
        }
        var fds: [Int32] = [0, 0]
        guard pipe(&fds) == 0 else {
            throw TurboSparkError(code: .unknown, message: "pipe() failed: errno \(errno)")
        }
        let flags = fcntl(fds[0], F_GETFL)
        guard flags >= 0, fcntl(fds[0], F_SETFL, flags | O_NONBLOCK) >= 0 else {
            let code = errno
            close(fds[0]); close(fds[1])
            throw TurboSparkError(code: .unknown, message: "could not configure stderr pipe: errno \(code)")
        }
        let saved = dup(STDERR_FILENO)
        guard saved >= 0, dup2(fds[1], STDERR_FILENO) >= 0 else {
            let code = errno
            close(fds[0]); close(fds[1]); if saved >= 0 { close(saved) }
            throw TurboSparkError(code: .unknown, message: "could not redirect stderr: errno \(code)")
        }
        // fd 2 now owns the write end; the extra descriptor is not needed.
        close(fds[1])
        let active = Capture(original: saved, readEnd: fds[0], handler: handler)

        let reader = DispatchSource.makeReadSource(fileDescriptor: fds[0], queue: queue)
        reader.setEventHandler { [weak self, weak active] in
            guard let self, let active else { return }
            self.readAvailable(from: active)
        }
        capture = active
        source = reader
        reader.resume()
    }

    /// Restores stderr, drains buffered output, and delivers any unterminated final line.
    /// When called by the handler, draining finishes after that callback returns.
    public func stop() {
        let onCaptureQueue = DispatchQueue.getSpecific(key: queueKey) != nil
        lock.lock()
        guard let active = capture else { lock.unlock(); return }
        var initiatedStop = false
        if !active.stopping {
            // Keep the pipe alive if restoring fd 2 fails so a later stop can retry.
            guard dup2(active.original, STDERR_FILENO) >= 0 else { lock.unlock(); return }
            active.stopping = true
            initiatedStop = true
        }
        lock.unlock()

        if onCaptureQueue {
            // A synchronous join here would wait on the callback currently calling stop().
            if initiatedStop { queue.async { self.finishStop(active) } }
            return
        }
        queue.sync { finishStop(active) }
    }

    private func readAvailable(from active: Capture) {
        lock.lock()
        let isCurrent = capture === active && !active.finished
        lock.unlock()
        guard isCurrent else { return }

        var buffer = [UInt8](repeating: 0, count: 8192)
        while true {
            let count = Darwin.read(active.readEnd, &buffer, buffer.count)
            if count > 0 {
                let chunk = Data(buffer[0..<count])
                tee(chunk, to: active.original)
                active.pending.append(chunk)
                while let newline = active.pending.firstIndex(of: 0x0A) {
                    let line = active.pending[active.pending.startIndex..<newline]
                    active.pending.removeSubrange(active.pending.startIndex...newline)
                    active.handler(String(decoding: line, as: UTF8.self))
                }
                continue
            }
            if count < 0 && errno == EINTR { continue }
            return // EOF, EAGAIN, or an unrecoverable read error.
        }
    }

    private func tee(_ data: Data, to descriptor: Int32) {
        data.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return }
            var offset = 0
            while offset < bytes.count {
                let count = Darwin.write(descriptor, base.advanced(by: offset), bytes.count - offset)
                if count > 0 {
                    offset += count
                } else if count < 0 && errno == EINTR {
                    continue
                } else {
                    return
                }
            }
        }
    }

    private func finishStop(_ active: Capture) {
        lock.lock()
        let shouldFinish = capture === active && active.stopping && !active.finished
        lock.unlock()
        guard shouldFinish else { return }

        // fd 2 has been restored. The nonblocking read drains bytes already in the pipe.
        readAvailable(from: active)
        if !active.pending.isEmpty {
            let finalLine = String(decoding: active.pending, as: UTF8.self)
            active.pending.removeAll()
            active.handler(finalLine)
        }

        lock.lock()
        guard capture === active && !active.finished else { lock.unlock(); return }
        active.finished = true
        let reader = source
        source = nil
        lock.unlock()

        reader?.cancel()
        close(active.readEnd)
        close(active.original)

        lock.lock()
        if capture === active { capture = nil }
        lock.unlock()
    }
}
