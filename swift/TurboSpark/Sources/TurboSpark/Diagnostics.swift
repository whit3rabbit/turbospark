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
/// serial queue and must not block it. Lines carry no level or source: the
/// engine does not emit one.
public final class StderrCapture: @unchecked Sendable {
    public static let shared = StderrCapture()

    private let lock = NSLock()
    private var original: Int32 = -1
    private var readEnd: Int32 = -1
    private var source: DispatchSourceRead?
    private let queue = DispatchQueue(label: "com.turbospark.stderr-capture")
    private var pending = Data()

    private init() {}

    /// Whether a capture is currently running.
    public var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return source != nil
    }

    /// Starts copying stderr to `handler`, one call per complete line.
    public func start(_ handler: @escaping @Sendable (String) -> Void) throws {
        lock.lock(); defer { lock.unlock() }
        guard source == nil else {
            throw TurboSparkError(code: .invalidArgument, message: "stderr capture is already running")
        }
        var fds: [Int32] = [0, 0]
        guard pipe(&fds) == 0 else {
            throw TurboSparkError(code: .unknown, message: "pipe() failed: errno \(errno)")
        }
        let saved = dup(STDERR_FILENO)
        guard saved >= 0, dup2(fds[1], STDERR_FILENO) >= 0 else {
            close(fds[0]); close(fds[1]); if saved >= 0 { close(saved) }
            throw TurboSparkError(code: .unknown, message: "could not redirect stderr: errno \(errno)")
        }
        // fd 2 now owns the write end; the extra descriptor is not needed.
        close(fds[1])
        original = saved
        readEnd = fds[0]
        pending = Data()

        let reader = DispatchSource.makeReadSource(fileDescriptor: fds[0], queue: queue)
        reader.setEventHandler { [weak self] in
            guard let self else { return }
            var buffer = [UInt8](repeating: 0, count: 8192)
            let n = read(self.readEnd, &buffer, buffer.count)
            guard n > 0 else { return }
            let chunk = Data(buffer[0..<n])
            _ = chunk.withUnsafeBytes { write(self.original, $0.baseAddress, n) }
            self.pending.append(chunk)
            while let newline = self.pending.firstIndex(of: 0x0A) {
                let line = self.pending[self.pending.startIndex..<newline]
                self.pending.removeSubrange(self.pending.startIndex...newline)
                handler(String(decoding: line, as: UTF8.self))
            }
        }
        source = reader
        reader.resume()
    }

    /// Restores the original stderr and delivers any unterminated final line.
    public func stop() {
        lock.lock()
        guard let reader = source else { lock.unlock(); return }
        source = nil
        let restore = original, read = readEnd
        original = -1; readEnd = -1
        lock.unlock()

        // Put fd 2 back FIRST so no writer is still pointed at the pipe, then
        // let the reader drain what is already buffered before closing it.
        dup2(restore, STDERR_FILENO)
        queue.sync {}
        reader.cancel()
        close(read)
        close(restore)
    }
}
