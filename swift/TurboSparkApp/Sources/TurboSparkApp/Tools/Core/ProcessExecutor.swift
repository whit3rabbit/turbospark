import Foundation

/// Thread-safe growable byte buffer with a hard cap.
///
/// `Pipe.fileHandleForReading.readabilityHandler` fires on an arbitrary GCD
/// thread chosen by Foundation, never on the caller's thread, so this cannot
/// be a plain `var` behind the executor's async function.
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
/// concurrently with the wait (via `readabilityHandler`) is what breaks that
/// cycle, and a deadline that is re-checked on a timer -- never inside a
/// blocking read -- is what makes a timeout actually fire.
enum ProcessExecutor {
    struct Output {
        var stdout: String
        var stderr: String
        var exitCode: Int32
        var timedOut: Bool
    }

    static let defaultOutputCapBytes = 1_000_000
    static let defaultTimeoutSeconds: TimeInterval = 120

    /// Runs `executableURL` to completion (or until `timeoutSeconds` elapses),
    /// draining stdout/stderr concurrently so neither the process nor the
    /// caller can block on a full pipe buffer.
    static func run(
        executableURL: URL,
        arguments: [String],
        currentDirectoryURL: URL? = nil,
        environment: [String: String]? = nil,
        stdin inputData: Data? = nil,
        timeoutSeconds: TimeInterval,
        outputCapBytes: Int = defaultOutputCapBytes
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
        let stderrBuffer = CappedOutputBuffer(capBytes: outputCapBytes)

        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty {
                handle.readabilityHandler = nil
            } else {
                stdoutBuffer.append(chunk)
            }
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty {
                handle.readabilityHandler = nil
            } else {
                stderrBuffer.append(chunk)
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

        let deadline = Date().addingTimeInterval(timeoutSeconds)
        var timedOut = false
        while process.isRunning {
            if Date() >= deadline {
                timedOut = true
                process.terminate()
                let killDeadline = Date().addingTimeInterval(2.0)
                while process.isRunning && Date() < killDeadline {
                    try? await Task.sleep(nanoseconds: 50_000_000)
                }
                break
            }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }

        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        stderrPipe.fileHandleForReading.readabilityHandler = nil
        // Drain anything left sitting in the pipe after the last handler firing.
        let trailingOut = stdoutPipe.fileHandleForReading.availableData
        if !trailingOut.isEmpty { stdoutBuffer.append(trailingOut) }
        let trailingErr = stderrPipe.fileHandleForReading.availableData
        if !trailingErr.isEmpty { stderrBuffer.append(trailingErr) }

        let exitCode = process.isRunning ? -1 : process.terminationStatus
        return Output(
            stdout: stdoutBuffer.text,
            stderr: stderrBuffer.text,
            exitCode: exitCode,
            timedOut: timedOut
        )
    }
}
