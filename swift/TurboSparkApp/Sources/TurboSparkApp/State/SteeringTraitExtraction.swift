import Foundation
import TurboSpark

/// One built-in trait pack, as `turbospark-steer packs --json` reports it.
///
/// VALUES ONLY. The CLI's `title` and `summary` are English source text and
/// never reach the UI: every display string for a known trait id is
/// localized at the sheet's call sites, so a new pack added on the Rust side
/// without app copy falls back to its raw id rather than shipping English
/// under every language.
struct SteeringTraitPackReport: Decodable, Equatable, Identifiable {
    let id: String
    let title: String
    let summary: String
    let positivePrompts: Int
    let negativePrompts: Int
    let evalPrompts: Int
    let recommendedMode: String
    let recommendedScale: Double

    /// The pack's mode as the app stores it on a preset, defaulting to
    /// `ablate` (the engine's own default) when the CLI reports a spelling
    /// this build does not know.
    var presetMode: AppSteeringModeOption {
        AppSteeringModeOption(rawValue: recommendedMode) ?? .ablate
    }
}

/// Parses the extractor's stderr progress lines. Kept pure and separately
/// testable: the fraction it computes is what the sheet's bar shows, and a
/// miscounted bar reads as a hang.
enum SteeringTraitProgress {
    enum Side: String {
        case positive
        case negative
    }

    /// One "[positive] capturing prompt 9/24" line. The CLI emits one every
    /// 8th prompt per side; 1/24 on both sides means the walk started.
    struct Snapshot: Equatable {
        let side: Side
        let done: Int
        let total: Int
    }

    /// The overall fraction across both sides, given each side's LATEST
    /// snapshot. An unknown side is assumed to carry the SAME total as the
    /// known one: the walk runs the sides sequentially, and counting only
    /// the started side would render a bar that reads 50% just before the
    /// second side begins and snaps back to ~25% when it does. The in-app
    /// path always runs a pack, whose sides are equal by construction; the
    /// CLI's custom-corpus path is not reachable from this UI.
    static func overallFraction(positive: Snapshot?, negative: Snapshot?) -> Double? {
        guard positive != nil || negative != nil else { return nil }
        let done = [positive, negative].compactMap { $0 }
            .reduce(0) { $0 + min($1.done, $1.total) }
        let totals = [positive?.total, negative?.total].compactMap { $0 }
        let total: Int
        if totals.count == 1 {
            total = totals[0] * 2
        } else {
            total = totals.reduce(0, +)
        }
        guard total > 0 else { return nil }
        return Double(done) / Double(total)
    }

    static func parse(_ line: String) -> Snapshot? {
        // One leading marker word in brackets, then "capturing prompt N/M".
        // Deliberately plain string work: the CLI is this repo's own binary
        // and the format is owned here on both sides.
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("["), let bracketEnd = trimmed.firstIndex(of: "]") else {
            return nil
        }
        let sideRaw = String(trimmed[trimmed.index(after: trimmed.startIndex)..<bracketEnd])
        guard let side = Side(rawValue: sideRaw) else { return nil }
        let rest = trimmed[trimmed.index(after: bracketEnd)...].trimmingCharacters(in: .whitespaces)
        guard rest.hasPrefix("capturing prompt ") else { return nil }
        let counts = rest.dropFirst("capturing prompt ".count)
        let parts = counts.split(separator: "/")
        guard parts.count == 2,
            let done = Int(parts[0]),
            let total = Int(parts[1]),
            total > 0
        else { return nil }
        return Snapshot(side: side, done: done, total: total)
    }
}

/// Locates the `turbospark-steer` CLI the extraction runs, mirroring
/// `AgentTerminalLaunch.findCLI`: beside the app executable first, then the
/// user's PATH, then the two make-install prefixes and the two Homebrew
/// prefixes a GUI app's minimal launchd PATH omits. Injectable inputs keep
/// the ordering testable without touching the filesystem.
enum SteeringTraitTool {
    static func candidateURLs(
        bundleDirectory: String?, path: String?, home: String
    ) -> [URL] {
        var candidates: [URL] = []
        if let bundleDirectory {
            candidates.append(
                URL(fileURLWithPath: bundleDirectory).appendingPathComponent("turbospark-steer"))
        }
        if let path {
            let directories = path.split(separator: ":").map(String.init)
            candidates += directories.map {
                URL(fileURLWithPath: $0).appendingPathComponent("turbospark-steer")
            }
        }
        candidates.append(
            URL(fileURLWithPath: home).appendingPathComponent(".cargo/bin/turbospark-steer"))
        candidates.append(
            URL(fileURLWithPath: home).appendingPathComponent(".local/bin/turbospark-steer"))
        candidates.append(URL(fileURLWithPath: "/opt/homebrew/bin/turbospark-steer"))
        candidates.append(URL(fileURLWithPath: "/usr/local/bin/turbospark-steer"))
        return candidates
    }

    static func findSteerCLI() -> URL? {
        candidateURLs(
            bundleDirectory: Bundle.main.executableURL?.deletingLastPathComponent().path,
            path: ProcessInfo.processInfo.environment["PATH"],
            home: FileManager.default.homeDirectoryForCurrentUser.path
        )
        .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }
}

/// Runs the `turbospark-steer` binary. A CHILD PROCESS, deliberately, rather
/// than an FFI re-implementation: extraction opens its own Metal session per
/// side and reuses the exact capture, reduction, and write path the CLI
/// verifies against real installs. Re-implementing it behind the ABI would
/// be a second copy of a measured pipeline, and a second Metal session
/// inside the app process while a chat session is open.
@MainActor
final class SteeringTraitExtractionController: ObservableObject {
    enum Phase: Equatable {
        case idle
        case capturing(fraction: Double?)
        case finishing
        case failed(message: String)
    }

    @Published private(set) var phase: Phase = .idle
    /// Set while a run is live so the sheet's dismiss path can stop it;
    /// otherwise a backed-out extraction leaves a process running.
    private var runningProcess: Process?
    /// The LATEST snapshot per side, so the overall fraction climbs toward
    /// 1 as both walks advance instead of resetting each line.
    private var latest: [SteeringTraitProgress.Side: SteeringTraitProgress.Snapshot] = [:]

    /// Lists the built-in packs, or the error. The CLI is fast (no model
    /// involved), so a synchronous read over a short pipe off the main
    /// actor is fine; a model open is the slow part and that is only in
    /// `extract`.
    nonisolated static func listPacks(cliURL: URL) -> Result<[SteeringTraitPackReport], Error> {
        let process = Process()
        process.executableURL = cliURL
        process.arguments = ["packs", "--json"]
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        do {
            try process.run()
            let data = stdout.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else {
                let message = String(
                    decoding: stderr.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
                return .failure(SteeringTraitError.toolFailed(
                    status: process.terminationStatus, message: message))
            }
            return .success(try JSONDecoder().decode([SteeringTraitPackReport].self, from: data))
        } catch {
            return .failure(error)
        }
    }

    /// Extracts one trait direction for `modelPath` into `outputURL`,
    /// publishing progress into `phase`. Throws `CancellationError` when the
    /// task is cancelled; the process is terminated with it.
    func extract(
        cliURL: URL,
        modelPath: String,
        traitID: String,
        outputURL: URL
    ) async throws {
        // A task cancelled before it got here (the sheet dismissed while the
        // Task was still queued) must not launch anything.
        try Task.checkCancellation()
        let process = Process()
        process.executableURL = cliURL
        process.arguments = [
            "extract",
            "--model", modelPath,
            "--trait", traitID,
            "--out", outputURL.path,
        ]
        let stderr = Pipe()
        // Progress lines and early refusals arrive on stderr; the final
        // diagnostics table goes to stdout and is not needed here.
        process.standardError = stderr
        // Discarded rather than an unread Pipe, which would stall the child
        // once its diagnostics table filled the pipe buffer.
        process.standardOutput = FileHandle.nullDevice
        // Installed BEFORE run: a tool that refuses in milliseconds can exit
        // before a handler set afterwards, and the wait would never resume.
        let exit = ProcessExit(process)
        runningProcess = process
        latest = [:]
        phase = .capturing(fraction: nil)

        // The readability handler fires on an arbitrary queue, so the line
        // split happens behind a lock and only the MainActor hop touches
        // published state.
        let buffer = ProgressLineBuffer { [weak self] snapshot in
            Task { @MainActor [weak self] in
                self?.record(snapshot)
            }
        }
        stderr.fileHandleForReading.readabilityHandler = { handle in
            buffer.consume(handle.availableData)
        }

        defer {
            stderr.fileHandleForReading.readabilityHandler = nil
            runningProcess = nil
        }

        try await withTaskCancellationHandler {
            try process.run()
            // Cancelled between the check above and the launch: onCancel saw
            // a process that was not running yet, so stop it here.
            if Task.isCancelled { process.terminate() }
            let status = await exit.status()
            guard !Task.isCancelled else { throw CancellationError() }
            guard status == 0 else {
                let message = buffer.tail(lines: 6)
                throw SteeringTraitError.toolFailed(status: status, message: message)
            }
        } onCancel: {
            // terminate() on a process that never launched raises an
            // Objective-C exception, and onCancel runs immediately for a task
            // that is already cancelled.
            if process.isRunning { process.terminate() }
        }
        phase = .finishing
    }

    /// Stops a live run. Called from the sheet's dismiss path.
    func cancel() {
        if let runningProcess, runningProcess.isRunning {
            runningProcess.terminate()
        }
        runningProcess = nil
        phase = .idle
    }

    private func record(_ snapshot: SteeringTraitProgress.Snapshot) {
        latest[snapshot.side] = snapshot
        let fraction = SteeringTraitProgress.overallFraction(
            positive: latest[.positive], negative: latest[.negative])
        phase = .capturing(fraction: fraction)
    }
}

/// Line-splitting stderr accumulator. Lock-protected because the
/// readability handler runs off the main actor; `onLine` hops to the
/// caller's actor.
final class ProgressLineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    /// The partial line not yet terminated by a newline.
    private var data = Data()
    /// The last complete non-progress lines, kept because `consume` drops
    /// every complete line from `data`: without them a failure would
    /// report a bare exit status instead of the tool's refusal.
    private var recent: [String] = []
    private static let recentLimit = 20
    private let onLine: @Sendable (SteeringTraitProgress.Snapshot) -> Void

    init(onLine: @escaping @Sendable (SteeringTraitProgress.Snapshot) -> Void) {
        self.onLine = onLine
    }

    func consume(_ chunk: Data) {
        var lines: [String] = []
        var snapshots: [SteeringTraitProgress.Snapshot] = []
        lock.lock()
        data.append(chunk)
        while let newline = data.firstIndex(of: UInt8(ascii: "\n")) {
            let line = String(decoding: data[..<newline], as: UTF8.self)
            data.removeSubrange(...newline)
            lines.append(line)
        }
        for line in lines {
            if let snapshot = SteeringTraitProgress.parse(line) {
                snapshots.append(snapshot)
            } else if !line.trimmingCharacters(in: .whitespaces).isEmpty {
                recent.append(line)
            }
        }
        if recent.count > Self.recentLimit {
            recent.removeFirst(recent.count - Self.recentLimit)
        }
        lock.unlock()
        for snapshot in snapshots {
            onLine(snapshot)
        }
    }

    /// The last `count` lines, for a failure message that names the actual
    /// refusal rather than a bare exit status.
    func tail(lines count: Int) -> String {
        lock.lock()
        defer { lock.unlock() }
        var lines = recent
        let partial = String(decoding: data, as: UTF8.self)
        if !partial.trimmingCharacters(in: .whitespaces).isEmpty {
            lines.append(partial)
        }
        return lines.suffix(count).joined(separator: "\n")
    }
}

/// One-shot exit waiter, attached at init so it must be created before
/// `run()`. The exit and the wait can arrive in either order; the lock makes
/// whichever comes second do the single continuation resume.
private final class ProcessExit: @unchecked Sendable {
    private let lock = NSLock()
    private var exitStatus: Int32?
    private var waiter: CheckedContinuation<Int32, Never>?

    init(_ process: Process) {
        process.terminationHandler = { [self] process in
            lock.lock()
            exitStatus = process.terminationStatus
            let pending = waiter
            waiter = nil
            lock.unlock()
            pending?.resume(returning: process.terminationStatus)
        }
    }

    func status() async -> Int32 {
        await withCheckedContinuation { continuation in
            lock.lock()
            if let exitStatus {
                lock.unlock()
                continuation.resume(returning: exitStatus)
            } else {
                waiter = continuation
                lock.unlock()
            }
        }
    }
}

enum SteeringTraitError: LocalizedError {
    case toolFailed(status: Int32, message: String)

    var errorDescription: String? {
        switch self {
        case let .toolFailed(status, message):
            let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty
                ? "turbospark-steer exited with status \(status)"
                : trimmed
        }
    }
}

/// Builds the preset a finished extraction lands in. Kept apart from the
/// view so the shape read and the reuse-by-path rule are testable without a
/// model open. Naming and output-path helpers stay nonisolated (pure
/// string math); [`makePreset`] is MainActor because the shape read goes
/// through the engine wrapper beside the rest of the app's model-touching
/// code.
enum SteeringTraitPresetFactory {
    /// The output path for one model-plus-trait pair, deterministic so a
    /// re-extraction overwrites its own previous file instead of stacking
    /// copies in the profile directory.
    static func outputURL(modelAlias: String, traitID: String) -> URL {
        let alias = sanitized(modelAlias)
        let trait = sanitized(traitID)
        return AppStorageRoot.subdirectory("steering-vectors")
            .appendingPathComponent("\(alias)-\(trait).gguf")
    }

    static func sanitized(_ raw: String) -> String {
        let allowed = raw.lowercased().map { character -> Character in
            character.isLetter || character.isNumber ? character : "-"
        }
        let cleaned = String(allowed)
            .reduce("") { partial, character in
                // Collapse runs of separators so an alias like "Qwen 3.8"
                // yields one dash, not three.
                if character == "-", partial.last == "-" { return partial }
                return partial + String(character)
            }
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return cleaned.isEmpty ? "model" : cleaned
    }

    /// The preset for a finished extraction, with its shape already read
    /// through the engine's own parser. Re-extracting the same pair replaces
    /// the preset that points at the same file rather than stacking rows.
    @MainActor
    static func makePreset(
        traitID: String,
        traitName: String,
        report: SteeringTraitPackReport,
        scale: Double,
        outputURL: URL,
        modelAlias: String,
        existing: [AppSteeringPreset]
    ) -> AppSteeringPreset {
        var preset = AppSteeringPreset(
            name: traitName,
            vectorPath: outputURL.path,
            mode: report.presetMode,
            scale: scale,
            notes: "Extracted from \(modelAlias) on this Mac (trait: \(traitID))",
            vectorHidden: nil,
            vectorSpannedLayers: nil)
        let shaped = AppModel.readingVectorShape(into: preset)
        preset = shaped.preset
        if let existing = existing.first(where: {
            URL(fileURLWithPath: $0.vectorPath).standardizedFileURL.path
                == outputURL.standardizedFileURL.path
        }) {
            preset.id = existing.id
        }
        return preset
    }
}
