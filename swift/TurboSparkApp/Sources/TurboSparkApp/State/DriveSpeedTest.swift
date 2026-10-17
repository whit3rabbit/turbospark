import Darwin
import Foundation

/// Why a speed test did not run. Cancellation is not one of these: it is
/// reported as Swift's `CancellationError`.
enum DriveSpeedTestError: Error, Equatable, Sendable {
    /// This Mac (the internal library) is never speed tested (5.7).
    case thisMac
    /// No regular file of at least `Configuration.minimumFileBytes` was found
    /// where the walk could reach it.
    case noLargeFile
    /// The drive or the file could not be opened or read. The payload is the
    /// system's errno text, never a path or a name from the drive.
    case unreadable(String)
}

/// What a finished speed test measured.
struct DriveSpeedResult: Equatable, Sendable {
    let bytesRead: Int64
    let seconds: Double

    var bytesPerSecond: Double { seconds > 0 ? Double(bytesRead) / seconds : 0 }
    /// Decimal megabytes per second, the unit of the advisor's 150 and 500
    /// MB/s thresholds and of requirement 5.6's display.
    var megabytesPerSecond: Double { bytesPerSecond / 1_000_000 }
}

/// Measures how fast a library's drive reads, with the cache out of the way
/// (requirement 5.6).
///
/// The test reads from the LARGEST regular file in the library, because that
/// is a model weight file and the thing a load or an expert stream actually
/// reads. It opens the file with `F_NOCACHE`: a read the page cache answers
/// measures memory, and would report a USB 2.0 drive as fast.
///
/// **The drive is untrusted.** Anyone who can write the drive can plant a
/// link, so the walk never follows one: every entry is classified with
/// `fstatat(AT_SYMLINK_NOFOLLOW)`, every folder is entered with
/// `openat(O_NOFOLLOW)`, and the file is opened the same way from the
/// library's own descriptor. The walk is bounded by entry count, depth and
/// time, and hidden entries (`.staging`, the download cache) are skipped.
enum DriveSpeedTest {
    /// The design's numbers by default. Tests shrink them and inject the
    /// clock and the hooks; nothing else about the file system is injectable.
    struct Configuration: Sendable {
        /// A library with no regular file at least this big is refused.
        var minimumFileBytes: Int64 = 256 << 20
        var maximumReadBytes: Int64 = 1 << 30
        var chunkBytes: Int = 8 << 20
        /// The read stops after this long, checked after each chunk.
        var timeLimit: TimeInterval = 10
        /// Finding the file stops after this long. With the read limit it keeps
        /// the whole test under requirement 5.6's 15 seconds.
        var walkTimeLimit: TimeInterval = 3
        var maximumEntries = 20_000
        var maximumDepth = 8
        /// Monotonic seconds. Sleep does not count.
        var now: @Sendable () -> TimeInterval = DriveSpeedTest.uptime
        /// Picks the starting slot, 0 through the argument, inclusive.
        var pickSlot: @Sendable (Int64) -> Int64 = { Int64.random(in: 0...max($0, 0)) }
        /// Test hooks. `afterScan` runs between choosing the file and opening
        /// it, `onOpen` gets the descriptor once caching is off, and
        /// `onChunk` gets the running chunk count after each chunk.
        var afterScan: (@Sendable () -> Void)?
        var onOpen: (@Sendable (Int32) -> Void)?
        var onChunk: (@Sendable (Int) -> Void)?
    }

    /// The largest regular file the walk found.
    struct Candidate {
        /// NUL-terminated names from the library root, kept as bytes because
        /// a drive can hold names that are not valid UTF-8.
        let components: [[CChar]]
        let size: Int64

        /// For tests and diagnostics only. Not sanitized: do not show it.
        var relativePath: String { components.map { String(cString: $0) }.joined(separator: "/") }
    }

    struct ScanOutcome {
        var best: Candidate?
        /// Entries looked at, not counting hidden ones.
        var visited: Int
        /// The walk gave up at a limit and may have missed a larger file.
        var truncated: Bool
    }

    // MARK: Entry points

    /// Runs the test on its own thread, so neither the main actor nor the
    /// cooperative pool waits on the drive. Cancelling the calling task stops
    /// the read at the next chunk and throws `CancellationError`.
    static func measureReadSpeed(
        libraryRoot: URL,
        isInternal: Bool,
        configuration: Configuration = Configuration()
    ) async throws -> DriveSpeedResult {
        let flag = CancellationFlag()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<DriveSpeedResult, Error>) in
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        continuation.resume(returning: try measure(
                            libraryRoot: libraryRoot, isInternal: isInternal,
                            configuration: configuration, isCancelled: { flag.isRaised }))
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        } onCancel: {
            flag.raise()
        }
    }

    /// The blocking measurement. `isCancelled` is polled before the walk, at
    /// each entry, and before each chunk.
    static func measure(
        libraryRoot: URL,
        isInternal: Bool,
        configuration: Configuration,
        isCancelled: @escaping () -> Bool
    ) throws -> DriveSpeedResult {
        // Before any disk access: This Mac is never tested (5.7).
        if isInternal { throw DriveSpeedTestError.thisMac }
        if isCancelled() { throw CancellationError() }

        let root = try openDirectory(libraryRoot.path)
        defer { close(root) }

        let outcome = try scan(rootDescriptor: root, configuration: configuration, isCancelled: isCancelled)
        guard let best = outcome.best, best.size >= configuration.minimumFileBytes else {
            throw DriveSpeedTestError.noLargeFile
        }
        configuration.afterScan?()
        if isCancelled() { throw CancellationError() }

        let file = try openForUncachedRead(root: root, components: best.components)
        defer { close(file) }
        configuration.onOpen?(file)

        // Trust the descriptor, not the walk: the file may have changed since.
        var info = stat()
        guard fstat(file, &info) == 0 else { throw DriveSpeedTestError.unreadable(errnoText()) }
        guard (info.st_mode & S_IFMT) == S_IFREG, Int64(info.st_size) >= configuration.minimumFileBytes else {
            throw DriveSpeedTestError.noLargeFile
        }
        let fileBytes = Int64(info.st_size)

        let chunk = max(configuration.chunkBytes, 4_096)
        let planned = min(configuration.maximumReadBytes, fileBytes)
        let offset = startOffset(fileBytes: fileBytes, readBytes: planned, chunkBytes: Int64(chunk), pickSlot: configuration.pickSlot)

        // A page-aligned buffer keeps uncached reads on the direct path.
        var raw: UnsafeMutableRawPointer?
        guard posix_memalign(&raw, Int(getpagesize()), chunk) == 0, let buffer = raw else {
            throw DriveSpeedTestError.unreadable(errnoText(ENOMEM))
        }
        defer { free(buffer) }

        var total: Int64 = 0
        var chunks = 0
        let started = configuration.now()
        var finished = started
        while total < planned {
            if isCancelled() { throw CancellationError() }
            let wanted = Int(min(Int64(chunk), planned - total))
            var count = pread(file, buffer, wanted, off_t(offset + total))
            while count < 0 && errno == EINTR {
                count = pread(file, buffer, wanted, off_t(offset + total))
            }
            if count < 0 { throw DriveSpeedTestError.unreadable(errnoText()) }
            if count == 0 { break }   // the file shrank under the read
            total += Int64(count)
            chunks += 1
            configuration.onChunk?(chunks)
            finished = configuration.now()
            // At least one chunk is read, so a rate always exists.
            if finished - started >= configuration.timeLimit { break }
        }
        guard total > 0 else { throw DriveSpeedTestError.noLargeFile }
        return DriveSpeedResult(bytesRead: total, seconds: max(finished - started, 1e-9))
    }

    // MARK: Finding the file

    /// Walks the library for its largest regular file. Exposed for tests of
    /// the limits; `measure` is the product entry point.
    static func scanLargest(
        libraryRoot: URL,
        configuration: Configuration,
        isCancelled: @escaping () -> Bool
    ) throws -> ScanOutcome {
        let root = try openDirectory(libraryRoot.path)
        defer { close(root) }
        return try scan(rootDescriptor: root, configuration: configuration, isCancelled: isCancelled)
    }

    private static func scan(
        rootDescriptor: Int32,
        configuration: Configuration,
        isCancelled: @escaping () -> Bool
    ) throws -> ScanOutcome {
        // A duplicate, because closedir closes the descriptor it was given and
        // the caller still needs the root for the open.
        let duplicate = dup(rootDescriptor)
        guard duplicate >= 0, let directory = fdopendir(duplicate) else {
            let reason = errnoText()
            if duplicate >= 0 { close(duplicate) }
            throw DriveSpeedTestError.unreadable(reason)
        }
        defer { closedir(directory) }

        let walker = Walker(configuration: configuration, isCancelled: isCancelled)
        try walker.scan(directory, prefix: [], depth: 0)
        return ScanOutcome(best: walker.best, visited: walker.visited, truncated: walker.truncated)
    }

    /// Where in the file the read starts: a random multiple of the chunk size
    /// that still leaves room for the whole read, so a read near the end of a
    /// big file is never cut short by the end.
    static func startOffset(
        fileBytes: Int64,
        readBytes: Int64,
        chunkBytes: Int64,
        pickSlot: (Int64) -> Int64
    ) -> Int64 {
        guard chunkBytes > 0 else { return 0 }
        let highest = max(fileBytes - readBytes, 0) / chunkBytes
        return min(max(pickSlot(highest), 0), highest) * chunkBytes
    }

    // MARK: Descriptors

    /// Opens a folder without following a link in its last component.
    fileprivate static func openDirectory(_ path: String) throws -> Int32 {
        let descriptor = open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw DriveSpeedTestError.unreadable(errnoText()) }
        return descriptor
    }

    /// Opens the chosen file from the library's own descriptor, one component
    /// at a time, refusing a link at every step, and turns caching off.
    private static func openForUncachedRead(root: Int32, components: [[CChar]]) throws -> Int32 {
        guard let last = components.last else { throw DriveSpeedTestError.noLargeFile }
        var current = root
        var ownsCurrent = false
        defer { if ownsCurrent { close(current) } }
        for name in components.dropLast() {
            let next = name.withUnsafeBufferPointer {
                openat(current, $0.baseAddress!, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            }
            guard next >= 0 else { throw DriveSpeedTestError.unreadable(errnoText()) }
            if ownsCurrent { close(current) }
            current = next
            ownsCurrent = true
        }
        // O_NONBLOCK so a pipe swapped in after the walk cannot hang the open.
        let file = last.withUnsafeBufferPointer {
            openat(current, $0.baseAddress!, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        }
        guard file >= 0 else { throw DriveSpeedTestError.unreadable(errnoText()) }
        // A cached read would measure memory. If caching cannot be turned off
        // there is no honest number to give.
        guard fcntl(file, F_NOCACHE, 1) == 0 else {
            let reason = errnoText()
            close(file)
            throw DriveSpeedTestError.unreadable(reason)
        }
        return file
    }

    fileprivate static func errnoText(_ code: Int32 = errno) -> String {
        String(cString: strerror(code))
    }

    @Sendable static func uptime() -> TimeInterval {
        Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1_000_000_000
    }
}

/// The walk behind `DriveSpeedTest.scanLargest`.
fileprivate final class Walker {
    private let configuration: DriveSpeedTest.Configuration
    private let isCancelled: () -> Bool
    private let started: TimeInterval
    private(set) var visited = 0
    private(set) var truncated = false
    private(set) var best: DriveSpeedTest.Candidate?

    init(configuration: DriveSpeedTest.Configuration, isCancelled: @escaping () -> Bool) {
        self.configuration = configuration
        self.isCancelled = isCancelled
        self.started = configuration.now()
    }

    func scan(_ directory: UnsafeMutablePointer<DIR>, prefix: [[CChar]], depth: Int) throws {
        let descriptor = dirfd(directory)
        while let entry = readdir(directory) {
            let name = Walker.name(of: entry)
            // ".", ".." and hidden entries such as `.staging`.
            if name.first == CChar(UInt8(ascii: ".")) { continue }
            if isCancelled() { throw CancellationError() }
            if visited >= configuration.maximumEntries || configuration.now() - started >= configuration.walkTimeLimit {
                truncated = true
                return
            }
            visited += 1

            // lstat semantics: a link is classified as a link, never followed.
            var info = stat()
            let status = name.withUnsafeBufferPointer { fstatat(descriptor, $0.baseAddress!, &info, AT_SYMLINK_NOFOLLOW) }
            guard status == 0 else { continue }

            switch info.st_mode & S_IFMT {
            case S_IFREG:
                let size = Int64(info.st_size)
                if size > (best?.size ?? 0) {
                    best = DriveSpeedTest.Candidate(components: prefix + [name], size: size)
                }
            case S_IFDIR:
                guard depth < configuration.maximumDepth else { continue }
                let child = name.withUnsafeBufferPointer {
                    openat(descriptor, $0.baseAddress!, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                }
                guard child >= 0 else { continue }
                guard let childDirectory = fdopendir(child) else {
                    close(child)
                    continue
                }
                defer { closedir(childDirectory) }
                try scan(childDirectory, prefix: prefix + [name], depth: depth + 1)
                if truncated { return }
            default:
                // Links, devices, pipes and sockets.
                continue
            }
        }
    }

    private static func name(of entry: UnsafeMutablePointer<dirent>) -> [CChar] {
        let length = Int(entry.pointee.d_namlen)
        return withUnsafePointer(to: &entry.pointee.d_name) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: length + 1) { characters in
                Array(UnsafeBufferPointer(start: characters, count: length)) + [0]
            }
        }
    }
}

/// Set from the cancellation handler, read from the reader's thread.
private final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var raised = false

    var isRaised: Bool {
        lock.lock()
        defer { lock.unlock() }
        return raised
    }

    func raise() {
        lock.lock()
        raised = true
        lock.unlock()
    }
}
