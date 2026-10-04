import Darwin
import Foundation

/// Reads the REPL file roots that are current right now for a session's
/// chat. The production source resolves the chat's attached project and
/// returns its `AppProjectPermissions.replFileAccessRoots`; the broker calls
/// it again for every operation, so a revoked or narrowed grant takes effect
/// on the very next request (5.5, 5.7). Workers never receive this list.
typealias REPLFileGrantSource = @Sendable (_ chatID: UUID?) async -> [String]

/// One bounded worker-to-broker file request. The request carries no chat
/// identity and no grant information; the parent side of the channel owns
/// the identity and the broker owns the grants.
struct REPLFileAccessRequest: Sendable, Equatable {
    enum Operation: Sendable, Equatable {
        case readFile
        case writeFile
        case list

        /// Maps the facade's fixed wire name; nil for anything the facade
        /// never sends.
        static func named(_ name: String) -> Operation? {
            switch name {
            case "readFile": return .readFile
            case "writeFile": return .writeFile
            case "list": return .list
            default: return nil
            }
        }
    }

    var id: UUID
    var operation: Operation
    var path: String
    var payload: Data?
}

/// Typed broker response: bounded data, an entry listing, a completed write,
/// or a denial whose text is safe to surface as a script error.
enum REPLFileAccessResult: Sendable, Equatable {
    case data(Data)
    case entries([String])
    case written
    case denied(String)
}

/// The bounded request channel a worker-side facade uses to reach the
/// app-side file broker. Task 3.1 binds this seam to the real
/// worker-process IPC; the request and result types above are the wire
/// contract, so that binding needs no facade redesign. The worker side of
/// the channel sees only requests and results, never grants or identity.
protocol REPLFileRequestChannel: Sendable {
    /// Sends one bounded file request and awaits its typed result.
    func send(_ request: REPLFileAccessRequest) async -> REPLFileAccessResult
}

/// In-process channel binding used until task 3.1 provides real
/// worker-process IPC. The chat identity is fixed by the app side when the
/// channel is built; worker code can neither read nor override it.
struct REPLInProcessFileChannel: REPLFileRequestChannel {
    let broker: REPLFileAccessBroker
    let chatID: UUID?

    func send(_ request: REPLFileAccessRequest) async -> REPLFileAccessResult {
        await broker.perform(request, chatID: chatID)
    }
}

/// Fail-closed channel for workers created without a broker binding: every
/// operation denies, because no capability was granted (5.5).
struct REPLNoAccessFileChannel: REPLFileRequestChannel {
    func send(_ request: REPLFileAccessRequest) async -> REPLFileAccessResult {
        .denied("REPL file access denied: no file access is granted to this session.")
    }
}

/// App-side broker for every worker file operation. For each request it
/// reads the chat's CURRENT user-configured roots, canonicalizes the
/// requested path with symlink resolution, and performs the read, write, or
/// list only when the canonical path remains under a currently granted
/// root. It never trusts a root list captured when the worker was created,
/// so revoked grants fail closed on the next operation.
///
/// Canonicalization mirrors the house `AppToolRegistry.resolveSecurePath`
/// discipline: symlinks are resolved on BOTH sides before the prefix
/// compare, so a symlink planted inside a root cannot pivot the check
/// elsewhere. This broker extends that discipline to multiple roots and
/// deliberately omits the project tools' spill-root exception, because the
/// REPL facade grants arbitrary user-chosen absolute roots and nothing
/// else.
final class REPLFileAccessBroker: Sendable {
    private let grants: REPLFileGrantSource

    init(grants: @escaping REPLFileGrantSource) {
        self.grants = grants
    }

    /// Bound on the request path, matching a PATH_MAX-scale transport bound.
    static let maximumPathCharacters = 4_096

    /// Bound on payloads in both directions. Reuses the app's existing
    /// single-read ceiling (`AppFileReadLimits.maximumBytes`) so the REPL
    /// facade never moves more bytes per request than the file tools may
    /// read in one call.
    static let maximumTransferBytes = AppFileReadLimits.maximumBytes

    func perform(
        _ request: REPLFileAccessRequest,
        chatID: UUID?
    ) async -> REPLFileAccessResult {
        guard request.path.count <= Self.maximumPathCharacters else {
            return .denied("REPL file request denied: the path is too long.")
        }
        if let payload = request.payload, payload.count > Self.maximumTransferBytes {
            return .denied("REPL file request denied: the payload is over the transfer limit.")
        }
        // Only the parent side supplies the chat identity. Without it no
        // project grant can be resolved, so the broker denies outright
        // rather than guessing a project.
        guard let chatID else {
            return .denied(
                "REPL file access denied: this session has no chat identity, "
                    + "so no project file grant can be resolved.")
        }
        let roots = await grants(chatID)
        guard !roots.isEmpty else {
            return .denied("REPL file access denied: no file root is granted for this session.")
        }
        guard let target = Self.canonicalTarget(for: request.path, under: roots) else {
            return .denied("REPL file access denied: the path is outside the granted file roots.")
        }
        switch request.operation {
        case .readFile:
            return Self.read(at: target)
        case .writeFile:
            return Self.write(request.payload, at: target)
        case .list:
            return Self.list(at: target)
        }
    }

    /// Resolves the requested path against the granted roots. A relative
    /// path is tried against every granted root; an absolute path is used
    /// as-is. The first root whose canonical form contains the canonical
    /// target wins; a path that canonicalizes outside every root returns
    /// nil regardless of how it was spelled.
    static func canonicalTarget(for rawPath: String, under rootPaths: [String]) -> URL? {
        let cleaned = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty, !cleaned.hasPrefix("~") else { return nil }
        for rootPath in rootPaths {
            let trimmedRoot = rootPath.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedRoot.isEmpty, !trimmedRoot.hasPrefix("~") else { continue }
            let root = URL(fileURLWithPath: trimmedRoot, isDirectory: true)
                .standardizedFileURL
                .resolvingSymlinksInPath()
            let candidate = (cleaned.hasPrefix("/")
                ? URL(fileURLWithPath: cleaned)
                : root.appendingPathComponent(cleaned))
                .standardizedFileURL
                .resolvingSymlinksInPath()
            if candidate.path == root.path || candidate.path.hasPrefix(root.path + "/") {
                return candidate
            }
        }
        return nil
    }

    /// Reads through a descriptor opened with O_NOFOLLOW so a symlink
    /// swapped into the final component between canonicalization and the
    /// open cannot redirect the read.
    private static func read(at url: URL) -> REPLFileAccessResult {
        let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW)
        guard descriptor >= 0 else {
            return .denied(Self.ioFailure("read", url, errno))
        }
        defer { close(descriptor) }
        var status = stat()
        guard fstat(descriptor, &status) == 0 else {
            return .denied(Self.ioFailure("stat", url, errno))
        }
        let size = Int(status.st_size)
        let cap = AppFileReadLimits.maximumBytes
        if size > cap {
            return .denied(
                "REPL file read denied: \(url.lastPathComponent) is "
                    + "\(size / 1_024 / 1_024) MB, over the "
                    + "\(cap / 1_024 / 1_024) MB limit for a single read.")
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        do {
            guard let data = try handle.readToEnd() else {
                return .denied("REPL file read failed: the file could not be read.")
            }
            return .data(data)
        } catch {
            return .denied("REPL file read failed: \(error.localizedDescription)")
        }
    }

    /// Writes through a descriptor opened with O_CREAT | O_TRUNC | O_NOFOLLOW
    /// so a symlink planted at the destination name cannot redirect the
    /// write outside the granted root.
    private static func write(_ payload: Data?, at url: URL) -> REPLFileAccessResult {
        guard let payload else {
            return .denied("REPL file write failed: writeFile requires string contents.")
        }
        let descriptor = open(url.path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0o644)
        guard descriptor >= 0 else {
            return .denied(Self.ioFailure("write", url, errno))
        }
        defer { close(descriptor) }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        do {
            try handle.write(contentsOf: payload)
            return .written
        } catch {
            return .denied("REPL file write failed: \(error.localizedDescription)")
        }
    }

    /// Lists a directory. The O_DIRECTORY | O_NOFOLLOW pre-open validates
    /// that the final component is a real directory rather than a symlink
    /// swapped in after canonicalization; the listing then reads the same
    /// canonical path.
    private static func list(at url: URL) -> REPLFileAccessResult {
        let descriptor = open(url.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW)
        guard descriptor >= 0 else {
            return .denied(Self.ioFailure("list", url, errno))
        }
        close(descriptor)
        do {
            return .entries(try FileManager.default.contentsOfDirectory(atPath: url.path).sorted())
        } catch {
            return .denied("REPL file list failed: \(error.localizedDescription)")
        }
    }

    /// Names only the requested file's last component. An IO failure is
    /// surfaced to the script as a rejection, and a relative request must
    /// not leak the granted root's absolute location through that text
    /// (the size-cap denial already follows the same rule).
    private static func ioFailure(_ operation: String, _ url: URL, _ code: Int32) -> String {
        let reason = String(cString: strerror(code))
        return "REPL file \(operation) failed for \(url.lastPathComponent): \(reason)."
    }
}
