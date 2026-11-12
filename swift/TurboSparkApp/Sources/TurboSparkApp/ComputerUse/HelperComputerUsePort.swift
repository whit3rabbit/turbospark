import AppKit
import ComputerUseKit
import Foundation
import Security

/// The app's connection to `TurboSpark Computer Use.app`.
///
/// Launched through LaunchServices so macOS attributes its grants to the
/// helper rather than to TurboSpark (a `Process` child would inherit
/// TurboSpark as its responsible app). One helper instance per connection;
/// it exits when the socket closes.
public actor HelperComputerUsePort: ComputerUseControlPort {
    public struct Welcome: Sendable {
        public let manifest: ComputerUseBackendManifest
        public let status: ComputerUseHelperStatus
    }

    private let fd: Int32
    private let sessionDirectory: URL
    private let queue = DispatchQueue(label: "com.whit3rabbit.turbospark.computer-use-client")

    private init(fd: Int32, sessionDirectory: URL) {
        self.fd = fd
        self.sessionDirectory = sessionDirectory
    }

    /// The helper inside the running app bundle, or nil (unbundled `swift
    /// run`, tests), in which case computer use stays unavailable.
    public static func installedHelperURL(in bundle: Bundle = .main) -> URL? {
        let url = bundle.bundleURL.appendingPathComponent(ComputerUseHelperLayout.relativeBundlePath)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    public static func launch(helperURL: URL) async throws -> (HelperComputerUsePort, Welcome) {
        let directory = try makeSessionDirectory()
        let token = try randomToken()
        let tokenURL = directory.appendingPathComponent(ComputerUseHelperLayout.tokenFileName)
        guard FileManager.default.createFile(
            atPath: tokenURL.path, contents: Data(token.utf8), attributes: [.posixPermissions: 0o600])
        else {
            throw ComputerUsePortError(kind: .failed, message: "Could not prepare the helper session.")
        }

        let configuration = NSWorkspace.OpenConfiguration()
        configuration.arguments = ["--session-dir", directory.path]
        configuration.activates = false
        configuration.addsToRecentItems = false
        configuration.createsNewApplicationInstance = true
        _ = try await NSWorkspace.shared.openApplication(at: helperURL, configuration: configuration)

        let socketPath = directory.appendingPathComponent(ComputerUseHelperLayout.socketFileName).path
        let fd = try await connect(socketPath: socketPath, timeoutSeconds: 10)
        let port = HelperComputerUsePort(fd: fd, sessionDirectory: directory)
        let response = try await port.send(.hello(token: token, protocolVersion: ComputerUseProtocol.currentVersion))
        guard case .welcome(let manifest, let status) = response else {
            await port.close()
            if case .error(let kind, let message) = response {
                throw ComputerUsePortError(kind: kind, message: message)
            }
            throw ComputerUsePortError(kind: .failed, message: "The helper did not answer.")
        }
        return (port, Welcome(manifest: manifest, status: status))
    }

    public func perform(_ command: ComputerUseCommand) async throws -> ComputerUseResultValue {
        switch try await send(.command(command)) {
        case .result(let value):
            return value
        case .error(let kind, let message):
            throw ComputerUsePortError(kind: kind, message: message)
        default:
            throw ComputerUsePortError(kind: .failed, message: "The helper returned an unexpected reply.")
        }
    }

    public func resumeAfterStop() async {
        _ = try? await send(.resume)
    }

    public func status() async throws -> ComputerUseHelperStatus {
        guard case .status(let status) = try await send(.status) else {
            throw ComputerUsePortError(kind: .failed, message: "The helper returned an unexpected reply.")
        }
        return status
    }

    /// Shows the macOS prompt for `kind`. Call only from a user's click.
    public func requestPermission(_ kind: ComputerUsePermissionKind) async throws -> ComputerUseHelperStatus {
        guard case .status(let status) = try await send(.requestPermission(kind)) else {
            throw ComputerUsePortError(kind: .failed, message: "The helper returned an unexpected reply.")
        }
        return status
    }

    public func shutdown() async {
        _ = try? await send(.shutdown)
        close()
    }

    private func close() {
        Darwin.close(fd)
        try? FileManager.default.removeItem(at: sessionDirectory)
    }

    /// One request, one response. The actor serializes callers; the blocking
    /// socket IO runs on `queue`, off the cooperative pool.
    private func send(_ request: ComputerUseRequest) async throws -> ComputerUseResponse {
        let fd = self.fd
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    try ComputerUseFraming.write(request, to: fd)
                    continuation.resume(returning: try ComputerUseFraming.read(ComputerUseResponse.self, from: fd))
                } catch {
                    continuation.resume(throwing: ComputerUsePortError(
                        kind: .failed, message: "The computer-use helper is not responding."))
                }
            }
        }
    }

    /// A short path: `sockaddr_un` holds 104 bytes, which an Application
    /// Support path can exceed. The per-user temporary directory is private.
    private static func makeSessionDirectory() throws -> URL {
        let name = "tscu-" + String(try randomToken().prefix(12))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        return url
    }

    private static func randomToken() throws -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw ComputerUsePortError(kind: .failed, message: "Could not generate a session token.")
        }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }

    private static func connect(socketPath: String, timeoutSeconds: Int) async throws -> Int32 {
        let deadline = Date().addingTimeInterval(TimeInterval(timeoutSeconds))
        while Date() < deadline {
            if FileManager.default.fileExists(atPath: socketPath), let fd = tryConnect(socketPath) {
                return fd
            }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        throw ComputerUsePortError(kind: .failed, message: "The computer-use helper did not start.")
    }

    private static func tryConnect(_ path: String) -> Int32? {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        ComputerUseFraming.disableSigpipe(fd)
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        guard path.utf8.count < MemoryLayout.size(ofValue: address.sun_path) else {
            Darwin.close(fd)
            return nil
        }
        withUnsafeMutableBytes(of: &address.sun_path) { raw in
            raw.copyBytes(from: path.utf8)
            raw[path.utf8.count] = 0
        }
        let result = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard result == 0 else {
            Darwin.close(fd)
            return nil
        }
        return fd
    }
}
