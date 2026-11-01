import ComputerUseKit
import Foundation
import Security

/// Serves exactly one authenticated connection, then exits with it.
///
/// The app creates `sessionDirectory` (0700) and writes a one-time token file
/// into it before launching this helper. The token never travels in argv,
/// which any same-user process can read with `ps`.
final class HelperServer: @unchecked Sendable {
    /// How long the app has to connect before the helper gives up.
    static let acceptTimeoutSeconds: Int32 = 30

    private let sessionDirectory: URL
    private let service: HelperService

    init(sessionDirectory: URL, service: HelperService) {
        self.sessionDirectory = sessionDirectory
        self.service = service
    }

    func start() {
        let thread = Thread { [self] in
            let code = run()
            DispatchQueue.main.async { exit(code) }
        }
        thread.name = "computer-use-server"
        thread.start()
    }

    private func run() -> Int32 {
        guard let token = takeToken() else {
            log("refusing: session directory or token is missing or not private")
            return 77
        }
        let socketPath = sessionDirectory.appendingPathComponent(ComputerUseHelperLayout.socketFileName).path
        guard let listener = listen(at: socketPath) else { return 71 }
        defer {
            close(listener)
            unlink(socketPath)
        }
        guard let fd = acceptOne(listener) else {
            log("no connection within \(Self.acceptTimeoutSeconds)s")
            return 0
        }
        defer { close(fd) }

        let peer = peerInfo(fd)
        guard case .hello(let presented, let version)? = try? ComputerUseFraming.read(ComputerUseRequest.self, from: fd),
              version == ComputerUseProtocol.currentVersion,
              ComputerUsePeerPolicy.accepts(
                  peer: peer,
                  ownUID: getuid(),
                  expectedBundlePath: ComputerUsePeerPolicy
                      .containingAppBundle(helperBundleURL: Bundle.main.bundleURL)?.path,
                  presentedToken: presented,
                  expectedToken: token)
        else {
            try? ComputerUseFraming.write(
                ComputerUseResponse.error(kind: .unauthorized, message: "Connection refused."), to: fd)
            log("refused peer \(peer.bundlePath ?? "?")")
            return 77
        }

        let welcome = onMain { await self.service.welcome() }
        guard (try? ComputerUseFraming.write(welcome, to: fd)) != nil else { return 0 }

        while let request = try? ComputerUseFraming.read(ComputerUseRequest.self, from: fd) {
            if case .shutdown = request {
                try? ComputerUseFraming.write(ComputerUseResponse.ok, to: fd)
                return 0
            }
            let response = onMain { await self.service.handle(request) }
            guard (try? ComputerUseFraming.write(response, to: fd)) != nil else { break }
        }
        // The app closed the socket (quit, crash, or turned the feature off).
        return 0
    }

    /// Reads and deletes the token. The directory must be ours and private,
    /// or another user could have planted it.
    private func takeToken() -> String? {
        var info = stat()
        guard lstat(sessionDirectory.path, &info) == 0,
              (info.st_mode & S_IFMT) == S_IFDIR,
              info.st_uid == getuid(),
              (info.st_mode & 0o077) == 0
        else { return nil }
        let tokenURL = sessionDirectory.appendingPathComponent(ComputerUseHelperLayout.tokenFileName)
        defer { unlink(tokenURL.path) }
        guard let data = try? Data(contentsOf: tokenURL),
              let token = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
              token.count >= 32
        else { return nil }
        return token
    }

    private func listen(at path: String) -> Int32? {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        let capacity = MemoryLayout.size(ofValue: address.sun_path)
        guard path.utf8.count < capacity else {
            close(fd)
            log("socket path too long")
            return nil
        }
        withUnsafeMutableBytes(of: &address.sun_path) { raw in
            raw.copyBytes(from: path.utf8)
            raw[path.utf8.count] = 0
        }
        unlink(path)
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bound == 0, chmod(path, 0o600) == 0, Darwin.listen(fd, 1) == 0 else {
            close(fd)
            return nil
        }
        return fd
    }

    private func acceptOne(_ listener: Int32) -> Int32? {
        var poller = pollfd(fd: listener, events: Int16(POLLIN), revents: 0)
        guard poll(&poller, 1, Self.acceptTimeoutSeconds * 1_000) > 0 else { return nil }
        let fd = accept(listener, nil, nil)
        guard fd >= 0 else { return nil }
        ComputerUseFraming.disableSigpipe(fd)
        return fd
    }

    /// The peer's uid, and its on-disk path and signature validity, both
    /// taken from the socket's audit token so a recycled pid cannot stand in.
    private func peerInfo(_ fd: Int32) -> ComputerUsePeer {
        var uid: uid_t = .max
        var gid: gid_t = .max
        _ = getpeereid(fd, &uid, &gid)

        var token = audit_token_t()
        var length = socklen_t(MemoryLayout<audit_token_t>.size)
        // SOL_LOCAL = 0, LOCAL_PEERTOKEN = 6 (sys/un.h).
        guard getsockopt(fd, 0, 6, &token, &length) == 0 else {
            return ComputerUsePeer(uid: uid, bundlePath: nil, signatureValid: false)
        }
        let tokenData = withUnsafeBytes(of: &token) { Data($0) }
        var code: SecCode?
        let attributes = [kSecGuestAttributeAudit: tokenData] as CFDictionary
        guard SecCodeCopyGuestWithAttributes(nil, attributes, [], &code) == errSecSuccess, let code else {
            return ComputerUsePeer(uid: uid, bundlePath: nil, signatureValid: false)
        }
        let valid = SecCodeCheckValidity(code, [], nil) == errSecSuccess
        var staticCode: SecStaticCode?
        var pathURL: CFURL?
        if SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode {
            _ = SecCodeCopyPath(staticCode, [], &pathURL)
        }
        return ComputerUsePeer(uid: uid, bundlePath: (pathURL as URL?)?.path, signatureValid: valid)
    }

    /// Runs async main-actor work from this blocking server thread.
    private func onMain(_ work: @escaping @MainActor () async -> ComputerUseResponse) -> ComputerUseResponse {
        let semaphore = DispatchSemaphore(value: 0)
        let box = ResponseBox()
        Task { @MainActor in
            box.value = await work()
            semaphore.signal()
        }
        semaphore.wait()
        return box.value ?? .error(kind: .failed, message: "No response.")
    }

    private func log(_ message: String) {
        FileHandle.standardError.write(Data("turbospark-computer-use: \(message)\n".utf8))
    }
}

private final class ResponseBox: @unchecked Sendable {
    var value: ComputerUseResponse?
}
