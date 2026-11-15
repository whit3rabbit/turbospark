import CryptoKit
import Foundation

/// Downloads the zip of a `cdnZip` plugin source and proves its bytes (ZP-05).
///
/// Runs only from `PluginMarketplaceManager.install`, which is the user's
/// explicit Install action: nothing here is reachable from launch, from
/// marketplace listing, or from a timer, so the offline default holds.
///
/// Rules, each with a test:
/// - the URL must be `https` (no userinfo, no fragment); `http` is refused;
/// - the declared digest must be exactly 64 lowercase hex characters, so a
///   missing, short, upper-case or otherwise malformed digest is refused
///   BEFORE any request is made;
/// - redirects are followed at most `maxRedirects` times and only to `https`;
///   a redirect to `http` or past the bound fails the install;
/// - the body is streamed to a file under a byte cap (an early refusal from
///   Content-Length, and a hard stop at the cap for a server that lies or
///   omits it) while the SHA-256 is computed over exactly those bytes;
/// - a digest mismatch deletes the file and throws: nothing is extracted,
///   nothing is installed.
///
/// The digest sits in the same marketplace row as the URL, so it proves the
/// bytes are the ones the row names (corruption, a swapped CDN object), not
/// that the row itself is trustworthy. The marketplace the user added is the
/// trust anchor; a signed catalog would be a separate decision (ZP-05 draft).
enum CdnZipDownloader {
    /// Test seam. Production is `.production`: https only. `allowLoopbackHTTP`
    /// lets the FIRST request (never a redirect) use plain http to 127.0.0.1
    /// so a local fixture server can stand in for the CDN; it is only
    /// reachable through the internal `PluginMarketplaceManager` init.
    struct Policy: Equatable {
        var allowLoopbackHTTP = false
        var maxDownloadBytes = 64 * 1_024 * 1_024
        var maxRedirects = 3
        var idleTimeout: TimeInterval = 30
        var totalTimeout: TimeInterval = 180
        static let production = Policy()
    }

    enum Failure: Error, Equatable, LocalizedError {
        case invalidURL(String)
        case insecureURL(String)
        case invalidDigest
        case badStatus(Int)
        case tooLarge(Int)
        case redirectRefused(String)
        case digestMismatch(expected: String, actual: String)
        case timedOut
        case transport(String)

        var errorDescription: String? {
            switch self {
            case .invalidURL(let raw): return "The plugin archive URL is not valid: \(raw)"
            case .insecureURL(let raw): return "The plugin archive URL must use https: \(raw)"
            case .invalidDigest:
                return "The marketplace entry needs a sha256 of exactly 64 lowercase hex characters for its archive."
            case .badStatus(let code): return "The archive server answered HTTP \(code)."
            case .tooLarge(let cap): return "The archive is over the \(cap / (1_024 * 1_024)) MiB download limit."
            case .redirectRefused(let why): return "The archive download redirect was refused: \(why)."
            case .digestMismatch(let expected, let actual):
                return "The downloaded archive does not match its declared sha256 (expected \(expected), got \(actual)). Nothing was installed."
            case .timedOut: return "The archive download took too long."
            case .transport(let message): return "The archive download failed: \(message)"
            }
        }
    }

    /// Exactly 64 lowercase hex characters.
    static func isValidDigest(_ value: String) -> Bool {
        value.utf8.count == 64
            && value.utf8.allSatisfy { ($0 >= 0x30 && $0 <= 0x39) || ($0 >= 0x61 && $0 <= 0x66) }
    }

    /// Validates the row's URL and digest without touching the network.
    static func validate(url raw: String, sha256: String, policy: Policy = .production) throws -> URL {
        guard isValidDigest(sha256) else { throw Failure.invalidDigest }
        guard let url = URL(string: raw), let scheme = url.scheme?.lowercased(),
            let host = url.host, !host.isEmpty
        else { throw Failure.invalidURL(raw) }
        guard url.user == nil, url.password == nil, url.fragment == nil else {
            throw Failure.invalidURL(raw)
        }
        if scheme == "https" { return url }
        if scheme == "http", policy.allowLoopbackHTTP, host == "127.0.0.1" { return url }
        throw Failure.insecureURL(raw)
    }

    /// Whether a redirect hop is acceptable: https only (the loopback
    /// allowance never applies to a redirect), no credentials in the URL, and
    /// within the hop bound. `hop` is 1 for the first redirect.
    static func redirectVerdict(to target: URL?, hop: Int, policy: Policy) -> Failure? {
        guard hop <= policy.maxRedirects else {
            return .redirectRefused("more than \(policy.maxRedirects) redirects")
        }
        guard let target, target.scheme?.lowercased() == "https", let host = target.host,
            !host.isEmpty, target.user == nil, target.password == nil
        else { return .redirectRefused("only https targets are allowed") }
        return nil
    }

    private final class RedirectGuard: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        let policy: Policy
        private let lock = NSLock()
        private var hops = 0
        private var refusal: Failure?

        init(policy: Policy) { self.policy = policy }

        var failure: Failure? {
            lock.lock()
            defer { lock.unlock() }
            return refusal
        }

        func urlSession(
            _ session: URLSession, task: URLSessionTask,
            willPerformHTTPRedirection response: HTTPURLResponse,
            newRequest request: URLRequest,
            completionHandler: @escaping (URLRequest?) -> Void
        ) {
            lock.lock()
            hops += 1
            let verdict = CdnZipDownloader.redirectVerdict(to: request.url, hop: hops, policy: policy)
            if let verdict { refusal = verdict }
            lock.unlock()
            // nil refuses the hop: the task then completes with the 3xx
            // response, which the caller turns into the recorded refusal.
            completionHandler(verdict == nil ? request : nil)
        }
    }

    /// Downloads to `destination` (a file path in a caller-owned temp dir),
    /// verifies the digest, and returns the verified file. On any failure the
    /// partial file is removed.
    @discardableResult
    static func download(
        url raw: String, sha256: String, to destination: URL, policy: Policy = .production
    ) async throws -> URL {
        let url = try validate(url: raw, sha256: sha256, policy: policy)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = policy.idleTimeout
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }
        let redirectGuard = RedirectGuard(policy: policy)

        var request = URLRequest(url: url)
        // Identity, so the counted and hashed bytes are the bytes on the wire.
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        request.setValue("application/zip, application/octet-stream", forHTTPHeaderField: "Accept")

        FileManager.default.createFile(atPath: destination.path, contents: nil)
        guard let handle = try? FileHandle(forWritingTo: destination) else {
            throw Failure.transport("could not create the download file")
        }
        var finished = false
        defer {
            try? handle.close()
            if !finished { try? FileManager.default.removeItem(at: destination) }
        }

        let startedNanos = DispatchTime.now().uptimeNanoseconds
        let totalNanos = UInt64(policy.totalTimeout * 1_000_000_000)
        var hasher = SHA256()
        var received = 0
        do {
            let (bytes, response) = try await session.bytes(for: request, delegate: redirectGuard)
            if let refusal = redirectGuard.failure { throw refusal }
            guard let http = response as? HTTPURLResponse else {
                throw Failure.transport("not an HTTP response")
            }
            guard http.statusCode == 200 else { throw Failure.badStatus(http.statusCode) }
            if http.expectedContentLength > Int64(policy.maxDownloadBytes) {
                throw Failure.tooLarge(policy.maxDownloadBytes)
            }
            var buffer = Data()
            buffer.reserveCapacity(64 * 1_024)
            for try await byte in bytes {
                received += 1
                // Per byte, so a server that drips one byte per idle window
                // still hits the total deadline.
                if DispatchTime.now().uptimeNanoseconds - startedNanos > totalNanos {
                    throw Failure.timedOut
                }
                guard received <= policy.maxDownloadBytes else {
                    throw Failure.tooLarge(policy.maxDownloadBytes)
                }
                buffer.append(byte)
                if buffer.count >= 64 * 1_024 {
                    hasher.update(data: buffer)
                    try handle.write(contentsOf: buffer)
                    buffer.removeAll(keepingCapacity: true)
                }
            }
            if !buffer.isEmpty {
                hasher.update(data: buffer)
                try handle.write(contentsOf: buffer)
            }
        } catch let failure as Failure {
            throw failure
        } catch {
            if let refusal = redirectGuard.failure { throw refusal }
            throw Failure.transport(error.localizedDescription)
        }

        let actual = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard actual == sha256 else { throw Failure.digestMismatch(expected: sha256, actual: actual) }
        finished = true
        return destination
    }
}
