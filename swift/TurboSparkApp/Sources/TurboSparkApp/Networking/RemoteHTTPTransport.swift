import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

struct RemoteHTTPRequest: Sendable {
    var url: URL
    var method: String
    var headers: [String: String]
    var body: Data
}

struct RemoteHTTPResponse: Sendable {
    var status: Int
    /// Lowercased names.
    var headers: [String: String]
    /// Raw body chunks. Ends when the server closes; throws on a transport
    /// failure. Cancelling the consumer must cancel the connection.
    var body: AsyncThrowingStream<Data, Error>
}

/// The one seam between the family clients and the network. Tests inject a
/// scripted transport; the app uses `URLSessionRemoteTransport`. A transport
/// never follows a redirect itself: it returns the 3xx and its `location`
/// header, and `RemoteStreamRunner` decides under `RemoteEndpointPolicy`.
protocol RemoteHTTPTransport: Sendable {
    func send(_ request: RemoteHTTPRequest) async throws -> RemoteHTTPResponse
}

/// The production transport.
///
/// An ephemeral session with no cookie jar, no URL cache and no connectivity
/// waiting, so a key header is never persisted or replayed from a cache.
/// TLS is the system default (ATS), which is why a non-loopback `http` base
/// URL is refused earlier by `RemoteEndpointPolicy`. Redirects are captured
/// by the shared `HttpRequestRedirectDelegate` (it answers the redirect with
/// nil, so the 3xx surfaces here) and never followed by URLSession.
struct URLSessionRemoteTransport: RemoteHTTPTransport {
    let firstByteTimeout: TimeInterval

    init(firstByteTimeout: TimeInterval = RemoteLimits.standard.firstByteTimeout) {
        self.firstByteTimeout = firstByteTimeout
    }

    func send(_ request: RemoteHTTPRequest) async throws -> RemoteHTTPResponse {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.waitsForConnectivity = false
        configuration.timeoutIntervalForRequest = firstByteTimeout
        let session = URLSession(configuration: configuration)

        var urlRequest = URLRequest(url: request.url)
        urlRequest.httpMethod = request.method
        urlRequest.httpBody = request.body.isEmpty ? nil : request.body
        for (name, value) in request.headers { urlRequest.setValue(value, forHTTPHeaderField: name) }

        let redirect = HttpRequestRedirectDelegate()
        let bytes: URLSession.AsyncBytes
        let response: URLResponse
        do {
            (bytes, response) = try await session.bytes(for: urlRequest, delegate: redirect)
        } catch {
            session.invalidateAndCancel()
            throw error
        }
        guard let http = response as? HTTPURLResponse else {
            session.invalidateAndCancel()
            throw RemoteProviderError.badStream("non-HTTP response")
        }
        var headers: [String: String] = [:]
        for (key, value) in http.allHeaderFields {
            if let k = key as? String, let v = value as? String { headers[k.lowercased()] = v }
        }
        if let target = redirect.takeRedirect()?.url, headers["location"] == nil {
            headers["location"] = target.absoluteString
        }
        let stream = AsyncThrowingStream<Data, Error> { continuation in
            let task = Task {
                do {
                    var chunk = Data()
                    for try await byte in bytes {
                        chunk.append(byte)
                        // Flush per line so a short event does not wait on a
                        // full buffer.
                        if byte == 0x0A || chunk.count >= 4096 {
                            continuation.yield(chunk)
                            chunk.removeAll(keepingCapacity: true)
                        }
                    }
                    if !chunk.isEmpty { continuation.yield(chunk) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                session.finishTasksAndInvalidate()
            }
            continuation.onTermination = { _ in
                task.cancel()
                session.invalidateAndCancel()
            }
        }
        return RemoteHTTPResponse(status: http.statusCode, headers: headers, body: stream)
    }
}
