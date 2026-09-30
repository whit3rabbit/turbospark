import Foundation
import Network

final class BrowserAutomationHTTPFixtureServer {
    private final class StartupState {
        var port: UInt16?
        var error: NWError?
    }

    private let listener: NWListener
    private let queue = DispatchQueue(label: "BrowserAutomationHTTPFixtureServer")
    private let ready = DispatchSemaphore(value: 0)
    private let startupState: StartupState
    private let lock = NSLock()
    private var requestCounts: [String: Int] = [:]

    var port: UInt16 {
        lock.lock()
        defer { lock.unlock() }
        return startupState.port ?? 0
    }

    init() throws {
        let state = StartupState()
        startupState = state
        listener = try NWListener(using: .tcp, on: .any)
        listener.stateUpdateHandler = { [weak self, listener] newState in
            guard let self else { return }
            self.lock.lock()
            switch newState {
            case .ready:
                state.port = listener.port?.rawValue
            case .failed(let error):
                state.error = error
            default:
                break
            }
            self.lock.unlock()
            if case .ready = newState { self.ready.signal() }
            if case .failed = newState { self.ready.signal() }
        }
        listener.newConnectionHandler = { [weak self] connection in
            self?.serve(connection)
        }
        listener.start(queue: queue)

        guard ready.wait(timeout: .now() + 5) == .success else {
            throw NSError(domain: "BrowserAutomationHTTPFixtureServer", code: 1)
        }
        if let error = startupState.error { throw error }
        guard startupState.port != nil else {
            throw NSError(domain: "BrowserAutomationHTTPFixtureServer", code: 2)
        }
    }

    deinit {
        listener.cancel()
    }

    func url(_ path: String) -> URL {
        URL(string: "http://127.0.0.1:\(port)\(path)")!
    }

    func requestCount(for path: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return requestCounts[path, default: 0]
    }

    private func serve(_ connection: NWConnection) {
        connection.start(queue: queue)
        connection.receive(minimumIncompleteLength: 1, maximumLength: 32_768) { [weak self] data, _, _, _ in
            guard let self, let data, let request = String(data: data, encoding: .utf8) else {
                connection.cancel()
                return
            }
            let requestTarget = request.components(separatedBy: " ").dropFirst().first ?? "/"
            let path = URLComponents(string: "http://fixture\(requestTarget)")?.path ?? "/"
            self.lock.lock()
            self.requestCounts[path, default: 0] += 1
            self.lock.unlock()

            if path == "/drop" {
                connection.cancel()
                return
            }

            let response: (status: String, headers: [String], body: Data)
            switch path {
            case "/redirect":
                response = (
                    "302 Found",
                    ["Location: http://localhost:\(self.port)/landing"],
                    Data()
                )
            case "/form":
                let html = """
                <!doctype html><html><body>
                  <form id="cross-origin" method="get" action="http://localhost:\(self.port)/submitted">
                    <button type="submit">Submit</button>
                  </form>
                </body></html>
                """
                response = ("200 OK", ["Content-Type: text/html; charset=utf-8"], Data(html.utf8))
            case "/popup-link":
                let html = """
                <!doctype html><html><body>
                  <a id="popup" target="_blank" href="http://localhost:\(self.port)/popup-target">Open</a>
                </body></html>
                """
                response = ("200 OK", ["Content-Type: text/html; charset=utf-8"], Data(html.utf8))
            case "/landing", "/submitted", "/popup-target", "/ok":
                let title = path.dropFirst().capitalized
                let html = "<!doctype html><html><head><title>\(title)</title></head><body>\(title)</body></html>"
                response = ("200 OK", ["Content-Type: text/html; charset=utf-8"], Data(html.utf8))
            default:
                response = ("404 Not Found", ["Content-Type: text/plain"], Data("missing".utf8))
            }

            var headers = response.headers
            headers.append("Content-Length: \(response.body.count)")
            headers.append("Connection: close")
            let head = (["HTTP/1.1 \(response.status)"] + headers + ["", ""]).joined(separator: "\r\n")
            var bytes = Data(head.utf8)
            bytes.append(response.body)
            connection.send(content: bytes, completion: .contentProcessed { _ in connection.cancel() })
        }
    }
}
