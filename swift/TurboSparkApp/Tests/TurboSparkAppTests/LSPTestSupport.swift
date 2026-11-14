import Foundation
@testable import TurboSparkApp

/// OC-10 test double: a language server that lives in the test process. It
/// speaks real Content-Length framing over the `LSPTransport` seam, so the
/// client's codec, handshake and routing are exercised without spawning
/// anything. No real language server is ever started by these tests.
final class ScriptedLSPServer: LSPTransport, @unchecked Sendable {
    struct Behavior {
        var serverName = "fake-ls"
        /// Diagnostics (as LSP JSON) to publish for a synced document.
        var diagnostics: @Sendable (_ uri: String, _ text: String) -> [[String: Any]] = { _, _ in [] }
        /// When false the server never publishes (a hung or slow server).
        var publishes = true
        /// Added to the document version in the published message; -1 makes
        /// every publish look stale.
        var versionOffset = 0
        /// Methods the server never answers.
        var silentMethods: Set<String> = []
        var documentSymbols: Any = [Any]()
        var workspaceSymbols: Any = [Any]()
        /// A server-initiated request sent right after `initialized`.
        var serverRequestAfterInit: [String: Any]?
    }

    private let queue = DispatchQueue(label: "scripted-lsp")
    private let lock = NSLock()
    private var decoder = LSPFrameDecoder()
    private var onData: (@Sendable (Data) -> Void)?
    private var onClose: (@Sendable () -> Void)?
    private var _received: [[String: Any]] = []
    private var _killed = false
    var behavior: Behavior

    init(_ behavior: Behavior = Behavior()) { self.behavior = behavior }

    var stderrText: String { "" }
    var killed: Bool { lock.lock(); defer { lock.unlock() }; return _killed }
    var received: [[String: Any]] { lock.lock(); defer { lock.unlock() }; return _received }
    /// Waits for every write so far to be processed.
    func drain() { queue.sync {} }
    func receivedMethods() -> [String] { drain(); return received.compactMap { $0["method"] as? String } }

    func start(onData: @escaping @Sendable (Data) -> Void, onClose: @escaping @Sendable () -> Void) throws {
        self.onData = onData
        self.onClose = onClose
    }

    func write(_ data: Data) throws {
        lock.lock(); let dead = _killed; lock.unlock()
        if dead { throw LSPClientError.notRunning }
        queue.async { [self] in
            guard let bodies = try? decoder.append(data) else { return }
            for body in bodies {
                guard let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else { continue }
                lock.lock(); _received.append(json); lock.unlock()
                respond(to: json)
            }
        }
    }

    func kill() {
        lock.lock()
        let already = _killed
        _killed = true
        lock.unlock()
        if !already { onClose?() }
    }

    /// Pushes raw bytes at the client, bypassing framing (hostile server).
    func inject(_ data: Data) { queue.async { self.onData?(data) } }

    private func send(_ object: [String: Any]) {
        guard let frame = try? LSPFraming.encode(object: object) else { return }
        onData?(frame)
    }

    private func reply(_ id: Any, _ result: Any?) {
        send(["jsonrpc": "2.0", "id": id, "result": result ?? NSNull()])
    }

    private func respond(to json: [String: Any]) {
        let method = json["method"] as? String ?? ""
        let id = json["id"]
        if let id, json["method"] == nil { _ = id; return }   // a response to our own server request
        if behavior.silentMethods.contains(method) { return }
        switch method {
        case "initialize":
            reply(id as Any, ["capabilities": [String: Any](), "serverInfo": ["name": behavior.serverName]])
        case "initialized":
            if let request = behavior.serverRequestAfterInit { send(request) }
        case "textDocument/didOpen", "textDocument/didChange":
            guard behavior.publishes, let params = json["params"] as? [String: Any],
                  let document = params["textDocument"] as? [String: Any],
                  let uri = document["uri"] as? String else { return }
            let version = (document["version"] as? Int ?? 0) + behavior.versionOffset
            let text = (document["text"] as? String)
                ?? ((params["contentChanges"] as? [[String: Any]])?.first?["text"] as? String) ?? ""
            send(["jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                  "params": ["uri": uri, "version": version, "diagnostics": behavior.diagnostics(uri, text)] as [String: Any]])
        case "textDocument/documentSymbol": reply(id as Any, behavior.documentSymbols)
        case "workspace/symbol": reply(id as Any, behavior.workspaceSymbols)
        case "shutdown": reply(id as Any, nil)
        case "exit": kill()
        default:
            if let id { send(["jsonrpc": "2.0", "id": id, "error": ["code": -32601, "message": "nope"] as [String: Any]]) }
        }
    }

    static func error(line: Int, character: Int = 0, message: String, severity: Int = 1) -> [String: Any] {
        ["range": ["start": ["line": line, "character": character], "end": ["line": line, "character": character + 3]],
         "severity": severity, "message": message, "source": "fake", "code": 7]
    }
}

/// A temporary project folder with a few source files.
final class LSPTestProject {
    let root: URL
    init(files: [String: String] = ["Sources/A.swift": "let x: Int = \"no\"\n"]) throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("lsp-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        root = PathContainment.canonical(base)
        for (path, text) in files {
            let url = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try text.write(to: url, atomically: true, encoding: .utf8)
        }
    }
    deinit { try? FileManager.default.removeItem(at: root) }
}

extension LSPServerConfig {
    static func fake(id: String = "fake", extensions: [String] = ["swift"], enabled: Bool = true) -> LSPServerConfig {
        LSPServerConfig(id: id, name: "fake-ls", extensions: extensions, languageID: extensions.first ?? "x",
                        command: "/usr/bin/true", args: [], enabled: enabled, isDiscovered: false)
    }
}
