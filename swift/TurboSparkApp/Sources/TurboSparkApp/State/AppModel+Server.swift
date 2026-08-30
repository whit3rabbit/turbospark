import Foundation
import TurboSpark

extension AppModel {
    /// Starts an in-process HTTP server sharing the currently open
    /// `session`'s engine. No-op if no model is loaded or one is already
    /// running.
    ///
    /// Requests an OS-assigned port (`ServerOptions.port == 0`) rather than
    /// a fixed one: this is a local, single-machine feature, and the bound
    /// port is read back from `TurboSparkServer.info()` for display once the
    /// call returns.
    public func startServer() {
        guard let session, server == nil, !serverBusy else { return }
        serverBusy = true
        Task {
            defer { serverBusy = false }
            do {
                let key = serverAPIKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
                let options = ServerOptions(port: 0, apiKey: key.isEmpty ? nil : key)
                let started = try await session.startServer(options: options)
                self.server = started
                let info = try started.info()
                showToast("Server listening on 127.0.0.1:\(info.port)", style: .success)
            } catch {
                let msg = "Failed to start server: \(error.localizedDescription)"
                self.error = msg
                showToast(msg, style: .error)
            }
        }
    }

    /// Stops the running server, if any. Safe to call when none is running.
    public func stopServer() {
        guard let server else { return }
        server.stop()
        self.server = nil
        showToast("Server stopped", style: .info)
    }

    /// The bound port of a running server, or `nil` when none is running or
    /// its info could not be read (e.g. a race with it stopping).
    public var serverPort: Int? {
        guard let server, let info = try? server.info() else { return nil }
        return Int(info.port)
    }
}
