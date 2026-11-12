import Foundation
import TurboSpark

/// Why the Embeddings tab's Attach is unavailable, stated next to it.
enum EmbeddingAttachBlock: Equatable {
    case serverStopped
    case noSelection
    case busy
    case alreadyAttached
    /// The probe refused the family; the reason comes from Rust.
    case refused(String)
}

extension AppModel {
    /// Attached ids that serve `/v1/embeddings`.
    var serverEmbeddingModelIDs: [String] { serverInfo?.embeddingModels ?? [] }

    /// Attached ids that are not embedding models: what the Text tab lists.
    var serverTextModelIDs: [String] {
        let embedding = Set(serverEmbeddingModelIDs)
        return (serverInfo?.models ?? []).filter { !embedding.contains($0) }
    }

    nonisolated static func embeddingAttachBlock(
        serverRunning: Bool, selection: String, probe: EmbeddingProbe?, busy: Bool,
        attached: [String]
    ) -> EmbeddingAttachBlock? {
        let key = selection.trimmingCharacters(in: .whitespacesAndNewlines)
        guard serverRunning else { return .serverStopped }
        guard !key.isEmpty else { return .noSelection }
        if busy { return .busy }
        // The server keys an embedding model on the attach argument.
        if attached.contains(key) { return .alreadyAttached }
        if let probe, !probe.runnable { return .refused(probe.refusedBecause ?? "") }
        return nil
    }

    /// Attaches an embedding model (pinned alias or directory) to the
    /// running server. The blocking C call runs off the main actor; polling
    /// and Stop wait for it (see `serverEmbeddingAttachInFlight`).
    func attachEmbeddingModelToServer(_ input: String) {
        let arg = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let server, !arg.isEmpty, !serverBusy else { return }
        serverBusy = true
        serverEmbeddingAttachInFlight = true
        Task {
            let result = await Task.detached {
                Result(catching: { try server.attachEmbeddingModel(arg) })
            }.value
            serverEmbeddingAttachInFlight = false
            serverBusy = false
            if self.server === server {
                refreshServerInfo()
                switch result {
                case .success(let id):
                    showToast(AppStrings.localized("Serving \(id)"), style: .success)
                case .failure(let error):
                    let msg = AppStrings.localized(
                        "Could not serve \(arg): \(error.localizedDescription)")
                    self.error = msg
                    showToast(msg, style: .error)
                }
            }
            if serverStopAfterEmbeddingAttach {
                serverStopAfterEmbeddingAttach = false
                stopServer()
            }
        }
    }
}
