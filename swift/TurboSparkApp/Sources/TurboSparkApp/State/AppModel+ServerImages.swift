import Foundation
import TurboSpark

extension AppModel {
    /// One resident pipeline is shared by the Images screen and the API worker.
    /// Call only while holding ImageJobCoordinator's permit.
    func sharedImageSession(for model: ImageInstalledModel) throws -> any ImageGenerationSession {
        if imageSessionPath != model.path || imageSession == nil {
            imageSession?.cancel()
            imageSession = MLXImageGenerationSession(model: model)
            imageSessionPath = model.path
        }
        guard let imageSession else {
            throw TurboSparkError(code: .open, message: "image session did not open")
        }
        return imageSession
    }

    public func attachImageModelToServer(_ model: ImageInstalledModel) {
        guard let server, Self.supportsMLXImageModel(modelID: model.modelID) else { return }
        do {
            try server.attachImageModel(id: model.alias)
            serverImageAttachedModel = model
            serverImageProgress = "Ready"
            refreshServerInfo()
        } catch {
            showToast("Could not serve image model: \(error.localizedDescription)", style: .error)
        }
    }

    public func detachImageModelFromServer() {
        guard let server else { return }
        cancelServerImageJobs()
        do {
            try server.detachImageModel()
            serverImageAttachedModel = nil
            serverImageProgress = "Idle"
            refreshServerInfo()
        } catch {
            showToast("Could not detach image model: \(error.localizedDescription)", style: .error)
        }
    }

    func cancelServerImageJobs() {
        for task in serverImageTasks.values { task.cancel() }
        serverImageTasks.removeAll()
    }

    func handleServerImageEvents(_ server: TurboSparkServer) {
        for event in server.pollImageEvents() {
            switch event {
            case let .cancel(id):
                serverImageTasks.removeValue(forKey: id)?.cancel()
            case let .start(id, request):
                guard let model = serverImageAttachedModel, model.alias == request.model else {
                    try? server.completeImageRequest(id: id, png: nil, error: "image model was detached")
                    continue
                }
                serverImageTasks[id] = Task { [weak self] in
                    guard let self else { return }
                    await self.runServerImageRequest(id: id, request: request, model: model, server: server)
                }
            }
        }
    }

    private func runServerImageRequest(
        id: UInt64, request: ServerImageRequest,
        model: ImageInstalledModel, server: TurboSparkServer
    ) async {
        defer { serverImageTasks[id] = nil }
        let coordinator = ImageJobCoordinator.shared
        let acquired = await coordinator.acquire()
        guard acquired else { return }
        defer { Task { await coordinator.release() } }
        do {
            try Task.checkCancellation()
            let session = try sharedImageSession(for: model)
            let options = ImageGenerateOptions(
                prompt: request.prompt, seed: request.seed,
                width: request.width, height: request.height,
                steps: model.schedulerSteps)
            let png = try await withTaskCancellationHandler {
                var output: Data?
                for try await event in session.generate(options) {
                    try Task.checkCancellation()
                    switch event {
                    case let .stage(name, completed, total):
                        serverImageProgress = "\(name) \(completed)/\(total)"
                    case let .finished(result): output = result.png
                    case .cancelled: throw CancellationError()
                    }
                }
                return output
            } onCancel: {
                session.cancel()
            }
            guard let png else { throw TurboSparkError(code: .generate, message: "No image returned") }
            try server.completeImageRequest(id: id, png: png)
            serverImageProgress = "Ready"
        } catch is CancellationError {
            serverImageProgress = "Cancelled"
        } catch {
            try? server.completeImageRequest(id: id, png: nil, error: error.localizedDescription)
            serverImageProgress = "Failed: \(error.localizedDescription)"
        }
    }
}
