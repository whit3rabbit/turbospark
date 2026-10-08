import Foundation
import TurboSpark

/// Adopting and deleting installed audio models. Text and image models already
/// have both controls; without them an audio install made outside the app was
/// invisible and a bad one could not be removed.
extension AudioWorkspaceController {
    /// The unadopted install matching the selected profile, if any.
    var selectedAdoptable: AudioLegacyInstall? {
        guard let id = recipe.modelID else { return nil }
        return needsAdoption.first { $0.identity.alias == id }
    }

    /// Whether the selected model is a managed install this app can delete.
    var canDeleteSelectedModel: Bool {
        guard let profile = selectedProfile else { return false }
        return (installedPaths[profile.identity.alias] != nil || selectedAdoptable != nil) && !isBusy && !isInstalling
            && !isManagingInstall && !hasRecordingActivity
    }

    func adoptSelectedModel() {
        guard let legacy = selectedAdoptable, !isManagingInstall, !isBusy else { return }
        isManagingInstall = true; error = nil
        let token = epoch
        Task { [weak self] in
            let result = await Task.detached { Result { try AudioCatalog.adopt(legacy.identity) } }.value
            guard let self, self.valid(token) else { return }
            self.isManagingInstall = false
            switch result {
            case .success(let record):
                self.installedPaths[record.alias] = record.path
                self.status = String(localized: "Audio model installed", bundle: .module)
                self.refreshModels()
            case .failure(let failure): self.error = failure.localizedDescription
            }
        }
    }

    /// Deletes the selected model's managed install. Releases the resident
    /// session first: deleting the weights out from under a mapped model is
    /// the kind of failure that shows up much later as a crash.
    func deleteSelectedModel() {
        guard canDeleteSelectedModel, let profile = selectedProfile else { return }
        isManagingInstall = true; error = nil
        idleTask?.cancel()
        let token = epoch
        // An unadopted install has no receipt, so it takes the legacy path.
        let legacy = selectedAdoptable != nil
        let old = session; let oldAccess = modelAccessURL
        session = nil; sessionKey = nil; modelAccessURL = nil
        Task { [weak self] in
            let result = await Task.detached { () -> Result<Void, Error> in
                old?.close()
                return Result {
                    if legacy { try AudioCatalog.deleteLegacy(profile.identity) }
                    else { try AudioCatalog.delete(profile.identity) }
                }
            }.value
            oldAccess?.stopAccessingSecurityScopedResource()
            guard let self, self.valid(token) else { return }
            self.isManagingInstall = false
            switch result {
            case .success:
                self.installedPaths[profile.identity.alias] = nil
                self.status = String(localized: "Model deleted", bundle: .module)
                self.refreshModels()
            case .failure(let failure): self.error = failure.localizedDescription
            }
        }
    }
}

extension AudioWorkspaceController {
    /// Whether `error` means another job owns the audio device, so the caller
    /// should retry shortly rather than treat the model as broken. The engine's
    /// own code is authoritative; the message check is kept for an error that
    /// reached here wrapped by something that dropped the code.
    static func isDeviceBusy(_ error: Error) -> Bool {
        if let engine = error as? TurboSparkError, engine.code == .busy { return true }
        return error.localizedDescription.contains("audio device is busy")
    }
}
