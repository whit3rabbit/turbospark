import Foundation
import MLX
import TurboSpark

/// Adopting and deleting installed audio models. Text and image models already
/// have both controls; without them an audio install made outside the app was
/// invisible and a bad one could not be removed.
extension AudioWorkspaceController {
    /// The unadopted install matching the selected profile, if any.
    var selectedAdoptable: AudioLegacyInstall? {
        guard let id = recipe.modelID else { return nil }
        return adoptable(id)
    }

    func adoptable(_ alias: String) -> AudioLegacyInstall? { needsAdoption.first { $0.identity.alias == alias } }

    /// Whether the selected model is a managed install this app can delete.
    var canDeleteSelectedModel: Bool {
        guard let profile = selectedProfile else { return false }
        return canDelete(profile)
    }

    /// Whether `profile` has a managed install, adopted or not, that can be
    /// deleted now.
    func canDelete(_ profile: AudioProfile) -> Bool {
        (installedPaths[profile.identity.alias] != nil || adoptable(profile.identity.alias) != nil) && !isBusy && !isInstalling
            && !isManagingInstall && !hasRecordingActivity
    }

    /// What a move needs to know about an installed audio model, or nil when it
    /// cannot be moved: an install with no receipt (it is adopted first; the engine
    /// refuses to move it), a checkpoint folder the user chose, or nothing installed.
    func moveSubject(for profile: AudioProfile) -> MoveSubject? {
        let alias = profile.identity.alias
        guard installedPaths[alias] != nil, let record = installedRecords.first(where: { $0.alias == alias }) else { return nil }
        return MoveSubject.make(audio: record)
    }

    func adoptSelectedModel() {
        if let legacy = selectedAdoptable { adopt(legacy) }
    }

    func adopt(_ legacy: AudioLegacyInstall) {
        guard !isManagingInstall, !isBusy else { return }
        isManagingInstall = true; error = nil
        let token = epoch
        Task { [weak self] in
            let result = await Task.detached { Result { try AudioCatalog.adopt(legacy.identity) } }.value
            guard let self else { return }
            defer { self.isManagingInstall = false }
            guard self.valid(token) else { return }
            switch result {
            case .success(let record):
                self.installedPaths[record.alias] = record.path
                self.installedRecords.removeAll { $0.alias == record.alias }
                self.installedRecords.append(record)
                self.status = AppStrings.localized("Audio model installed")
                self.refreshModels()
                self.onInstalledModelsChanged?(record.alias)
            case .failure(let failure): self.error = failure.localizedDescription
            }
        }
    }

    /// Deletes the selected model's managed install. Releases the resident
    /// session first: deleting the weights out from under a mapped model is
    /// the kind of failure that shows up much later as a crash.
    func deleteSelectedModel() {
        if let profile = selectedProfile { deleteModel(profile) }
    }

    func deleteModel(_ profile: AudioProfile) {
        guard canDelete(profile) else { return }
        // Refused before anything is released: the resident session stays up when
        // the delete does not happen.
        if let reason = deleteBlockedReason?(profile.identity.alias) {
            error = reason
            return
        }
        isManagingInstall = true; error = nil
        idleTask?.cancel()
        let token = epoch
        // An unadopted install has no receipt, so it takes the legacy path.
        let legacy = adoptable(profile.identity.alias) != nil
        let old = session; let oldAccess = modelAccessURL
        session = nil; sessionKey = nil; modelAccessURL = nil
        let oldQwen = qwenModel
        qwenModel = nil
        fluidTranscriber = nil
        loadedModelAlias = nil
        Task { [weak self] in
            let result = await Task.detached { () -> Result<Void, Error> in
                old?.close()
                if oldQwen != nil {
                    MLX.Memory.clearCache()
                }
                return Result {
                    if legacy { try AudioCatalog.deleteLegacy(profile.identity) }
                    else { try AudioCatalog.delete(profile.identity) }
                }
            }.value
            oldAccess?.stopAccessingSecurityScopedResource()
            guard let self else { return }
            defer { self.isManagingInstall = false }
            guard self.valid(token) else { return }
            switch result {
            case .success:
                self.installedPaths[profile.identity.alias] = nil
                self.installedRecords.removeAll { $0.alias == profile.identity.alias }
                self.needsAdoption.removeAll { $0.identity.alias == profile.identity.alias }
                self.forgetChoices(of: profile.identity.alias)
                self.flushPreferences()
                self.status = AppStrings.localized("Model deleted")
                self.refreshModels()
                self.onInstalledModelsChanged?(profile.identity.alias)
            case .failure(let failure): self.error = failure.localizedDescription
            }
        }
    }

    /// Forgets a checkpoint folder chosen for `alias`. The folder stays where it is.
    func removeLocalModel(_ alias: String) {
        guard preferences.localModels[alias] != nil, !isBusy, !hasRecordingActivity else { return }
        if loadedModelAlias == alias { unloadModel() }
        preferences.localModels.removeValue(forKey: alias)
        forgetChoices(of: alias)
        flushPreferences()
        refreshModels()
        onInstalledModelsChanged?(alias)
    }

    /// Moves every choice that named `alias` (the recipe's model, the per-task
    /// remembered one, a Speech-to-Text mode's) to an installed model, or drops it.
    /// A model a checkpoint folder still serves keeps its choices. Without this the
    /// next `flushPreferences` saved the deleted id again, and the menu kept it.
    func forgetChoices(of alias: String) {
        reassignSpeechToTextModels(afterDeleting: alias)
        guard installedPaths[alias] == nil, preferences.localModels[alias] == nil else { return }
        preferences.selectedModels = preferences.selectedModels.filter { $0.value != alias }
        if recipe.modelID == alias { recipe.modelID = nil }
        restoreModelSelection()
        normalizeModelOptions()
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
