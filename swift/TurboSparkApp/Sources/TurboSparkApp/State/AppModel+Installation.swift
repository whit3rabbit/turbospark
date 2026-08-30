import Foundation
import TurboSpark

extension AppModel {
    /// Initiates a background download and build of a model catalog alias.
    public func installModel(alias: String) {
        guard !isInstallingModel, !generating else { return }
        installEpoch += 1
        let myEpoch = installEpoch
        isInstallingModel = true
        installStageText = "Preparing install..."
        installProgressFraction = nil
        installDownloadedBytes = nil
        installTotalBytes = nil
        installETAText = nil
        error = nil
        showToast("Starting download for '\(alias)'...", style: .info)

        installTask = Task {
            do {
                var maxBytes: UInt64 = 0
                for try await event in TurboSparkCatalog.install(alias) {
                    switch event {
                    case .stage(let text):
                        self.installStageText = text
                    case .bytes(let done, let total):
                        maxBytes = max(maxBytes, done)
                        self.installDownloadedBytes = maxBytes
                        self.installTotalBytes = total
                        if total > 0 {
                            self.installProgressFraction = min(Double(maxBytes) / Double(total), 1.0)
                        }
                    case .finished(let model):
                        self.installStageText = "Installation complete!"
                        self.refreshModels()
                        self.selected = model
                        self.showToast("Successfully installed and loaded '\(model.alias)'", style: .success, duration: 4.0)
                        await self.open(model)
                    }
                }
            } catch is CancellationError {
                self.installStageText = "Installation cancelled"
                self.showToast("Installation cancelled", style: .warning)
            } catch {
                self.error = error.localizedDescription
                self.installStageText = nil
                self.showToast("Installation failed: \(error.localizedDescription)", style: .error, duration: 5.0)
            }
            guard self.installEpoch == myEpoch else { return }
            self.isInstallingModel = false
            self.installProgressFraction = nil
            self.installDownloadedBytes = nil
            self.installTotalBytes = nil
            self.installETAText = nil
            self.installTask = nil
        }
    }

    /// Initiates download and packaging of an arbitrary Hugging Face GGUF repository.
    public func installRepo(
        repo: String,
        alias: String,
        file: String? = nil,
        sidecarRepo: String? = nil
    ) {
        guard !isInstallingModel, !generating else { return }
        let trimmedRepo = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedAlias = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedRepo.isEmpty, !trimmedAlias.isEmpty else { return }

        installEpoch += 1
        let myEpoch = installEpoch
        isInstallingModel = true
        installStageText = "Connecting to Hugging Face..."
        installProgressFraction = nil
        installDownloadedBytes = nil
        installTotalBytes = nil
        installETAText = nil
        error = nil
        showToast("Pulling '\(trimmedRepo)'...", style: .info)

        installTask = Task {
            do {
                var maxBytes: UInt64 = 0
                for try await event in TurboSparkCatalog.install(
                    repo: trimmedRepo,
                    alias: trimmedAlias,
                    file: file,
                    sidecarRepo: sidecarRepo
                ) {
                    switch event {
                    case .stage(let text):
                        self.installStageText = text
                    case .bytes(let done, let total):
                        maxBytes = max(maxBytes, done)
                        self.installDownloadedBytes = maxBytes
                        self.installTotalBytes = total
                        if total > 0 {
                            self.installProgressFraction = min(Double(maxBytes) / Double(total), 1.0)
                        }
                    case .finished(let model):
                        self.installStageText = "Installation complete!"
                        self.refreshModels()
                        self.selected = model
                        self.showToast("Successfully installed and loaded '\(model.alias)'", style: .success, duration: 4.0)
                        await self.open(model)
                    }
                }
            } catch is CancellationError {
                self.installStageText = "Installation cancelled"
                self.showToast("Installation cancelled", style: .warning)
            } catch {
                self.error = error.localizedDescription
                self.installStageText = nil
                self.showToast("Installation failed: \(error.localizedDescription)", style: .error, duration: 5.0)
            }
            guard self.installEpoch == myEpoch else { return }
            self.isInstallingModel = false
            self.installProgressFraction = nil
            self.installDownloadedBytes = nil
            self.installTotalBytes = nil
            self.installETAText = nil
            self.installTask = nil
        }
    }

    /// Cancels any currently active model installation task.
    public func cancelInstall() {
        installTask?.cancel()
        installTask = nil
        isInstallingModel = false
        installStageText = nil
        installProgressFraction = nil
        installDownloadedBytes = nil
        installTotalBytes = nil
        installETAText = nil
        // Cancellation is cooperative: the cancelled Task keeps running
        // until its next suspension point notices, so its own tail can
        // still fire after this call returns. Bumping the epoch here too
        // (on top of each new install bumping it at its own start) means
        // that stale tail is a no-op even if nothing new has started yet.
        installEpoch += 1
    }
}
