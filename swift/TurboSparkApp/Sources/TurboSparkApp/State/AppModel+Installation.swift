import Foundation
import TurboSpark

extension AppModel {
    public func installModel(alias: String) {
        guard !isInstallingModel, !generating else { return }
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
                let msg = "Installation failed: \(error.localizedDescription)"
                self.error = msg
                self.showToast(msg, style: .error, duration: 5.0)
            }
            self.isInstallingModel = false
            self.installTask = nil
        }
    }

    public func installRepo(repo: String, alias: String, file: String? = nil, sidecarRepo: String? = nil) {
        guard !isInstallingModel, !generating else { return }
        isInstallingModel = true
        installStageText = "Preparing repository download..."
        installProgressFraction = nil
        installDownloadedBytes = nil
        installTotalBytes = nil
        error = nil
        showToast("Starting download for '\(alias)' from Hugging Face...", style: .info)

        installTask = Task {
            do {
                var maxBytes: UInt64 = 0
                for try await event in TurboSparkCatalog.install(repo: repo, alias: alias, file: file, sidecarRepo: sidecarRepo) {
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
                let msg = "Installation failed: \(error.localizedDescription)"
                self.error = msg
                self.showToast(msg, style: .error, duration: 5.0)
            }
            self.isInstallingModel = false
            self.installTask = nil
        }
    }

    public func cancelInstall() {
        installTask?.cancel()
        installTask = nil
        isInstallingModel = false
        installStageText = "Cancelled"
        showToast("Installation cancelled", style: .warning)
    }


    public func discardModelDownload() {
        cancelInstall()
        refreshModels()
    }

    public func recheckModelAtCurrentLocation() {
        refreshModels()
    }
}
