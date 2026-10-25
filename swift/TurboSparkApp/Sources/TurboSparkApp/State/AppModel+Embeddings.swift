import Foundation
import TurboSpark

extension AppModel {
    /// Reloads the pinned embedding profiles and their install state. A
    /// catalog error leaves the list empty: the embedding fields still take
    /// any path or alias by hand.
    func refreshEmbeddingProfiles() {
        embeddingProfiles = (try? EmbeddingCatalog.profiles()) ?? []
    }

    /// The download request for `profile`, at the pin the user is shown.
    static func embeddingDownloadRequest(_ profile: EmbeddingProfile) -> ModelDownload.Request {
        .embedding(alias: profile.alias, revision: profile.revision)
    }

    @discardableResult
    func enqueueEmbeddingDownload(_ profile: EmbeddingProfile) -> Bool {
        let request = Self.embeddingDownloadRequest(profile)
        guard profile.canRun, Self.isSafeInstallAlias(profile.alias),
              !isModelAlreadyInstalled(request) else { return false }
        return enqueueModelDownload(request)
    }

    /// Probes a typed embedding model path or alias off the main actor and
    /// stores the verdict under the trimmed input. Rust decides the family;
    /// the app only shows the result.
    func refreshEmbeddingProbe(_ input: String) async {
        let key = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { return }
        let probe = await Task.detached { try? EmbeddingCatalog.probe(key) }.value
        if let probe { embeddingProbes[key] = probe }
    }

    /// The native transfer owns the shared install slot until its blocking
    /// call exits. Cancel goes through the shared install controls, as for
    /// audio profiles. A nil `revision` (a row saved before revisions were
    /// recorded) installs the current pin.
    func startEmbeddingModelInstall(alias: String, revision: String?) {
        let revision = revision ?? embeddingProfiles.first { $0.alias == alias }?.revision ?? ""
        Self.modelInstallOwner = self
        installingAlias = alias
        installEpoch += 1
        let myEpoch = installEpoch
        let downloadID = activeModelDownloadID
        isInstallingModel = true
        installStageText = AppStrings.localized("Downloading embedding model")
        installProgressFraction = nil
        installDownloadedBytes = nil
        installTotalBytes = nil
        installETAText = nil
        let startedAt = Date()

        installTask = Task { [self] in
            let cancellationMonitor = Task { @MainActor in
                while !Task.isCancelled {
                    guard self.installEpoch == myEpoch else { return }
                    if self.isCancellingModelInstall || self.modelDownloadsShuttingDown {
                        TurboSparkCatalog.cancelInstall()
                    }
                    do { try await Task.sleep(for: .milliseconds(100)) } catch { return }
                }
            }
            defer {
                cancellationMonitor.cancel()
                if Self.modelInstallOwner === self { Self.modelInstallOwner = nil }
                if self.installEpoch == myEpoch {
                    self.installTask = nil
                    self.isInstallingModel = false
                    self.installingAlias = nil
                    self.installStageText = nil
                    self.installProgressFraction = nil
                    self.installDownloadedBytes = nil
                    self.installTotalBytes = nil
                    self.installETAText = nil
                }
                self.startNextModelDownloadIfPossible()
            }
            do {
                _ = try await Task.detached {
                    try EmbeddingCatalog.install(alias: alias, revision: revision) { [weak self] progress in
                        Task { @MainActor in
                            guard let self, self.installEpoch == myEpoch,
                                  self.activeModelDownloadID == downloadID,
                                  !self.modelDownloadsShuttingDown else { return }
                            if self.isCancellingModelInstall {
                                TurboSparkCatalog.cancelInstall()
                                return
                            }
                            self.recordModelDownloadProgress(
                                done: progress.completedBytes, total: progress.totalBytes)
                            self.installETAText = self.isInstallPaused ? nil : Self.installETA(
                                done: self.installDownloadedBytes ?? 0, total: progress.totalBytes,
                                elapsed: Date().timeIntervalSince(startedAt) - self.installPausedDuration)
                        }
                    }
                }.value
                guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
                self.setModelDownloadStatus(.completed)
                self.refreshEmbeddingProfiles()
                // A probe of the alias said "not installed" until now.
                self.embeddingProbes = [:]
                self.showToast(AppStrings.localized("Embedding model installed"), style: .success)
            } catch {
                guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
                if !self.isCancellingModelInstall {
                    self.setModelDownloadStatus(.failed, failure: error.localizedDescription)
                    self.showToast(error.localizedDescription, style: .error)
                }
            }
            guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
            self.finishModelInstallCancellation()
        }
    }
}
