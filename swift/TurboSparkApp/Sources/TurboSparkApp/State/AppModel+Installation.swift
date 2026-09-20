import Foundation
import TurboSpark

extension AppModel {
    /// The remaining-time string for a running install, or nil when there is
    /// nothing honest to say yet (state#102).
    ///
    /// **`installETAText` HAD NO WRITER**, only a declaration, a reset to nil
    /// on every install path, and a view reading it -- so the row it feeds
    /// never appeared and the field read as a feature that was there.
    /// `swift/CLAUDE.md` Gotcha 36's tell applied to a published property
    /// rather than to a setting: grep for the WRITER, not for the field.
    ///
    /// Nil below a floor of elapsed time and downloaded bytes, because a rate
    /// measured over the first instant of a multi-gigabyte stream produces a
    /// number that is wrong by an order of magnitude and then visibly
    /// corrects itself, which is worse than no estimate. A pure static so it
    /// can be asserted without a download.
    static func installETA(done: UInt64, total: UInt64, elapsed: TimeInterval) -> String? {
        guard total > 0, done > 0, done < total, elapsed >= 2.0 else { return nil }
        let rate = Double(done) / elapsed
        guard rate > 0 else { return nil }
        let remaining = Double(total - done) / rate
        guard remaining.isFinite, remaining >= 1 else { return nil }
        let seconds = Int(remaining.rounded())
        if seconds < 60 { return "about \(seconds)s remaining" }
        if seconds < 3600 { return "about \(seconds / 60)m remaining" }
        let hours = seconds / 3600
        let minutes = (seconds % 3600) / 60
        return "about \(hours)h \(minutes)m remaining"
    }

    /// Whether this machine will refuse to install `alias`, and why.
    ///
    /// **THE GATE IS HERE AS WELL AS ON THE BUTTON**, and that is the whole
    /// point. A conservative rule spelled at three sites and a permissive one
    /// at the site users reach is how every project in this app came to run
    /// shell commands unprompted (swift Gotcha 28). A view may disable its
    /// button from this; nothing may install past it.
    ///
    /// `nil` when the catalog has no row for the alias: an unsized candidate
    /// is not a refused one, and refusing what nothing has measured would
    /// block every side-loaded install.
    public func installBlockReason(alias: String) -> String? {
        guard let row = fit(for: alias) else { return nil }
        let decision = ModelInstallGate.decide(
            probeRunnable: nil,
            refusedBecause: nil,
            verdict: row.verdict,
            installBytes: row.installBytes,
            freeDiskBytes: ModelInstallGate.freeSpace(at: AppStorageRoot.directory))
        return decision.isBlocked ? decision.reason : nil
    }

    /// Initiates a background download and build of a model catalog alias.
    public func installModel(alias: String) {
        // A refusal is reported rather than swallowed: a Download button that
        // does nothing reads as a broken button.
        if let why = installBlockReason(alias: alias) {
            showToast(
                "'\(alias)' will not run on this machine. \(why)",
                style: .warning, duration: 8.0)
            return
        }
        enqueueModelDownload(.catalog(alias: alias))
    }

    /// Test seam for a deterministic install stream. Production downloads
    /// enter through the shared queue above.
    func installModel(
        alias: String,
        stream: @escaping (String) -> AsyncThrowingStream<InstallEvent, Error>
    ) {
        let request = ModelDownload.Request.catalog(alias: alias)
        guard canQueueModelDownload(request), Self.modelInstallOwner == nil,
              !hasActiveModelDownload else { return }
        beginModelDownload(.catalog(alias: alias))
        startCatalogModelInstall(alias: alias, stream: stream)
    }

    func startCatalogModelInstall(
        alias: String,
        stream: @escaping (String) -> AsyncThrowingStream<InstallEvent, Error>
    ) {
        Self.modelInstallOwner = self
        installingAlias = alias
        installEpoch += 1
        let myEpoch = installEpoch
        isInstallingModel = true
        installStageText = "Preparing install..."
        installProgressFraction = nil
        installDownloadedBytes = nil
        installTotalBytes = nil
        installETAText = nil
        error = nil

        let startedAt = Date()
        installTask = Task {
            defer {
                if Self.modelInstallOwner === self { Self.modelInstallOwner = nil }
                self.installTask = nil
                self.startNextModelDownloadIfPossible()
            }
            do {
                var maxBytes: UInt64 = 0
                for try await event in stream(alias) {
                    guard self.installEpoch == myEpoch else { return }
                    if self.modelDownloadsShuttingDown {
                        TurboSparkCatalog.cancelInstall()
                        continue
                    }
                    if self.isCancellingModelInstall {
                        // Keep consuming until the native writer exits. This also
                        // catches Cancel pressed before its worker registered.
                        if case .finished = event {
                            self.setModelDownloadStatus(.completed)
                            self.refreshModels()
                        } else {
                            TurboSparkCatalog.cancelInstall()
                        }
                        continue
                    }
                    switch event {
                    case .stage(let text):
                        self.recordModelDownloadStage(text)
                        self.installStageText = text
                    case .bytes(let done, let total):
                        maxBytes = max(maxBytes, done)
                        self.recordModelDownloadProgress(done: maxBytes, total: total)
                        self.installETAText = self.isInstallPaused ? nil : Self.installETA(
                            done: maxBytes, total: total,
                            elapsed: Date().timeIntervalSince(startedAt) - self.installPausedDuration)
                    case .finished(let model):
                        guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
                        self.setModelDownloadStatus(.loading)
                        self.installStageText = "Installation complete!"
                        self.refreshModels()
                        self.selected = model
                        // **THE TOAST FOLLOWS THE OPEN, NOT THE INSTALL**
                        // (state#83). It claimed "installed and loaded"
                        // BEFORE awaiting an open that returns silently at
                        // its own `guard !generating, !opening` -- so an
                        // install that finished while a turn was running
                        // reported a model as loaded that was not, and the
                        // Chat pane still showed nothing.
                        await self.open(model)
                        guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
                        self.setModelDownloadStatus(.completed)
                        if self.session != nil, self.selected?.path == model.path {
                            self.showToast(
                                "Successfully installed and loaded '\(model.alias)'",
                                style: .success, duration: 4.0)
                        } else {
                            self.showToast(
                                "Installed '\(model.alias)'. Load it from the Installed pane.",
                                style: .info, duration: 5.0)
                        }
                    }
                }
            } catch is CancellationError {
                // A stale consumer must not clear another install's progress.
                guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
                self.setModelDownloadStatus(.cancelled)
                self.installStageText = "Stopped watching install"
            } catch {
                guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
                if !self.isCancellingModelInstall {
                    self.setModelDownloadStatus(.failed, failure: error.localizedDescription)
                    self.error = error.localizedDescription
                    self.installStageText = nil
                    self.showToast("Installation failed: \(error.localizedDescription)", style: .error, duration: 5.0)
                }
            }
            guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
            self.finishModelInstallCancellation()
            self.installingAlias = nil
            self.isInstallingModel = false
            // Cleared here rather than left showing the last stage forever
            // (state#52): every other field of the progress row is reset and
            // this one is what the row actually READS, so an install that
            // finished left "Installation complete!" under an idle button.
            self.installStageText = nil
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
        let trimmedRepo = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedAlias = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedRepo.isEmpty, !trimmedAlias.isEmpty else { return }
        enqueueModelDownload(.repository(
            repo: trimmedRepo, alias: trimmedAlias, file: file, sidecarRepo: sidecarRepo))
    }

    func startRepositoryModelInstall(
        repo: String,
        alias: String,
        file: String? = nil,
        sidecarRepo: String? = nil
    ) {
        Self.modelInstallOwner = self
        installingAlias = alias

        installEpoch += 1
        let myEpoch = installEpoch
        isInstallingModel = true
        installStageText = "Connecting to Hugging Face..."
        installProgressFraction = nil
        installDownloadedBytes = nil
        installTotalBytes = nil
        installETAText = nil
        error = nil

        let startedAt = Date()
        installTask = Task {
            defer {
                if Self.modelInstallOwner === self { Self.modelInstallOwner = nil }
                self.installTask = nil
                self.startNextModelDownloadIfPossible()
            }
            do {
                var maxBytes: UInt64 = 0
                for try await event in TurboSparkCatalog.install(
                    repo: repo,
                    alias: alias,
                    file: file,
                    sidecarRepo: sidecarRepo
                ) {
                    guard self.installEpoch == myEpoch else { return }
                    if self.modelDownloadsShuttingDown {
                        TurboSparkCatalog.cancelInstall()
                        continue
                    }
                    if self.isCancellingModelInstall {
                        // Keep consuming until the native writer exits. This also
                        // catches Cancel pressed before its worker registered.
                        if case .finished = event {
                            self.setModelDownloadStatus(.completed)
                            self.refreshModels()
                        } else {
                            TurboSparkCatalog.cancelInstall()
                        }
                        continue
                    }
                    switch event {
                    case .stage(let text):
                        self.recordModelDownloadStage(text)
                        self.installStageText = text
                    case .bytes(let done, let total):
                        maxBytes = max(maxBytes, done)
                        self.recordModelDownloadProgress(done: maxBytes, total: total)
                        self.installETAText = self.isInstallPaused ? nil : Self.installETA(
                            done: maxBytes, total: total,
                            elapsed: Date().timeIntervalSince(startedAt) - self.installPausedDuration)
                    case .finished(let model):
                        guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
                        self.setModelDownloadStatus(.loading)
                        self.installStageText = "Installation complete!"
                        self.refreshModels()
                        self.selected = model
                        // **THE TOAST FOLLOWS THE OPEN, NOT THE INSTALL**
                        // (state#83). It claimed "installed and loaded"
                        // BEFORE awaiting an open that returns silently at
                        // its own `guard !generating, !opening` -- so an
                        // install that finished while a turn was running
                        // reported a model as loaded that was not, and the
                        // Chat pane still showed nothing.
                        await self.open(model)
                        guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
                        self.setModelDownloadStatus(.completed)
                        if self.session != nil, self.selected?.path == model.path {
                            self.showToast(
                                "Successfully installed and loaded '\(model.alias)'",
                                style: .success, duration: 4.0)
                        } else {
                            self.showToast(
                                "Installed '\(model.alias)'. Load it from the Installed pane.",
                                style: .info, duration: 5.0)
                        }
                    }
                }
            } catch is CancellationError {
                // A stale consumer must not clear another install's progress.
                guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
                self.setModelDownloadStatus(.cancelled)
                self.installStageText = "Stopped watching install"
            } catch {
                guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
                if !self.isCancellingModelInstall {
                    let desc = error.localizedDescription
                    self.setModelDownloadStatus(.failed, failure: desc)
                    if desc.contains("401") || desc.contains("403") || desc.contains("gated") || desc.localizedCaseInsensitiveContains("unauthorized") {
                        self.showToast("Installation failed: Authentication required. Check your Hugging Face API token in Settings.", style: .error, duration: 6.0)
                    } else {
                        self.showToast("Installation failed: \(desc)", style: .error, duration: 5.0)
                    }
                }
            }
            guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
            self.finishModelInstallCancellation()
            self.installingAlias = nil
            self.isInstallingModel = false
            // Cleared here rather than left showing the last stage forever
            // (state#52): every other field of the progress row is reset and
            // this one is what the row actually READS, so an install that
            // finished left "Installation complete!" under an idle button.
            self.installStageText = nil
            self.installProgressFraction = nil
            self.installDownloadedBytes = nil
            self.installTotalBytes = nil
            self.installETAText = nil
            self.installTask = nil
        }
    }

    /// Stop the native writer before releasing its install slot. Cancelling
    /// only the Swift consumer would permit Retry to race the old writer.
    public func cancelInstall() {
        guard isInstallingModel, !isCancellingModelInstall else { return }
        isCancellingModelInstall = true
        setModelDownloadStatus(.cancelling)
        installETAText = nil
        TurboSparkCatalog.cancelInstall()
    }

    public func cancelActiveModelDownload() {
        if isInstallingImageModel {
            cancelImageInstall()
        } else {
            cancelInstall()
        }
    }

    /// Called only after the native stream has terminated. A finished event
    /// may win a race with Cancel; keep that completed row instead of claiming
    /// a valid install was discarded.
    func finishModelInstallCancellation() {
        guard isCancellingModelInstall else { return }
        if activeModelDownloadID != nil {
            setModelDownloadStatus(.cancelled)
        }
        isCancellingModelInstall = false
        isInstallPaused = false
    }

    /// A busy native writer does not block another model from joining the
    /// queue. Only a duplicate live request or generation blocks the button.
    public func canInstall(alias: String) -> Bool {
        canQueueModelDownload(.catalog(alias: alias))
    }

    public func canInstallImageModel(alias: String) -> Bool {
        canQueueModelDownload(.image(alias: alias))
    }
}
