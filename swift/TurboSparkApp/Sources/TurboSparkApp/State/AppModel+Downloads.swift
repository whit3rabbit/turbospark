import Foundation
import TurboSpark

extension AppModel {
    var hasActiveModelDownload: Bool {
        isInstallingModel || isInstallingImageModel
    }

    var canPauseModelDownload: Bool {
        // Image installs have no pause signal, so the button would do nothing.
        !modelDownloadsShuttingDown && hasActiveModelDownload && !isInstallingImageModel
            && !isInstallPaused
            && !isCancellingModelInstall && !opening
            && modelDownloads.first(where: { $0.id == activeModelDownloadID })?.status == .running
    }

    func isModelDownloadPending(_ request: ModelDownload.Request) -> Bool {
        modelDownloads.contains {
            !$0.status.isTerminal && $0.request.queueKey == request.queueKey
        }
    }

    func canQueueModelDownload(_ request: ModelDownload.Request) -> Bool {
        let hasRoom = modelDownloads.count < ModelDownloadHistoryStore.limit
            || modelDownloads.contains(where: { $0.status.isTerminal })
        return !modelDownloadsShuttingDown && !generating && hasRoom
            && !isModelDownloadPending(request)
    }

    @discardableResult
    func enqueueAudioDownload(profile: AudioProfile) -> Bool {
        guard !profile.identity.repository.isEmpty,
              Self.isSafeInstallAlias(profile.identity.alias),
              !isModelAlreadyInstalled(.audio(identity: profile.identity)) else { return false }
        return enqueueModelDownload(.audio(identity: profile.identity))
    }

    /// Adds one request to the shared FIFO. Main-actor serialization and the
    /// queue-key check make repeated button clicks idempotent.
    @discardableResult
    func enqueueModelDownload(_ request: ModelDownload.Request) -> Bool {
        guard canQueueModelDownload(request) else {
            if isModelDownloadPending(request) { isDownloadManagerExpanded = true }
            return false
        }

        // History is bounded, but a live row must never be evicted to make
        // room for an older completed row.
        if modelDownloads.count >= ModelDownloadHistoryStore.limit,
           let terminal = modelDownloads.lastIndex(where: { $0.status.isTerminal }) {
            modelDownloads.remove(at: terminal)
        }
        guard modelDownloads.count < ModelDownloadHistoryStore.limit else { return false }

        modelDownloads.insert(ModelDownload(request: request, status: .queued), at: 0)
        isDownloadManagerExpanded = true
        persistModelDownloads()
        startNextModelDownloadIfPossible()
        return true
    }

    /// Claims the oldest queued row only after the single native writer is
    /// free. Every model family enters through this dispatch point.
    func startNextModelDownloadIfPossible(
        start: ((ModelDownload.Request) -> Void)? = nil
    ) {
        guard !modelDownloadsShuttingDown, !generating,
              Self.modelInstallOwner == nil, !hasActiveModelDownload,
              let index = modelDownloads.lastIndex(where: { $0.status == .queued })
        else { return }

        let request = modelDownloads[index].request
        beginModelDownload(request)
        if let start = start ?? modelDownloadStartOverride {
            start(request)
            return
        }
        switch request {
        case .catalog(let alias):
            startCatalogModelInstall(alias: alias, stream: TurboSparkCatalog.install)
        case .repository(let repo, let alias, let file, let sidecarRepo):
            startRepositoryModelInstall(
                repo: repo, alias: alias, file: file, sidecarRepo: sidecarRepo)
        case .image(let alias):
            startImageModelInstall(alias: alias)
        case .audio(let identity):
            startAudioModelInstall(identity: identity)
        }
    }

    /// The native transfer owns the shared install slot until its blocking call exits.
    func startAudioModelInstall(
        identity: AudioProfileIdentity,
        install: @escaping @Sendable (AudioProfileIdentity, @escaping @Sendable (AudioDownloadProgress) -> Void) throws -> Void = { identity, progress in
            _ = try AudioCatalog.install(identity, progress: progress)
        }
    ) {
        Self.modelInstallOwner = self
        installingAlias = identity.alias
        installEpoch += 1
        let myEpoch = installEpoch
        let downloadID = activeModelDownloadID
        let audioEpoch = audioWorkspace.epoch
        isInstallingModel = true
        audioWorkspace.isInstalling = true
        audioWorkspace.status = String(localized: "Downloading audio model", bundle: .module)
        installStageText = audioWorkspace.status
        installProgressFraction = nil
        installDownloadedBytes = nil
        installTotalBytes = nil
        installETAText = nil
        let startedAt = Date()

        installTask = Task {
            // Cancel may arrive before the native worker registers its control handle.
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
                    if self.audioWorkspace.valid(audioEpoch) {
                        self.audioWorkspace.isInstalling = false
                        self.audioWorkspace.progress = nil
                    }
                }
                self.startNextModelDownloadIfPossible()
            }
            do {
                try await Task.detached {
                    try install(identity) { [weak self] progress in
                        Task { @MainActor in
                            guard let self, self.installEpoch == myEpoch,
                                  self.activeModelDownloadID == downloadID,
                                  !self.modelDownloadsShuttingDown,
                                  self.audioWorkspace.valid(audioEpoch) else { return }
                            if self.isCancellingModelInstall {
                                TurboSparkCatalog.cancelInstall()
                                return
                            }
                            self.recordModelDownloadProgress(done: progress.completedBytes, total: progress.totalBytes)
                            self.installETAText = self.isInstallPaused ? nil : Self.installETA(
                                done: self.installDownloadedBytes ?? 0, total: progress.totalBytes,
                                elapsed: Date().timeIntervalSince(startedAt) - self.installPausedDuration)
                            self.audioWorkspace.progress = self.installProgressFraction
                        }
                    }
                }.value
                guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown,
                      self.audioWorkspace.valid(audioEpoch) else { return }
                self.setModelDownloadStatus(.completed)
                self.audioWorkspace.refreshModels()
                self.audioWorkspace.status = String(localized: "Audio model installed", bundle: .module)
                self.showToast(String(localized: "Audio model installed", bundle: .module), style: .success)
            } catch {
                guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown,
                      self.audioWorkspace.valid(audioEpoch) else { return }
                if !self.isCancellingModelInstall {
                    self.setModelDownloadStatus(.failed, failure: error.localizedDescription)
                    self.audioWorkspace.error = error.localizedDescription
                    self.audioWorkspace.status = error.localizedDescription
                }
            }
            guard self.installEpoch == myEpoch, !self.modelDownloadsShuttingDown else { return }
            self.finishModelInstallCancellation()
        }
    }

    func beginModelDownload(_ request: ModelDownload.Request) {
        if let index = modelDownloads.firstIndex(where: {
            $0.status == .queued && $0.request.queueKey == request.queueKey
        }) {
            modelDownloads[index].status = .running
            modelDownloads[index].updatedAt = Date()
            activeModelDownloadID = modelDownloads[index].id
        } else {
            let download = ModelDownload(request: request)
            activeModelDownloadID = download.id
            modelDownloads.insert(download, at: 0)
            modelDownloads = Array(modelDownloads.prefix(ModelDownloadHistoryStore.limit))
        }
        isDownloadManagerExpanded = true
        isInstallPaused = false
        isCancellingModelInstall = false
        installPausedAt = nil
        installPausedDuration = 0
        downloadTransferMeter.reset(bytes: 0, at: ProcessInfo.processInfo.systemUptime)
        persistModelDownloads()
    }

    func setModelDownloadStatus(_ status: ModelDownload.Status, failure: String? = nil) {
        guard !modelDownloadsShuttingDown,
              let index = modelDownloads.firstIndex(where: { $0.id == activeModelDownloadID }) else { return }
        modelDownloads[index].status = status
        modelDownloads[index].failure = failure
        modelDownloads[index].updatedAt = Date()
        if status != .paused {
            isInstallPaused = false
        }
        if status.isTerminal {
            activeModelDownloadID = nil
            installPausedAt = nil
        }
        persistModelDownloads()
    }

    /// Maps native stage text onto the durable lifecycle without coupling the
    /// FFI event shape to Swift UI state. Paused and cancelling are explicit
    /// user states and must not be overwritten by a late worker callback.
    func recordModelDownloadStage(_ stage: String) {
        guard !modelDownloadsShuttingDown,
              let index = modelDownloads.firstIndex(where: { $0.id == activeModelDownloadID }),
              !modelDownloads[index].status.isTerminal,
              modelDownloads[index].status != .paused,
              modelDownloads[index].status != .cancelling
        else { return }
        let lower = stage.lowercased()
        let status: ModelDownload.Status
        // The Rust installers announce the byte transfer explicitly. Without
        // this, the first "[repack] ..." line labeled the whole transfer
        // "Packing". It has no end marker, so later repack lines still read
        // as packing.
        if lower == "[phase] transfer" {
            status = .running
        } else if lower.contains("verif") || lower.contains("validat")
            || lower.contains("manifest") || lower.contains("cleared the completed") {
            status = .verifying
        } else if lower.contains("pack") || lower.contains("repack")
            || lower.contains("convert") || lower.contains("writing") {
            status = .packing
        } else if lower.contains("download") || lower.contains("fetch")
            || lower.contains("stream") || lower.contains("connect")
            || lower.contains("preparing image source") {
            status = .running
        } else {
            return
        }
        guard modelDownloads[index].status != status else { return }
        modelDownloads[index].status = status
        modelDownloads[index].updatedAt = Date()
        persistModelDownloads()
    }

    func pauseModelDownload(signal: () -> Bool = { TurboSparkCatalog.pauseInstall() }) {
        guard canPauseModelDownload, signal() else { return }
        installPausedAt = Date()
        installETAText = nil
        isInstallPaused = true
        setModelDownloadStatus(.paused)
    }

    func resumeModelDownload(signal: () -> Bool = { TurboSparkCatalog.resumeInstall() }) {
        guard !modelDownloadsShuttingDown, hasActiveModelDownload, isInstallPaused,
              !isCancellingModelInstall, signal() else { return }
        if let pausedAt = installPausedAt {
            installPausedDuration += Date().timeIntervalSince(pausedAt)
        }
        installPausedAt = nil
        downloadTransferMeter.reset(
            bytes: installDownloadedBytes ?? 0, at: ProcessInfo.processInfo.systemUptime)
        setModelDownloadStatus(.running)
    }

    func canRetryModelDownload(_ download: ModelDownload) -> Bool {
        download.status.canRetry
            && canQueueModelDownload(download.request)
            && !isModelAlreadyInstalled(download.request)
    }

    /// A stale Failed row outlives a later successful install of the same
    /// alias. Retrying it would rewrite the install in place, under a
    /// session that may have the weights mapped.
    func isModelAlreadyInstalled(_ request: ModelDownload.Request) -> Bool {
        switch request {
        case .catalog(let alias), .repository(_, let alias, _, _):
            return installed.contains { $0.alias == alias }
        case .image(let alias):
            return imageModels.contains { $0.alias == alias }
        case .audio(let identity):
            return audioWorkspace.installedPaths[identity.alias] != nil
        }
    }

    func retryModelDownload(_ download: ModelDownload) {
        guard canRetryModelDownload(download) else { return }
        switch download.request {
        case .catalog(let alias):
            installModel(alias: alias)
        case .repository(let repo, let alias, let file, let sidecarRepo):
            installRepo(repo: repo, alias: alias, file: file, sidecarRepo: sidecarRepo)
        case .image(let alias):
            enqueueModelDownload(.image(alias: alias))
        case .audio(let identity):
            enqueueModelDownload(.audio(identity: identity))
        }
    }

    func clearFinishedDownloads() {
        modelDownloads.removeAll { $0.status.isTerminal }
        persistModelDownloads()
    }

    func restoreModelDownloads() {
        do {
            modelDownloads = try downloadHistoryStore.load()
            downloadHistoryWritable = true
            isDownloadManagerExpanded = modelDownloads.contains { $0.status == .interrupted }
        } catch {
            // Preserve an unreadable archive rather than overwriting it with
            // an empty history on the next download.
            downloadHistoryWritable = false
            NSLog("Could not load download history: %@", error.localizedDescription)
            showToast(String(localized: "Download history could not be loaded.", bundle: .module), style: .error)
        }
    }

    func persistModelDownloads(force: Bool = true, at uptime: TimeInterval = ProcessInfo.processInfo.systemUptime) {
        guard downloadHistoryWritable else { return }
        if !force, let last = lastDownloadHistorySaveUptime, uptime - last < 2 { return }
        do {
            try downloadHistoryStore.save(modelDownloads)
            lastDownloadHistorySaveUptime = uptime
            downloadHistorySaveFailed = false
        } catch {
            if !downloadHistorySaveFailed {
                NSLog("Could not save download history: %@", error.localizedDescription)
                showToast(String(localized: "Download history could not be saved.", bundle: .module), style: .error)
            }
            downloadHistorySaveFailed = true
        }
    }

    func recordModelDownloadProgress(
        done: UInt64, total: UInt64,
        at uptime: TimeInterval = ProcessInfo.processInfo.systemUptime,
        now: Date = Date()
    ) {
        guard !modelDownloadsShuttingDown,
              let index = modelDownloads.firstIndex(where: { $0.id == activeModelDownloadID })
        else { return }
        let received = max(modelDownloads[index].downloadedBytes, done)
        modelDownloads[index].downloadedBytes = received
        if total > 0 { modelDownloads[index].totalBytes = total }
        modelDownloads[index].updatedAt = now
        installDownloadedBytes = received
        installTotalBytes = modelDownloads[index].totalBytes
        if let expected = installTotalBytes, expected > 0 {
            installProgressFraction = min(Double(received) / Double(expected), 1)
        }
        if !isInstallPaused { downloadTransferMeter.record(bytes: received, at: uptime) }
        persistModelDownloads(force: false, at: uptime)
    }

    func modelDownloadSpeed(at uptime: TimeInterval = ProcessInfo.processInfo.systemUptime) -> Double? {
        guard hasActiveModelDownload, !isInstallPaused, !isCancellingModelInstall,
              !modelDownloadsShuttingDown,
              modelDownloads.first(where: { $0.id == activeModelDownloadID })?.status == .running
        else { return nil }
        return downloadTransferMeter.bytesPerSecond(at: uptime)
    }

    func prepareModelDownloadsForShutdown() {
        guard !modelDownloadsShuttingDown else { return }
        modelDownloadsShuttingDown = true
        if let index = modelDownloads.firstIndex(where: { $0.id == activeModelDownloadID }) {
            modelDownloads[index].status = modelDownloads[index].status.afterRelaunch
            modelDownloads[index].updatedAt = Date()
        }
        // Flush before the profile vault closes. Later callbacks from a
        // stopping native worker must not write into a newly unlocked profile.
        persistModelDownloads()
        if hasActiveModelDownload { TurboSparkCatalog.cancelInstall() }
    }
}
