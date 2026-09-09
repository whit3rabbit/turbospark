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
        guard !isInstallingModel, !generating else { return }
        // A refusal is reported rather than swallowed: a Download button that
        // does nothing reads as a broken button.
        if let why = installBlockReason(alias: alias) {
            showToast(
                "'\(alias)' will not run on this machine. \(why)",
                style: .warning, duration: 8.0)
            return
        }
        guard !abandonedInstallAliases.contains(alias) else {
            // A cancelled install's walk is still exiting; the refusal lifts
            // on its own once `installsFinished()` proves it exited (and
            // stays if it never does), so a second writer never races the
            // first on this directory.
            showToast(
                "'\(alias)' has a download still exiting in the background. Try again "
                    + "in a few seconds, or restart the app.",
                style: .warning, duration: 6.0)
            return
        }
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
        showToast("Starting download for '\(alias)'...", style: .info)

        let startedAt = Date()
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
                        self.installETAText = Self.installETA(
                            done: maxBytes, total: total,
                            elapsed: Date().timeIntervalSince(startedAt))
                    case .finished(let model):
                        // **THE EPOCH CHECK BELONGS ON THE SUCCESS ARM TOO**
                        // (state#52). The comment on the `catch` below says
                        // it comes first, and it existed only there -- so a
                        // `.finished` event buffered behind a cancel still
                        // ran `refreshModels`, reassigned `selected` and
                        // OPENED a model for an install nobody was watching,
                        // on top of whatever the user had started since.
                        guard self.installEpoch == myEpoch else { return }
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
                // **THE EPOCH CHECK COMES FIRST.** It used to sit below these
                // arms, so a cancelled install A whose delayed tail ran after
                // install B had started overwrote B's live progress text with
                // "Installation cancelled" and raised a toast about a
                // download nobody was watching.
                guard self.installEpoch == myEpoch else { return }
                self.installStageText = "Stopped watching install"
            } catch {
                guard self.installEpoch == myEpoch else { return }
                self.error = error.localizedDescription
                self.installStageText = nil
                self.showToast("Installation failed: \(error.localizedDescription)", style: .error, duration: 5.0)
            }
            guard self.installEpoch == myEpoch else { return }
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
        guard !isInstallingModel, !generating else { return }
        let trimmedRepo = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedAlias = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedRepo.isEmpty, !trimmedAlias.isEmpty else { return }
        // **THE TWO-WRITER PROTECTION APPLIES HERE TOO** (state#44).
        // `cancelInstall` records an abandoned alias because the engine
        // exposes no install-cancel call: dropping the consumer ends DELIVERY
        // while `ts_install_repo` keeps streaming the checkpoint into that
        // directory. `installModel` refuses a retry for the rest of the
        // process on that basis and this path did neither -- it never
        // recorded `installingAlias`, so a cancel here marked nothing, and it
        // never checked the set, so a retry wrote into a directory an
        // orphaned thread was still filling.
        guard !abandonedInstallAliases.contains(trimmedAlias) else {
            showToast(
                "'\(trimmedAlias)' has a download still running in the background from an earlier "
                    + "attempt. Restart the app before installing it again.",
                style: .warning, duration: 6.0)
            return
        }
        installingAlias = trimmedAlias

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

        let startedAt = Date()
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
                        self.installETAText = Self.installETA(
                            done: maxBytes, total: total,
                            elapsed: Date().timeIntervalSince(startedAt))
                    case .finished(let model):
                        // **THE EPOCH CHECK BELONGS ON THE SUCCESS ARM TOO**
                        // (state#52). The comment on the `catch` below says
                        // it comes first, and it existed only there -- so a
                        // `.finished` event buffered behind a cancel still
                        // ran `refreshModels`, reassigned `selected` and
                        // OPENED a model for an install nobody was watching,
                        // on top of whatever the user had started since.
                        guard self.installEpoch == myEpoch else { return }
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
                // **THE EPOCH CHECK COMES FIRST.** It used to sit below these
                // arms, so a cancelled install A whose delayed tail ran after
                // install B had started overwrote B's live progress text with
                // "Installation cancelled" and raised a toast about a
                // download nobody was watching.
                guard self.installEpoch == myEpoch else { return }
                self.installStageText = "Stopped watching install"
            } catch {
                guard self.installEpoch == myEpoch else { return }
                let desc = error.localizedDescription
                if desc.contains("401") || desc.contains("403") || desc.contains("gated") || desc.localizedCaseInsensitiveContains("unauthorized") {
                    self.showToast("Installation failed: Authentication required. Check your Hugging Face API token in Settings.", style: .error, duration: 6.0)
                } else {
                    self.showToast("Installation failed: \(desc)", style: .error, duration: 5.0)
                }
            }
            guard self.installEpoch == myEpoch else { return }
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

    /// Cancels an install for real.
    ///
    /// **TWO CANCELS REACH TWO THREADS.** `TurboSparkCatalog.cancelInstall()`
    /// sets the flag the blocking C walk polls at its next ranged chunk
    /// read; `installTask?.cancel()` ends the Swift consumer. The walk then
    /// dies the same death a network failure gives it -- the error carries
    /// "install cancelled", and NOTHING of the partial install is kept (the
    /// walk cannot resume either way).
    ///
    /// The alias sits in `abandonedInstallAliases` only for the seconds the
    /// walk takes to notice the flag: `watchCancelledWalkExit` watches
    /// `installsFinished()` and lifts the refusal once the walk has
    /// actually exited, so Cancel + Retry works without an app restart. If
    /// the walk somehow never notices, the refusal stays -- the pre-cancel
    /// behavior, still safe (two writers on one install directory is the
    /// case that corrupts something).
    public func cancelInstall() {
        let cancelledAlias = installingAlias
        let wasRunning = TurboSparkCatalog.cancelInstall() || isInstallingModel
        installTask?.cancel()
        installTask = nil
        installingAlias = nil
        isInstallingModel = false
        installStageText = nil
        installProgressFraction = nil
        installDownloadedBytes = nil
        installTotalBytes = nil
        installETAText = nil
        showToast(
            wasRunning
                ? "Download cancelled. Nothing was kept; '\(cancelledAlias ?? "the model")' can "
                    + "be re-installed in a few seconds."
                : "Nothing was downloading.",
            style: .info, duration: 5.0)
        // Cancellation is cooperative: the cancelled Task keeps running
        // until its next suspension point notices, so its own tail can
        // still fire after this call returns. Bumping the epoch here too
        // (on top of each new install bumping it at its own start) means
        // that stale tail is a no-op even if nothing new has started yet.
        installEpoch += 1
        if wasRunning, let cancelledAlias {
            abandonedInstallAliases.insert(cancelledAlias)
            watchCancelledWalkExit(alias: cancelledAlias)
        }
    }

    /// Lifts the abandoned-alias refusal once the cancelled walk has
    /// actually exited. Bounded: if the walk never notices the flag within
    /// 30 s, the refusal stays until restart, which is safe.
    private func watchCancelledWalkExit(alias: String) {
        let baseline = TurboSparkCatalog.installsFinished()
        Task { [weak self] in
            for _ in 0..<60 {
                try? await Task.sleep(nanoseconds: 500_000_000)
                guard let self, !self.abandonedInstallAliases.contains(alias) else { return }
                if TurboSparkCatalog.installsFinished() != baseline {
                    self.abandonedInstallAliases.remove(alias)
                    return
                }
            }
        }
    }

    /// Whether `alias` may be installed right now.
    ///
    /// False while any install is running, and false forever after this
    /// alias's install was abandoned -- there is no way to learn that the
    /// orphaned walk finished, so a second writer on the same directory is
    /// refused rather than raced.
    public func canInstall(alias: String) -> Bool {
        !isInstallingModel && !generating && !abandonedInstallAliases.contains(alias)
    }
}
