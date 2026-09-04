import Foundation
import TurboSpark

extension AppModel {
    /// Initiates a background download and build of a model catalog alias.
    public func installModel(alias: String) {
        guard !isInstallingModel, !generating else { return }
        guard !abandonedInstallAliases.contains(alias) else {
            // A previous install of this alias was abandoned and its walk
            // cannot be stopped, so it may still be writing that directory.
            showToast(
                "'\(alias)' has a download still running in the background from an earlier "
                    + "attempt. Restart the app before installing it again.",
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

    /// Stops WATCHING an install. It does not stop the install.
    ///
    /// **THE ENGINE HAS NO INSTALL-CANCEL CALL, AND THE BINDING SAYS SO IN
    /// SO MANY WORDS** (state#27): dropping the consumer ends DELIVERY, while
    /// `ts_install` blocks its own thread and keeps streaming the checkpoint
    /// to completion or failure with nobody listening
    /// (`TurboSpark/Catalog.swift`: "do not build a Stop button on this").
    /// This used to clear `isInstallingModel` synchronously, which reopened
    /// `installModel`'s own guard -- so a second install could start and
    /// write the same store the abandoned walk was still writing.
    ///
    /// Two things follow, and both are the honest version rather than the
    /// convenient one. The alias stays in `abandonedInstallAliases`, so
    /// re-installing THAT model is refused until the app restarts (two
    /// writers on one install directory is the case that corrupts
    /// something); a DIFFERENT model may still be installed, since the store
    /// writes are per directory and `installed.json` is rewritten whole by
    /// each on completion. And the message says what actually happens
    /// instead of claiming a stop. Making Cancel real needs
    /// `ts_install_cancel` on the Rust side, which does not exist.
    public func cancelInstall() {
        installTask?.cancel()
        installTask = nil
        if let alias = installingAlias {
            abandonedInstallAliases.insert(alias)
        }
        installingAlias = nil
        isInstallingModel = false
        installStageText = nil
        installProgressFraction = nil
        installDownloadedBytes = nil
        installTotalBytes = nil
        installETAText = nil
        showToast(
            "Stopped watching the download. It cannot be cancelled and keeps running in the "
                + "background; this model cannot be re-installed until the app restarts.",
            style: .warning, duration: 6.0)
        // Cancellation is cooperative: the cancelled Task keeps running
        // until its next suspension point notices, so its own tail can
        // still fire after this call returns. Bumping the epoch here too
        // (on top of each new install bumping it at its own start) means
        // that stale tail is a no-op even if nothing new has started yet.
        installEpoch += 1
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
