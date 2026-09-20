import AppKit
import Foundation
import UniformTypeIdentifiers

/// The UI-facing half of profile backups (`ProfileBackup` and
/// `ProfileBackupImport` hold the logic): panels, toasts, the
/// flush-before-export rule, and the folder-first registry-last ordering
/// that mirrors delete's trash-first rule.
extension AppModel {
    /// What the import sheet describes: where the backup came from and the
    /// name it suggests. `id` exists for SwiftUI's `sheet(item:)`.
    struct ProfileBackupImportOffer: Identifiable, Equatable {
        let id = UUID()
        let url: URL
        let manifest: ProfileBackup.Manifest
        let suggestedName: String
    }

    // MARK: - Export

    /// Gate for opening the export sheet; the pane shows it with every
    /// category preselected.
    var canBeginProfileBackupExport: Bool { !profileBackupInFlight }

    /// Asks for a destination and writes a backup of `profile` carrying the
    /// selected categories. Runs the save panel modally like `exportChat`;
    /// the archive itself runs on a background task and toasts its outcome.
    func runProfileBackupExport(_ profile: UserProfile, included: Set<String>) {
        guard !profileBackupInFlight else { return }
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.allowedContentTypes = [.zip]
        let clock = DateFormatter()
        clock.dateFormat = "yyyy-MM-dd"
        clock.locale = Locale(identifier: "en_US_POSIX")
        let base = ProfileBackup.sanitizedFileName(profile.name, fallback: "Profile")
        panel.nameFieldStringValue = "\(base) Profile Backup \(clock.string(from: Date())).zip"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        // The same flush the quit path ends with, so the backup cannot miss
        // the last keystroke or the last setting. Only the profile THIS run
        // owns can have dirty in-memory state; every other profile's folder
        // is already exactly what its disk says.
        if profile.id == UserProfileStore.active.id {
            persistChats()
            persistSettings()
            AppChatFileStore.flush()
        }

        let machineRoot = AppStorageRoot.machineRoot
        let turbosparkHome = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".turbospark", isDirectory: true)
        let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"]
            as? String ?? ""
        profileBackupInFlight = true
        Task.detached(priority: .userInitiated) { [weak self] in
            let outcome: Result<ProfileBackup.Manifest, Error>
            do {
                outcome = .success(try await ProfileBackup.export(
                    profile: profile,
                    machineRoot: machineRoot,
                    turbosparkHome: turbosparkHome,
                    destination: url,
                    appVersion: appVersion,
                    included: included))
            } catch {
                outcome = .failure(error)
            }
            // Bound once here rather than inside the MainActor closure: the
            // toast needs the model, and the task is bounded by the archive
            // run, so holding it is not a cycle.
            guard let self else { return }
            await MainActor.run {
                self.profileBackupInFlight = false
                switch outcome {
                case .success:
                    self.showToast("Backup of \"\(profile.name)\" exported.", style: .success)
                case .failure(let error):
                    self.showToast(
                        "Backup export failed: \(self.profileBackupFailureMessage(error))",
                        style: .error, duration: 8)
                }
            }
        }
    }

    // MARK: - Import

    /// Opens a backup, validates it without extracting, and returns what the
    /// import sheet needs, or nil (with a toast) when the file is refused.
    /// Async because validation shells out (`zipinfo`, `unzip -p`) without
    /// ever blocking the main thread on a pipe.
    func pickProfileBackupForImport() async -> ProfileBackupImportOffer? {
        guard !profileBackupInFlight else { return nil }
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.zip]
        panel.message = "Choose a TurboSpark profile backup (.zip)."
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        do {
            let manifest = try await ProfileBackupImport.readSummary(archive: url)
            return ProfileBackupImportOffer(
                url: url,
                manifest: manifest,
                suggestedName: suggestedImportName(for: manifest))
        } catch {
            showToast(profileBackupImportFailureMessage(error), style: .error, duration: 8)
            return nil
        }
    }

    /// The manifest's name, sanitized like every profile-name input and made
    /// unique: never the reserved "Default", never a case-insensitive
    /// duplicate of an existing row. A Default-user backup therefore lands
    /// as "Default Imported" rather than colliding with the built-in user.
    /// The sheet lets the result be edited; the registry mutation re-validates
    /// whatever survives editing.
    func suggestedImportName(for manifest: ProfileBackup.Manifest) -> String {
        let base = ProfileBackup.sanitizedFileName(
            manifest.profileName, fallback: "Imported Profile")
        let root = UserProfileStore.isReservedName(base) ? base + " Imported" : base
        let registry = UserProfileRegistry(
            profiles: profiles, activeProfileID: UserProfileStore.active.id)
        var candidate = root
        var collisions = 1
        while registry.profiles.contains(where: {
            $0.name.caseInsensitiveCompare(candidate) == .orderedSame
        }) {
            collisions += 1
            candidate = collisions == 2 ? "\(root) Imported" : "\(root) Imported \(collisions - 1)"
        }
        return candidate
    }

    /// Runs the import. Returns false (with a toast, sheet left open) when
    /// the chosen name fails validation; returns true once the heavy work is
    /// dispatched and the sheet may close. The copy fills the folder FIRST
    /// and the registry row is saved LAST, so any failure leaves no row
    /// pointing at a partial folder.
    @discardableResult
    func importProfileBackup(_ offer: ProfileBackupImportOffer, named chosenName: String) -> Bool {
        guard !profileBackupInFlight else { return false }
        var registry = UserProfileStore.loadRegistry()
        // A fresh identity on purpose: the manifest's id is provenance, never
        // the imported row's, so a Default backup's "default" id and any id
        // that collides with a live profile cannot enter the registry.
        let profile = UserProfile(
            id: UUID().uuidString, name: chosenName, createdAt: offer.manifest.profileCreatedAt)
        do {
            try UserProfileStore.adding(profile, to: &registry)
        } catch {
            showToast(profileMutationMessage(error), style: .error)
            return false
        }
        let destination = UserProfileStore.folder(of: profile)
        profileBackupInFlight = true
        Task.detached(priority: .userInitiated) { [weak self] in
            let outcome: Result<ProfileBackup.Manifest, Error>
            do {
                outcome = .success(try await ProfileBackupImport.install(
                    archive: offer.url, destination: destination))
            } catch {
                outcome = .failure(error)
            }
            guard let self else { return }
            await MainActor.run {
                self.profileBackupInFlight = false
                switch outcome {
                case .success:
                    if UserProfileStore.saveRegistry(registry) {
                        self.profiles = registry.profiles
                        self.showToast(
                            "Profile \"\(profile.name)\" imported from backup. Switch to it to start using it.",
                            style: .success)
                    } else {
                        // No row was saved, so the half-restored folder is an
                        // orphan; remove it rather than leave gigabytes with
                        // no owner.
                        try? FileManager.default.removeItem(at: destination)
                        self.surfaceStorageIssues()
                    }
                case .failure(let error):
                    try? FileManager.default.removeItem(at: destination)
                    self.showToast(
                        "Backup import failed: \(self.profileBackupImportFailureMessage(error))",
                        style: .error, duration: 8)
                }
            }
        }
        return true
    }

    // MARK: - Failure wording

    private func profileBackupFailureMessage(_ error: Error) -> String {
        switch error as? ProfileBackup.ProcessError {
        case .sourceMissing:
            return "the profile folder does not exist"
        case .stagingFailed:
            return "could not create a staging folder"
        case .processFailed(let step, _, let stderr):
            let reason = stderr.split(separator: "\n").first.map(String.init) ?? stderr
            return reason.isEmpty ? "the \(step) step failed" : reason
        case nil:
            return error.localizedDescription
        }
    }

    private func profileBackupImportFailureMessage(_ error: Error) -> String {
        switch error as? ProfileBackupImport.ImportError {
        case .unreadableArchive:
            return "that file is not a readable backup archive"
        case .unsafeEntries(let paths):
            let shown = paths.prefix(3).joined(separator: ", ")
            return "the archive contains unsafe paths (\(shown)) and was refused"
        case .manifestMissing:
            return "that archive has no profile-backup manifest"
        case .manifestCorrupt:
            return "the backup's manifest could not be read"
        case .unsupportedVersion(let version):
            return "that backup uses format \(version), which this app does not read; update the app first"
        case .layoutCorrupt:
            return "the backup is missing part of its declared layout"
        case nil:
            return error.localizedDescription
        }
    }
}
