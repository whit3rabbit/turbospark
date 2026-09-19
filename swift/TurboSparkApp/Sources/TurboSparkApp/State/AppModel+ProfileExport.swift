import AppKit
import Foundation
import UniformTypeIdentifiers

extension AppModel {
    func runEncryptedProfileBackupExport(passphrase: String) {
        guard !profileBackupInFlight else { return }
        persistChats()
        persistProjects()
        persistSettings()
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.allowedContentTypes = [
            UTType(filenameExtension: EncryptedProfileBackup.fileExtension) ?? .data,
        ]
        let date = Date().formatted(.iso8601.year().month().day())
        let name = ProfileBackup.sanitizedFileName(currentProfile.name, fallback: "Profile")
        panel.nameFieldStringValue = "\(name) Exact Backup \(date).\(EncryptedProfileBackup.fileExtension)"
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        let profile = currentProfile
        let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? ""
        profileBackupInFlight = true
        Task.detached(priority: .userInitiated) { [weak self] in
            let outcome = Result {
                try EncryptedProfileBackup.export(
                    profile: profile,
                    destination: destination,
                    passphrase: passphrase,
                    appVersion: appVersion)
            }
            guard let self else { return }
            await MainActor.run {
                self.profileBackupInFlight = false
                switch outcome {
                case .success:
                    self.showToast("Encrypted profile backup exported.", style: .success)
                case .failure(let error):
                    self.showToast(
                        "Encrypted backup failed: \(error.localizedDescription)",
                        style: .error, duration: 10)
                }
            }
        }
    }

    func runOpenProfileExport(included: Set<String>) {
        guard !profileBackupInFlight else { return }
        persistChats()
        persistProjects()
        persistSettings()
        let snapshot: ProfileExportSnapshot
        do {
            snapshot = try makeProfileExportSnapshot()
        } catch {
            showToast("Open export failed: \(error.localizedDescription)", style: .error)
            return
        }
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.allowedContentTypes = [.zip]
        let date = Date().formatted(.iso8601.year().month().day())
        let name = ProfileBackup.sanitizedFileName(currentProfile.name, fallback: "Profile")
        panel.nameFieldStringValue = "\(name) Open Export \(date).zip"
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        profileBackupInFlight = true
        Task.detached(priority: .userInitiated) { [weak self] in
            let outcome = Result {
                try OpenProfileExport.export(
                    snapshot: snapshot, included: included, destination: destination)
            }
            guard let self else { return }
            await MainActor.run {
                self.profileBackupInFlight = false
                switch outcome {
                case .success:
                    self.showToast("Plaintext profile export written.", style: .success)
                case .failure(let error):
                    self.showToast(
                        "Open export failed: \(error.localizedDescription)",
                        style: .error, duration: 10)
                }
            }
        }
    }

    func pickEncryptedProfileBackup() -> URL? {
        guard !profileBackupInFlight else { return nil }
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [
            UTType(filenameExtension: EncryptedProfileBackup.fileExtension) ?? .data,
        ]
        panel.message = "Choose an encrypted TurboSpark profile backup."
        return panel.runModal() == .OK ? panel.url : nil
    }

    @discardableResult
    func importEncryptedProfileBackup(
        _ archive: URL,
        named chosenName: String,
        passphrase: String
    ) -> Bool {
        guard !profileBackupInFlight else { return false }
        let newID = UUID().uuidString
        var registry = UserProfileStore.loadRegistry()
        let validation = UserProfile(id: newID, name: chosenName)
        do {
            try UserProfileStore.adding(validation, to: &registry)
            registry.profiles.removeAll { $0.id == newID }
        } catch {
            showToast(profileMutationMessage(error), style: .error)
            return false
        }
        let destination = UserProfileStore.storeDirectory(
            profileID: newID, machineRoot: AppStorageRoot.machineRoot)!
        profileBackupInFlight = true
        Task.detached(priority: .userInitiated) { [weak self] in
            let outcome: Result<(EncryptedProfileBackup.Manifest, UserProfile), Error>
            do {
                outcome = .success(try await EncryptedProfileBackup.restore(
                    archive: archive,
                    destination: destination,
                    newProfileID: newID,
                    displayName: chosenName.trimmingCharacters(in: .whitespacesAndNewlines),
                    passphrase: passphrase))
            } catch {
                outcome = .failure(error)
            }
            guard let self else { return }
            await MainActor.run {
                self.profileBackupInFlight = false
                switch outcome {
                case .success((_, let profile)):
                    registry.profiles.append(profile)
                    if UserProfileStore.saveRegistry(registry) {
                        self.profiles = registry.profiles
                        self.showToast(
                            "Encrypted backup restored as a locked profile.", style: .success)
                    } else {
                        try? FileManager.default.removeItem(at: destination)
                        self.surfaceStorageIssues()
                    }
                case .failure(let error):
                    try? FileManager.default.removeItem(at: destination)
                    self.showToast(
                        "Encrypted backup restore failed: \(error.localizedDescription)",
                        style: .error, duration: 10)
                }
            }
        }
        return true
    }
}
