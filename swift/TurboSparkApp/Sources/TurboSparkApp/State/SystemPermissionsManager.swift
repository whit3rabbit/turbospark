import AppKit
import Foundation
import SwiftUI

/// Status of filesystem accessibility for a folder.
public enum FolderAccessStatus: String, Sendable, Equatable {
    case granted = "Granted"
    case restricted = "Requires Permission"
    case notFound = "Not Found"
    case unknown = "Unknown"

    public var isGranted: Bool {
        self == .granted
    }

    public var statusColor: Color {
        switch self {
        case .granted: return .green
        case .restricted: return .orange
        case .notFound: return .secondary
        case .unknown: return .secondary
        }
    }

    public var systemIcon: String {
        switch self {
        case .granted: return "checkmark.circle.fill"
        case .restricted: return "exclamationmark.triangle.fill"
        case .notFound: return "questionmark.folder"
        case .unknown: return "circle.dotted"
        }
    }
}

/// Standard macOS user protected folders.
public enum SystemFolderType: String, CaseIterable, Identifiable, Sendable {
    case documents = "Documents"
    case downloads = "Downloads"
    case desktop = "Desktop"
    case home = "Home Directory"

    public var id: String { rawValue }

    public var systemImage: String {
        switch self {
        case .documents: return "doc.fill"
        case .downloads: return "arrow.down.doc.fill"
        case .desktop: return "menubar.dock.rectangle"
        case .home: return "house.fill"
        }
    }

    public var defaultURL: URL? {
        switch self {
        case .documents:
            return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        case .downloads:
            return FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
        case .desktop:
            return FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first
        case .home:
            return FileManager.default.homeDirectoryForCurrentUser
        }
    }

    public var pathDisplay: String {
        switch self {
        case .documents: return "~/Documents"
        case .downloads: return "~/Downloads"
        case .desktop: return "~/Desktop"
        case .home: return "~"
        }
    }

    public var description: String {
        switch self {
        case .documents:
            return "Used for accessing attached documents, notes, and user project files."
        case .downloads:
            return "Used for downloading model checkpoints, datasets, and reading imported files."
        case .desktop:
            return "Used for accessing desktop attachments and user workspace folders."
        case .home:
            return "Root user folder for config files (.turbospark, .mcp, scripts)."
        }
    }
}

/// A user-granted custom folder path for workspaces and tools.
public struct GrantedCustomFolder: Identifiable, Codable, Equatable, Sendable {
    public var id: String { path }
    public var path: String
    public var name: String
    public var dateAdded: Date

    public init(path: String, name: String? = nil, dateAdded: Date = Date()) {
        self.path = path
        self.name = name ?? (URL(fileURLWithPath: path).lastPathComponent.isEmpty ? path : URL(fileURLWithPath: path).lastPathComponent)
        self.dateAdded = dateAdded
    }
}

/// macOS Privacy & Security deep link targets.
public enum MacPrivacyPane: String, Sendable {
    case filesAndFolders = "Privacy_FilesAndFolders"
    case fullDiskAccess = "Privacy_AllFiles"
    case accessibility = "Privacy_Accessibility"

    public var url: URL? {
        URL(string: "x-apple.systempreferences:com.apple.preference.security?\(rawValue)")
    }
}

/// Observable manager for inspecting and requesting macOS filesystem and privacy permissions.
@MainActor
public final class SystemPermissionsManager: ObservableObject {
    public static let shared = SystemPermissionsManager()

    /// **THESE ARE macOS TCC GRANTS, NOT TOOL-SANDBOX GRANTS** (state#62).
    /// The pane advertised them as "granted project and workspace directories
    /// for file tools", and nothing in `AppToolSandbox` reads this list --
    /// `SandboxConfig()`'s `allowedWritePaths` is empty at all three call
    /// sites. Nor COULD it help without a second change: every file tool goes
    /// through `resolveSecurePath` first, which refuses anything outside the
    /// project root before the sandbox is consulted at all.
    ///
    /// What picking a folder here really does is move TCC for it, which is
    /// worth having and is what the copy says now. Widening the file tools'
    /// boundary to these paths is an access-control decision, not a wiring
    /// omission, and is deliberately not taken here.
    public static let customFoldersStorageKey = "TurboSpark_GrantedCustomFolders"

    @Published public private(set) var folderStatuses: [SystemFolderType: FolderAccessStatus] = [:]
    @Published public private(set) var customFolders: [GrantedCustomFolder] = []

    public init() {
        loadCustomFolders()
        // **PROBED OFF THE MAIN ACTOR.** Each probe is a full
        // `contentsOfDirectory` on `~/Documents`, `~/Downloads` and
        // `~/Desktop`; on a large home directory that is a visible hitch at
        // launch, and this initializer runs before the first view builds.
        refreshAllStatusesInBackground()
    }

    /// Probes accessibility for all standard system folders, synchronously.
    ///
    /// Kept for the explicit Refresh button, where the user has asked and is
    /// waiting for an answer.
    public func refreshAllStatuses() {
        var newStatuses: [SystemFolderType: FolderAccessStatus] = [:]
        for folder in SystemFolderType.allCases {
            newStatuses[folder] = checkFolderStatus(folder)
        }
        folderStatuses = newStatuses
    }

    /// The same probe, off the main actor, publishing when it lands.
    ///
    /// Also the hook for re-probing on `didBecomeActive`: a grant made in
    /// System Settings was invisible until the manual Refresh, so the pane
    /// kept saying "restricted" for a folder the user had just allowed.
    public func refreshAllStatusesInBackground() {
        // **EPOCHED** (state#62). This is called from `init`, from the
        // Refresh button and from `didBecomeActive`, so two probes overlap
        // routinely -- a user granting access in System Settings and coming
        // back triggers one while the launch probe may still be walking a
        // large home directory. Whichever finished LAST won, which is not
        // whichever STARTED last: the older, pre-grant reading could land on
        // top of the fresh one and the pane kept saying "restricted" for a
        // folder the user had just allowed, which is the exact symptom the
        // `didBecomeActive` hook was added to fix.
        probeGeneration += 1
        let generation = probeGeneration
        Task { [weak self] in
            let folders = SystemFolderType.allCases
            let probed = await Task.detached(priority: .utility) {
                var results: [SystemFolderType: FolderAccessStatus] = [:]
                for folder in folders {
                    results[folder] = Self.probe(folder)
                }
                return results
            }.value
            guard let self, self.probeGeneration == generation else { return }
            self.folderStatuses = probed
        }
    }

    /// Bumped per probe; a landing probe publishes only if it is still the
    /// newest.
    private var probeGeneration = 0

    /// `checkFolderStatus`'s body, `nonisolated` so it can run off the main
    /// actor. It touches only the filesystem.
    nonisolated static func probe(_ folder: SystemFolderType) -> FolderAccessStatus {
        // **THE HOME ROW CANNOT FAIL WHEN PROBED AT HOME** (state#62).
        // Listing `~` is not TCC-gated on macOS, so this row read `.granted`
        // on every machine whatever the user had allowed -- a status that can
        // only take one value, which is `swift/CLAUDE.md` Gotcha 22's badge
        // that cannot fail. What "full disk access" really controls for a
        // home directory is its PROTECTED children, so the probe asks about
        // one that is gated and that every account has.
        if folder == .home {
            let library = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support/com.apple.TCC")
            do {
                _ = try FileManager.default.contentsOfDirectory(atPath: library.path)
                return .granted
            } catch {
                return FileManager.default.fileExists(atPath: library.path)
                    ? .restricted : .notFound
            }
        }
        guard let url = folder.defaultURL else { return .notFound }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
            isDir.boolValue
        else { return .notFound }
        do {
            _ = try FileManager.default.contentsOfDirectory(atPath: url.path)
            return .granted
        } catch {
            return .restricted
        }
    }

    /// Evaluates if a specific system folder is accessible for reading.
    public func checkFolderStatus(_ folder: SystemFolderType) -> FolderAccessStatus {
        guard let url = folder.defaultURL else {
            return .notFound
        }
        return evaluatePathAccess(url: url)
    }

    /// Evaluates whether a directory at given URL exists and is readable.
    public func evaluatePathAccess(url: URL) -> FolderAccessStatus {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            return .notFound
        }
        guard isDir.boolValue else {
            return .notFound
        }

        // Attempt a non-modifying directory read operation to verify TCC access
        do {
            _ = try FileManager.default.contentsOfDirectory(atPath: url.path)
            return .granted
        } catch {
            return .restricted
        }
    }

    /// Prompts the user with an NSOpenPanel focused on the requested system folder,
    /// triggering macOS TCC / Security prompt and registering access.
    public func requestFolderAccess(for folder: SystemFolderType) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.prompt = "Grant Access"
        panel.message = "Select your \(folder.rawValue) folder to grant TurboSpark access."

        if let defaultURL = folder.defaultURL {
            panel.directoryURL = defaultURL
        }

        panel.begin { [weak self] response in
            // The URL is deliberately not kept (state#62): see below. Bound
            // and then unused, it was also the app's one live compiler
            // warning.
            if response == .OK, panel.url != nil {
                // **NOT `startAccessingSecurityScopedResource`.** That call
                // is only meaningful for a URL resolved from a security-scoped
                // BOOKMARK, and it was never balanced by a matching stop --
                // so in this unsandboxed app it did nothing at all while
                // reading as though access were being held open. What
                // actually grants access here is the user having picked the
                // folder in the panel, which is what moves TCC.
                self?.refreshAllStatusesInBackground()
            }
        }
    }

    /// Prompts the user to pick an arbitrary custom workspace folder to grant access to.
    public func addCustomFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.prompt = "Allow Folder"
        panel.message = "Select a folder to authorize for TurboSpark tool and project access."

        panel.begin { [weak self] response in
            guard let self = self, response == .OK, let selectedURL = panel.url else { return }
            // See `requestFolderAccess`: no scoped-resource call, because this
            // app is unsandboxed and the one here was never balanced.
            self.grantCustomFolder(path: selectedURL.path)
        }
    }

    /// Adds a folder path to the granted custom folders list.
    public func grantCustomFolder(path: String) {
        let clean = (path as NSString).standardizingPath
        guard !clean.isEmpty else { return }
        if !customFolders.contains(where: { $0.path == clean }) {
            customFolders.append(GrantedCustomFolder(path: clean))
            saveCustomFolders()
        }
    }

    /// Removes a folder from the granted custom folders list.
    public func removeCustomFolder(id: String) {
        customFolders.removeAll { $0.id == id }
        saveCustomFolders()
    }

    /// Opens the corresponding macOS System Settings Privacy pane.
    public func openSystemPrivacySettings(_ pane: MacPrivacyPane) {
        guard let url = pane.url else { return }
        NSWorkspace.shared.open(url)
    }

    /// Reveals a folder in Finder.
    public func revealInFinder(url: URL?) {
        guard let url = url else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    // MARK: - Persistence
    private static var storeURL: URL { AppStorageRoot.file("granted_folders.json") }

    /// Under `AppStorageRoot`, not `UserDefaults.standard` (state#57): the
    /// test suite was writing the developer's real grant list, and the
    /// preferences domain moves with the bundle identity (`swift/CLAUDE.md`
    /// Gotcha 12), so installing the app dropped every folder the user had
    /// authorized under `swift run`. The old defaults key is read once and
    /// left in place.
    private func loadCustomFolders() {
        var loaded =
            AppJSONStore.load(
                [GrantedCustomFolder].self, from: Self.storeURL, label: "granted folders") ?? []
        if loaded.isEmpty,
            let legacy = UserDefaults.standard.data(forKey: Self.customFoldersStorageKey),
            let migrated = try? JSONDecoder().decode([GrantedCustomFolder].self, from: legacy),
            !migrated.isEmpty
        {
            loaded = migrated
            AppJSONStore.save(loaded, to: Self.storeURL, label: "Granted folders")
        }
        customFolders = loaded
    }

    private func saveCustomFolders() {
        AppJSONStore.save(customFolders, to: Self.storeURL, label: "Granted folders")
    }
}
