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

    public static let customFoldersStorageKey = "TurboSpark_GrantedCustomFolders"

    @Published public private(set) var folderStatuses: [SystemFolderType: FolderAccessStatus] = [:]
    @Published public private(set) var customFolders: [GrantedCustomFolder] = []

    public init() {
        loadCustomFolders()
        refreshAllStatuses()
    }

    /// Probes accessibility for all standard system folders.
    public func refreshAllStatuses() {
        var newStatuses: [SystemFolderType: FolderAccessStatus] = [:]
        for folder in SystemFolderType.allCases {
            newStatuses[folder] = checkFolderStatus(folder)
        }
        folderStatuses = newStatuses
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
            if response == .OK, let selectedURL = panel.url {
                _ = selectedURL.startAccessingSecurityScopedResource()
                self?.refreshAllStatuses()
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
            _ = selectedURL.startAccessingSecurityScopedResource()
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
    private func loadCustomFolders() {
        guard let data = UserDefaults.standard.data(forKey: Self.customFoldersStorageKey) else {
            customFolders = []
            return
        }
        do {
            customFolders = try JSONDecoder().decode([GrantedCustomFolder].self, from: data)
        } catch {
            customFolders = []
        }
    }

    private func saveCustomFolders() {
        do {
            let data = try JSONEncoder().encode(customFolders)
            UserDefaults.standard.set(data, forKey: Self.customFoldersStorageKey)
        } catch {
            // Non-fatal persistence failure
        }
    }
}
