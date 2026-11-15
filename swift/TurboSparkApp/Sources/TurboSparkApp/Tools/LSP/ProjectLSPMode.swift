import Foundation

/// A project's say over the OC-10 language servers.
///
/// Same shape as `ProjectFormatterMode` (OC-11), same rule: two switches and
/// both must allow it. The app-wide `lsp.enabled` switch (off by default, the
/// user's trust grant) and this per-project field. The project can only NARROW
/// the app-wide switch: there is deliberately no "on" case, so a cloned or
/// imported project record can never start a server the user did not enable.
///
/// A missing key, an archive from before this field and a value a newer
/// release wrote all decode to `.inherit` (`decodeTolerant`).
public enum ProjectLSPMode: String, Codable, CaseIterable, Identifiable, Sendable {
    /// Follow the app-wide switch.
    case inherit
    /// Start no language server for this project, whatever the app-wide
    /// switch says.
    case off

    public var id: String { rawValue }

    /// Whether any server may run: the app-wide switch AND the project's say.
    static func allowsServers(appWideEnabled: Bool, project: ProjectLSPMode) -> Bool {
        appWideEnabled && project != .off
    }

    /// Whether a save turns the servers off where they were not: the moment
    /// `AppModel.updateProject` stops the project's running servers.
    static func isNarrowing(from old: ProjectLSPMode, to new: ProjectLSPMode) -> Bool {
        new == .off && old != .off
    }
}
