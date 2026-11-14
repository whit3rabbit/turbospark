import Foundation

/// A project's say over the OC-11 formatter pass.
///
/// The pass has TWO switches and both must allow it: the app-wide switch
/// (`FormatterSettings.isEnabled`, off by default, the user's trust grant) and
/// this per-project field. The project can only NARROW the app-wide switch:
/// there is deliberately no "on" case, so a project can never turn the pass on
/// when the app-wide switch is off. That keeps a cloned or imported project
/// record from widening what runs a command on disk.
///
/// An archive written before this field existed, a missing key and a value a
/// newer release wrote all decode to `.inherit` (`decodeTolerant`), which is
/// the unchanged pre-field behavior.
public enum ProjectFormatterMode: String, Codable, CaseIterable, Identifiable, Sendable {
    /// Follow the app-wide switch.
    case inherit
    /// Never run a formatter over this project's files, whatever the
    /// app-wide switch says.
    case off

    public var id: String { rawValue }

    /// Whether the pass may run: the app-wide switch AND the project's say.
    static func allowsPass(appWideEnabled: Bool, project: ProjectFormatterMode) -> Bool {
        appWideEnabled && project != .off
    }
}
