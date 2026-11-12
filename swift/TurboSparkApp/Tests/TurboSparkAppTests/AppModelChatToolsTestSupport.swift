import Foundation

@testable import TurboSparkApp

extension AppModel {
    /// A model whose Chat turns carry no tool vocabulary.
    ///
    /// Chat tools are ON by default (`chatToolsEnabled`), so a projectless
    /// chat's system prompt includes the tool addendum. The tests that pin the
    /// PROSE side of prompt assembly, or an exact history length, are about the
    /// tools-off case and say so by building their model here. The tools-on
    /// side is covered by `ChatToolsTests`.
    @MainActor
    static func withoutChatTools() -> AppModel {
        let model = AppModel()
        model.chatToolsEnabled = false
        return model
    }
}
