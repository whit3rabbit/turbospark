import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The chat pane's no-model routing: a saved chat stays readable with a
/// warning, and only an empty chat is replaced by the first-run setup screen.
@MainActor
final class ModelSetupRoutingTests: XCTestCase {
    private func makeModel(messages: [AppChatMessage]) -> AppModel {
        let appModel = AppModel()
        appModel.stopCronScheduler()
        appModel.installed = []
        appModel.isInstallingModel = false
        let chat = AppChat(title: "history", messages: messages)
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        return appModel
    }

    func testSetupScreenTermsRequireAnEmptyChatWithNothingInstalledOrInstalling() {
        for requiresInstall in [false, true] {
            for installing in [false, true] {
                for hasTranscript in [false, true] {
                    let shows = AppModel.showsModelSetupScreenTerms(
                        requiresInstall: requiresInstall, installing: installing,
                        hasTranscript: hasTranscript)
                    XCTAssertEqual(
                        shows, requiresInstall && !installing && !hasTranscript,
                        "requiresInstall=\(requiresInstall) installing=\(installing) "
                            + "hasTranscript=\(hasTranscript)")
                }
            }
        }
    }

    func testAChatWithHistoryStaysVisibleAndWarnsWhenNoModelIsInstalled() {
        let appModel = makeModel(messages: [
            AppChatMessage(role: .user, content: "Give me a poem about a spark."),
            AppChatMessage(role: .assistant, content: "In the dark of the engine's deep..."),
        ])

        XCTAssertTrue(appModel.hasOutputTranscript)
        XCTAssertFalse(
            appModel.showsModelSetupScreen,
            "Selecting a saved chat with nothing installed showed the setup screen "
                + "instead of the conversation.")
        XCTAssertTrue(appModel.showsNoModelInstalledBanner)
    }

    func testAnEmptyChatShowsTheSetupScreenWhenNoModelIsInstalled() {
        let appModel = makeModel(messages: [])

        XCTAssertFalse(appModel.hasOutputTranscript)
        XCTAssertTrue(appModel.showsModelSetupScreen)
    }

    func testAnInstallInFlightHidesBothTheSetupScreenAndTheWarning() {
        let appModel = makeModel(messages: [])
        appModel.isInstallingModel = true

        XCTAssertFalse(appModel.showsModelSetupScreen)
        XCTAssertFalse(appModel.showsNoModelInstalledBanner)
    }

    func testAnInstalledModelHidesBothTheSetupScreenAndTheWarning() {
        let appModel = makeModel(messages: [])
        appModel.installed = [
            InstalledModel(alias: "one", repo: "x/y", path: "/models/one.gturbo", family: "gemma4")
        ]

        XCTAssertFalse(appModel.showsModelSetupScreen)
        XCTAssertFalse(appModel.showsNoModelInstalledBanner)
    }
}
