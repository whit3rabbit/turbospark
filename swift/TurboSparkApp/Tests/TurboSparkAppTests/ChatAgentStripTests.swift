import XCTest
@testable import TurboSparkApp

/// The chat header strip that names the active agent profile (component
/// audit, agent-card). The Coding Agent is what every chat gets without a
/// project choice, so the strip appears only for another profile.
final class ChatAgentStripTests: XCTestCase {
    func testHiddenWithoutAProjectProfile() {
        XCTAssertNil(ChatAgentStripView.shownAgent(nil))
    }

    func testHiddenForTheDefaultProfile() {
        XCTAssertNil(ChatAgentStripView.shownAgent(.coder))
    }

    func testShownForEveryOtherProfile() {
        for type in AppAgentType.allCases where type != .coder {
            XCTAssertEqual(ChatAgentStripView.shownAgent(type), type)
        }
    }
}
