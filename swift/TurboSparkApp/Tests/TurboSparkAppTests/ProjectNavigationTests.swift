import AppKit
import SwiftUI
import XCTest

@testable import TurboSparkApp

@MainActor
final class ProjectNavigationTests: XCTestCase {
    func testChoosingProjectEntersCodingModeAndSelectsItsTask() {
        let model = AppModel()
        // Project selection is persisted across AppModel instances in the test host.
        defer { model.selectProject(id: nil) }
        let project = AppProject(name: "Workspace")
        model.projects = [project]
        model.setInteractionMode(.chat)
        model.chooseProjectForTask(id: project.id)
        XCTAssertEqual(model.interactionMode, .projects)
        XCTAssertEqual(model.selectedProjectID, project.id)
        XCTAssertEqual(model.selectedChat.projectID, project.id)
    }

    func testChoosingNoProjectLeavesTheProjectConversation() {
        let model = AppModel()
        // Project selection is persisted across AppModel instances in the test host.
        defer { model.selectProject(id: nil) }
        let project = AppProject(name: "Workspace")
        model.projects = [project]
        model.chooseProjectForTask(id: project.id)
        model.chooseProjectForTask(id: nil)
        XCTAssertNil(model.selectedProjectID)
        XCTAssertNil(model.selectedChat.projectID)
        XCTAssertEqual(model.interactionMode, .projects)
    }

    func testChoosingProjectDuringSubmissionDoesNotSwitchModeOrTask() {
        let model = AppModel()
        // Project selection is persisted across AppModel instances in the test host.
        defer { model.selectProject(id: nil) }
        let project = AppProject(name: "Workspace")
        model.projects = [project]
        model.setInteractionMode(.chat)
        let chatID = model.selectedChatID
        let projectID = model.selectedProjectID
        model.submitting = true
        model.chooseProjectForTask(id: project.id)
        XCTAssertEqual(model.interactionMode, .chat)
        XCTAssertEqual(model.selectedChatID, chatID)
        XCTAssertEqual(model.selectedProjectID, projectID)
        model.submitting = false
    }

    func testUnknownProjectDoesNotSwitchMode() {
        let model = AppModel()
        // Project selection is persisted across AppModel instances in the test host.
        defer { model.selectProject(id: nil) }
        model.projects = []
        model.setInteractionMode(.chat)
        model.chooseProjectForTask(id: UUID())
        XCTAssertEqual(model.interactionMode, .chat)
    }

    func testCreationSheetFitsItsDeclaredWidth() {
        let host = NSHostingView(rootView: ProjectSettingsSheet(
            model: AppModel(), editingProject: nil, onDismiss: {}))
        host.frame = NSRect(x: 0, y: 0, width: 580, height: 390)
        let window = NSWindow(contentRect: host.frame, styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        XCTAssertEqual(host.fittingSize.width, 580, accuracy: 1)
        XCTAssertEqual(host.fittingSize.height, 390, accuracy: 1)
    }
}
