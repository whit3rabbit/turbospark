import Foundation
import XCTest

@testable import TurboSparkApp

/// Task 6.11, review round 1: the wiring no behavior test can reach.
///
/// The cases that drive the move run an `AppModel` and a fake engine. None of them
/// builds a SwiftUI view, so a sheet that is never hosted, a button that is never
/// wired to `presentMoveSheet`, a sheet that can be dismissed mid-move and a Cancel
/// that answers Escape would all pass with the whole suite green. These scans read
/// the sources with comments removed (the 6.6 precedent, `LoadPathSourceScanTests`)
/// and hold each of those lines in place. Each was checked by deleting the line it
/// names: the case goes red.
final class ModelMoveSourceScanTests: XCTestCase {
    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
    }

    /// The file's code with comments removed and runs of blanks squeezed to one
    /// space, so a line that wraps is still one call.
    private func code(_ file: String) throws -> String {
        let text = try String(contentsOf: sourcesRoot.appendingPathComponent(file), encoding: .utf8)
        let joined = RawModelDeleteScanTests.code(of: text).joined(separator: "\n")
        return joined.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" }).joined(separator: " ")
    }

    private func assertContains(
        _ file: String, _ needle: String, _ why: String, line: UInt = #line
    ) throws {
        XCTAssertTrue(try code(file).contains(needle), "\(file) must contain `\(needle)`: \(why)", line: line)
    }

    // MARK: The sheet is hosted

    func testTheTextAndImageListHostsTheMoveSheet() throws {
        try assertContains(
            "Installation/ModelManagerView.swift", ".modelMoveSheet(model: model, kinds: [.text, .image])",
            "without the host, Move to... for a text or an image row opens nothing")
    }

    func testTheAudioPickerHostsTheMoveSheet() throws {
        try assertContains(
            "Audio/AudioModelPickerView.swift", ".modelMoveSheet(model: model, kinds: [.audio])",
            "without the host, Move to... for an audio model opens nothing")
    }

    func testEachKindIsHostedExactlyOnce() throws {
        // Two hosts for one kind would present the sheet twice; none would present
        // it never. `ModelMoveSheetHost` filters by kind, so each kind has one home.
        var hosts: [String: Int] = [:]
        let walker = try XCTUnwrap(FileManager.default.enumerator(at: sourcesRoot, includingPropertiesForKeys: nil))
        for case let url as URL in walker where url.pathExtension == "swift" {
            let relative = String(url.path.dropFirst(sourcesRoot.path.count + 1))
            let text = try code(relative)
            let count = text.components(separatedBy: ".modelMoveSheet(model:").count - 1
            if count > 0 { hosts[relative] = count }
        }
        XCTAssertEqual(
            hosts, ["Installation/ModelManagerView.swift": 1, "Audio/AudioModelPickerView.swift": 1],
            "the sheet is hosted once for text and image rows and once for audio")
    }

    // MARK: Something presents it

    func testTheDetailPaneActionBarOpensTheSheet() throws {
        try assertContains(
            "Installation/InstalledModelActionBarView.swift", "Button(action: onMove)",
            "the action bar's Move to... button calls onMove")
        try assertContains(
            "Installation/InstalledModelDetailPaneView.swift", "model.presentMoveSheet(for: subject)",
            "the detail pane's onMove presents the sheet for the row's subject")
        try assertContains(
            "Installation/InstalledModelDetailPaneView.swift", "onMove: {", "and hands the bar the closure")
    }

    func testTheImageRowsAndTheAudioPickerOpenTheSheet() throws {
        try assertContains(
            "Installation/ImageModelManagerView.swift", "model.presentMoveSheet(for: subject)",
            "Move to... on an installed image row")
        try assertContains(
            "Audio/AudioModelPickerView.swift", "model.presentMoveSheet(for: subject)",
            "Move to... on an installed audio model")
    }

    // MARK: A move cannot be dismissed by accident

    func testTheSheetCannotBeDismissedWhileItMoves() throws {
        try assertContains(
            "Components/ModelMoveSheet.swift", ".interactiveDismissDisabled(!sheet.canDismiss)",
            "the close gesture and Escape do nothing while the engine copies")
        // The host's setter must go through the guarded dismissal, never drop the sheet.
        try assertContains(
            "Components/ModelMoveSheet.swift", "_ = model.dismissMoveSheet()",
            "the sheet binding closes through dismissMoveSheet, which refuses mid-move")
    }

    /// The `.moving` arm of `buttonItems` (the body has a `.moving` arm of its own,
    /// which draws the progress), comments removed.
    private func movingArm() throws -> String {
        let text = try code("Components/ModelMoveSheet.swift")
        let function = try XCTUnwrap(text.range(of: "private func buttonItems"), "buttonItems is gone")
        let start = try XCTUnwrap(
            text.range(of: "case .moving:", range: function.upperBound..<text.endIndex),
            "the .moving arm of buttonItems is gone")
        let end = try XCTUnwrap(
            text.range(of: "case .finished:", range: start.upperBound..<text.endIndex), "the .finished arm is gone")
        return String(text[start.upperBound..<end.lowerBound])
    }

    func testTheMovingArmHasNoKeyboardShortcutAndNoCancelRole() throws {
        let arm = try movingArm()
        XCTAssertTrue(arm.contains("sheet.cancel()"), "the extraction found the Cancel button: \(arm)")
        XCTAssertFalse(arm.contains(".keyboardShortcut"), "Return and Escape do nothing while a move runs: \(arm)")
        XCTAssertFalse(arm.contains("role: .cancel"), "a cancel-role button answers Escape: \(arm)")
    }

    func testOnlyTheIdleArmsCloseOnEscape() throws {
        // The Close button keeps the cancel action, so Escape closes an idle sheet.
        let text = try code("Components/ModelMoveSheet.swift")
        let close = try XCTUnwrap(text.range(of: "private var closeButton"), "closeButton is gone")
        let tail = String(text[close.upperBound...].prefix(260))
        XCTAssertTrue(tail.contains("role: .cancel"), tail)
        XCTAssertTrue(tail.contains(".keyboardShortcut(.cancelAction)"), tail)
        let arm = try movingArm()
        XCTAssertFalse(arm.contains("closeButton"), "the moving arm shows no Close button")
    }

    // MARK: The menu-bar window

    func testTheMenuBarWindowLoadsOnlyThroughTheGate() throws {
        // The model list lives in the menu bar dashboard since main's redesign.
        let text = try code("Server/ServerMenuDashboardView.swift")
        XCTAssertTrue(
            text.contains("model.requestLoad(installed, bringWindowForward: true)"),
            "a load from the menu-bar window asks the choke point")
        for direct in ["beginChatLoad", ".open(", "openSession", "selectModel(", "TurboSparkSession("] {
            XCTAssertFalse(text.contains(direct), "the menu-bar window loads around the gate with \(direct)")
        }
    }

    // MARK: The loaders that open by path ask

    func testEveryLoaderThatOpensAModelByPathAsksWhetherItMoves() throws {
        let asks: [(file: String, needle: String, times: Int, why: String)] = [
            ("State/AppModel+LoadGate.swift", "if isMoving(model) { announceModelBeingMoved() return .busy }", 1, "requestLoad"),
            ("State/AppModel+LoadGate.swift", "if isMoving(model) { announceModelBeingMoved() return false }", 1, "canOpenNow"),
            ("State/AppModel+Gating.swift", "isMoving($0)", 1, "canLoadModel (the Model menu)"),
            ("Humanize/HumanizeController.swift", "guard !modelMoving(path) else { throw HumanizeError.modelMoving }", 2, "Humanize: the run and the engine open"),
            ("Humanize/HumanizeController.swift", "if let path = installedPath(), modelMoving(path) {", 1, "Humanize: the Load button"),
            ("State/AppModel+ServerImages.swift", "if isMoving(path: model.path) {", 2, "the image session and the image attach"),
            ("State/AppModel+ImageGeneration.swift", "isMoving(path: selected.path)", 1, "generateImage"),
            ("Audio/AudioWorkspaceController.swift", "if modelMoving?(path) == true {", 1, "the audio session"),
            ("Audio/Qwen3TTSSpeechJob.swift", "modelMoving?(installed) == true", 1, "the Qwen speech job"),
            ("State/AppModel+ServerAttachment.swift", "if isMoving(path: path) {", 1, "the server's audio attach"),
            ("State/AppModel+ServerAudio.swift", "if isMoving(path: model.path) {", 2, "the server's speech attach and request"),
            ("State/AppModel.swift", "controller.modelMoving = { [weak self] in self?.isMoving(path: $0) == true }", 1, "Humanize wiring"),
            ("State/AppModel.swift", "workspace.modelMoving = { [weak self] in self?.isMoving(path: $0) == true }", 1, "audio wiring"),
        ]
        for ask in asks {
            let found = try code(ask.file).components(separatedBy: ask.needle).count - 1
            XCTAssertGreaterThanOrEqual(
                found, ask.times, "\(ask.file) must hold `\(ask.needle)` \(ask.times) time(s) for \(ask.why), found \(found)")
        }
    }
}
