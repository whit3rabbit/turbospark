import Foundation
import XCTest

@testable import TurboSparkApp

/// Tool profiles narrow what a turn is offered. The properties worth pinning:
/// Default is byte-identical to the pre-profile capture, a removed tool is
/// refused at validation (not just unlisted), the prompt never tells the model
/// to call a tool it was not given, and a corrupt stored profile never widens.
@MainActor
final class ToolProfileTests: XCTestCase {
    private func capture(
        project: AppProject? = AppProject(name: "fixture", agentType: .coder),
        profile: AppToolProfile
    ) -> TurnAvailableTools {
        AppToolCatalog.captureTurnAvailableTools(
            for: project, chatToolsEnabled: true, globalMcpServers: [],
            contextTokens: nil, webToolsEnabled: true, profile: profile)
    }

    private func names(_ tools: TurnAvailableTools) -> Set<String> {
        Set(tools.definitions.map { $0.function.name.lowercased() })
    }

    // MARK: Filtering

    func testDefaultProfileLeavesTheCaptureUnchanged() {
        let project = AppProject(name: "fixture", agentType: .coder)
        XCTAssertTrue(AppToolProfile.default.allowsEverything)
        XCTAssertEqual(
            capture(project: project, profile: .default),
            AppToolCatalog.captureTurnAvailableTools(
                for: project, chatToolsEnabled: true, globalMcpServers: [],
                contextTokens: nil, webToolsEnabled: true))
        XCTAssertEqual(
            capture(project: nil, profile: .default),
            AppToolCatalog.captureTurnAvailableTools(
                for: nil, chatToolsEnabled: true, globalMcpServers: [],
                contextTokens: nil, webToolsEnabled: true))
    }

    func testMinimalProfileOffersOnlyItsToolsAndTheHarnessOnes() {
        let offered = names(capture(profile: .minimal))
        XCTAssertFalse(offered.isEmpty)
        let allowed = Set(AppToolProfile.minimal.allowedTools ?? [])
        for name in offered {
            let key = AppToolCatalog.profileKey(for: name)
            XCTAssertTrue(
                allowed.contains(key) || AppToolCatalog.profileAlwaysOnKeys.contains(key),
                "\(name) is outside the Minimal profile")
        }
        XCTAssertTrue(offered.contains("bash"))
        XCTAssertFalse(offered.contains("websearch"))
        XCTAssertFalse(offered.contains("todowrite"))
        XCTAssertFalse(offered.contains("agent"))
    }

    func testReadOnlyProfileOffersNothingThatWritesOrRuns() {
        let offered = names(capture(profile: .readOnly))
        XCTAssertFalse(offered.contains("bash"))
        XCTAssertFalse(offered.contains("fileedit"))
        XCTAssertFalse(offered.contains("filewrite"))
        XCTAssertFalse(offered.contains("apply_patch"))
        XCTAssertTrue(offered.contains("grep"))
    }

    func testProfileWithoutMcpDropsMcpAndTheDiscoveryTools() {
        let project = AppProject(name: "fixture", agentType: .coder)
        let withMcp = AppToolProfile(name: "a", allowedTools: ["read"], includesMcpAndCustom: true)
        let withoutMcp = AppToolProfile(name: "b", allowedTools: ["read"], includesMcpAndCustom: false)
        XCTAssertTrue(names(capture(project: project, profile: withMcp)).contains("tool_search"))
        let narrowed = capture(project: project, profile: withoutMcp)
        XCTAssertFalse(names(narrowed).contains("tool_search"))
        XCTAssertTrue(narrowed.deferredMcpTools.isEmpty)
    }

    func testProfileKeyCollapsesFileHandlerSpellings() {
        for name in ["FileRead", "read_file", "read", "cat"] {
            XCTAssertEqual(AppToolCatalog.profileKey(for: name), "read", name)
        }
        for name in ["FileWrite", "write_file", "write"] {
            XCTAssertEqual(AppToolCatalog.profileKey(for: name), "write", name)
        }
        for name in ["FileEdit", "edit_file", "edit"] {
            XCTAssertEqual(AppToolCatalog.profileKey(for: name), "edit", name)
        }
        XCTAssertEqual(AppToolCatalog.profileKey(for: "Bash"), "bash")
    }

    // MARK: Enforcement

    func testACallToARemovedToolIsRefusedNotJustUnlisted() {
        let narrowed = capture(profile: .readOnly)
        guard case .refused(.unavailableTool) = narrowed.validate(
            toolName: "Bash", argumentsJSON: #"{"command":"ls"}"#)
        else {
            return XCTFail("a tool outside the profile must be refused at validation")
        }
    }

    func testPromptDoesNotTeachToolsTheProfileRemoved() {
        let project = AppProject(name: "fixture", agentType: .coder)
        let full = AppToolCatalog.systemPromptAddendum(
            for: .coder, projectURL: nil, availableTools: capture(project: project, profile: .default))
        XCTAssertTrue(full.contains("TodoWrite"))
        XCTAssertTrue(full.contains("## Subagents"))

        let minimal = AppToolCatalog.systemPromptAddendum(
            for: .coder, projectURL: nil, availableTools: capture(project: project, profile: .minimal))
        XCTAssertFalse(minimal.contains("TodoWrite"))
        XCTAssertFalse(minimal.contains("## Subagents"))
    }

    // MARK: Persistence

    func testProfilesRoundTripAndOlderSettingsDefaultToNoneSelected() throws {
        var settings = MacAppSettings()
        XCTAssertTrue(settings.toolProfiles.isEmpty)
        let custom = AppToolProfile(
            name: "Mine", allowedTools: ["Read", " bash ", "read"], includesMcpAndCustom: false)
        settings.toolProfiles = [custom]
        settings.activeToolProfileID = custom.id.uuidString

        let restored = try JSONDecoder().decode(
            MacAppSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertEqual(restored.toolProfiles, [custom])
        XCTAssertEqual(restored.toolProfiles.first?.allowedTools, ["bash", "read"])
        XCTAssertEqual(restored.activeToolProfileID, custom.id.uuidString)

        let older = try JSONDecoder().decode(
            MacAppSettings.self, from: Data(#"{"temperature":0.4}"#.utf8))
        XCTAssertTrue(older.toolProfiles.isEmpty)
        XCTAssertEqual(older.activeToolProfileID, "")
    }

    func testACorruptProfileIsDroppedAndNeverWidenedToEveryTool() throws {
        let json = """
            {"toolProfiles":[
              {"id":"\(UUID().uuidString)","name":"bad","allowedTools":"oops"},
              {"id":"\(UUID().uuidString)","name":"good","allowedTools":["read"]}
            ]}
            """
        let restored = try JSONDecoder().decode(MacAppSettings.self, from: Data(json.utf8))
        XCTAssertEqual(restored.toolProfiles.map(\.name), ["good"])
    }

    // MARK: Model

    private var models: [AppModel] = []

    private static func reset(_ model: AppModel) {
        model.customToolProfiles = []
        model.activeToolProfileID = AppToolProfile.defaultID
        model.toolProfileOverride = nil
        model.chatToolsEnabled = true
        model.persistSettings()
    }

    /// `AppModel()` loads whatever an earlier test persisted into the shared
    /// test storage root, so every model test starts from a reset and puts
    /// the root back when it finishes.
    private func makeModel() -> AppModel {
        let model = AppModel()
        Self.reset(model)
        models.append(model)
        return model
    }

    override func tearDown() async throws {
        for model in models { Self.reset(model) }
        models = []
        try await super.tearDown()
    }

    func testSelectionFallsBackToDefaultWhenTheRowIsGone() {
        let model = makeModel()
        let id = model.addToolProfile(name: "Temp", allowedTools: ["read"], includesMcpAndCustom: false)
        XCTAssertEqual(model.selectedToolProfile.id, id)
        model.deleteToolProfile(id!)
        XCTAssertEqual(model.selectedToolProfile.id, AppToolProfile.defaultID)
        XCTAssertEqual(model.activeToolProfileID, AppToolProfile.defaultID)
    }

    func testBuiltInProfilesCannotBeEditedOrDeleted() {
        let model = makeModel()
        model.updateToolProfile(
            id: AppToolProfile.minimalID, name: "Hacked", allowedTools: [], includesMcpAndCustom: true)
        model.deleteToolProfile(AppToolProfile.minimalID)
        XCTAssertEqual(
            model.allToolProfiles.first { $0.id == AppToolProfile.minimalID }, .minimal)
        XCTAssertTrue(model.customToolProfiles.isEmpty)
    }

    func testComposerOverrideNeverMutatesTheSavedProfile() {
        let model = makeModel()
        model.interactionMode = .projects
        XCTAssertNil(model.toolProfileOverride)

        model.setToolEnabled("bash", false)
        let override = model.toolProfileOverride
        XCTAssertNotNil(override)
        XCTAssertFalse(model.isToolEnabled("bash"))
        XCTAssertTrue(model.isToolEnabled("grep"))
        XCTAssertEqual(model.selectedToolProfile, .default, "Default is never edited")

        model.setToolEnabled("bash", true)
        XCTAssertNil(model.toolProfileOverride, "back on the saved profile clears the override")

        model.setToolEnabled("bash", false)
        model.resetToolOverride()
        XCTAssertTrue(model.isToolEnabled("bash"))
    }

    func testSavingTheOverrideMakesAnEditableProfile() throws {
        let model = makeModel()
        model.setToolEnabled("bash", false)
        let id = try XCTUnwrap(model.saveToolOverride(as: "No shell"))
        XCTAssertEqual(model.selectedToolProfile.id, id)
        XCTAssertNil(model.toolProfileOverride)
        XCTAssertFalse(model.isToolEnabled("bash"))
        XCTAssertFalse(model.selectedToolProfile.isBuiltIn)
    }

    func testChatListsOnlyToolsThatCanRunWithoutAWorkspace() {
        let model = makeModel()
        model.interactionMode = .chat
        let chat = Set(model.toolGroupsForContext.flatMap { $0.entries.map(\.id) })
        XCTAssertFalse(chat.contains("bash"))
        XCTAssertFalse(chat.contains("read"))
        XCTAssertTrue(chat.contains("websearch"))

        model.interactionMode = .projects
        let project = Set(model.toolGroupsForContext.flatMap { $0.entries.map(\.id) })
        XCTAssertTrue(project.contains("bash"))
    }

    func testChatOverrideDoesNotSwitchOnFileToolsInProjects() {
        let model = makeModel()
        model.selectToolProfile(AppToolProfile.minimalID)
        model.interactionMode = .chat
        // Enabling every tool Chat can list must not collapse to "no allowlist".
        for key in model.toolGroupsForContext.flatMap({ $0.entries.map(\.id) }) {
            model.setToolEnabled(key, true)
        }
        model.interactionMode = .projects
        XCTAssertNotNil(model.activeToolProfile.allowedTools)
        XCTAssertFalse(model.isToolEnabled("grep"))
    }

    func testDotIsOffWhenChatToolsAreOffOrNothingIsEnabled() {
        let model = makeModel()
        model.interactionMode = .chat
        model.chatToolsEnabled = true
        XCTAssertTrue(model.toolsAreOn)
        model.chatToolsEnabled = false
        XCTAssertFalse(model.toolsAreOn)

        model.chatToolsEnabled = true
        let none = model.addToolProfile(name: "None", allowedTools: [], includesMcpAndCustom: false)
        XCTAssertNotNil(none)
        XCTAssertFalse(model.toolsAreOn)
    }
}
