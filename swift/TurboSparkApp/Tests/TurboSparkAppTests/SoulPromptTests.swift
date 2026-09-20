import Foundation
import XCTest

@testable import TurboSparkApp

final class SoulPromptTests: XCTestCase {
    private var tempDirectoryURL: URL!

    /// AppModel.init() loads whatever settings.json an earlier test in this
    /// process persisted, and the soul CRUD methods persist. Zero the soul
    /// fields (a plain assignment, deliberately NOT a persisting write) so
    /// each case starts from the fresh-install soul state.
    @MainActor
    private func freshSoulModel() -> AppModel {
        let model = AppModel()
        model.soulPrompts = []
        model.selectedSoulPromptID = nil
        model.soulPromptEnabled = false
        return model
    }

    override func setUp() {
        super.setUp()
        tempDirectoryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(
            at: tempDirectoryURL, withIntermediateDirectories: true)
    }

    override func tearDown() {
        if let tempDirectoryURL {
            try? FileManager.default.removeItem(at: tempDirectoryURL)
        }
        super.tearDown()
    }

    // MARK: - Fresh defaults: disabled, empty library

    func testFreshSettingsShipSoulDisabledWithNoLibrary() {
        let settings = MacAppSettings()

        XCTAssertFalse(settings.soulPromptEnabled)
        XCTAssertTrue(settings.soulPrompts.isEmpty)
        XCTAssertEqual(settings.activeSoulPromptID, "")
        XCTAssertEqual(settings.soulPrompt, "")
    }

    /// **THE LOAD-BEARING ONE.** SOUL stays off until enabled: the gate in
    /// `resolvedSoulPrompt` is the whole feature. Remove the
    /// `soulPromptEnabled` guard and this reddens, because a selected entry
    /// is present on the model.
    @MainActor
    func testADisabledSoulContributesNothingEvenWithASelectedEntry() {
        let model = freshSoulModel()
        let soul = model.addSoulPrompt(name: "Hermes", content: "SOUL-CONTENT")
        XCTAssertEqual(model.selectedSoulPromptID, soul.id)
        XCTAssertFalse(model.soulPromptEnabled)

        XCTAssertEqual(model.resolvedSoulPrompt, "")
    }

    @MainActor
    func testAnEnabledSelectedSoulResolvesToItsContent() {
        let model = freshSoulModel()
        let soul = model.addSoulPrompt(name: "Hermes", content: "  SOUL-CONTENT  ")
        model.soulPromptEnabled = true

        XCTAssertEqual(model.resolvedSoulPrompt, "SOUL-CONTENT")
        XCTAssertEqual(model.selectedSoulPrompt, soul)
    }

    @MainActor
    func testEnabledWithoutSelectionContributesNothing() {
        let model = freshSoulModel()
        model.addSoulPrompt(name: "Hermes", content: "SOUL-CONTENT")
        model.selectSoulPrompt(nil)
        model.soulPromptEnabled = true

        XCTAssertEqual(model.resolvedSoulPrompt, "")
    }

    // MARK: - External path detection

    func testHermesHomeOverrideAndDefaultFallback() {
        let home = URL(fileURLWithPath: "/tmp/test-user", isDirectory: true)
        XCTAssertEqual(
            SoulPromptStore.hermesHomeURL(
                environment: ["HERMES_HOME": "~/custom-hermes"], homeDirectory: home),
            home.appendingPathComponent("custom-hermes", isDirectory: true))
        XCTAssertEqual(
            SoulPromptStore.hermesHomeURL(environment: [:], homeDirectory: home),
            home.appendingPathComponent(".hermes", isDirectory: true))
    }

    func testOpenClawWorkspacePathOverridesAndDefault() {
        let home = URL(fileURLWithPath: "/tmp/test-user", isDirectory: true)
        XCTAssertEqual(
            SoulPromptStore.openClawWorkspaceURL(
                environment: ["OPENCLAW_WORKSPACE_DIR": "/tmp/openclaw-workspace"],
                homeDirectory: home),
            URL(fileURLWithPath: "/tmp/openclaw-workspace", isDirectory: true))
        XCTAssertEqual(
            SoulPromptStore.openClawWorkspaceURL(
                environment: ["OPENCLAW_STATE_DIR": "/tmp/openclaw-state"],
                homeDirectory: home),
            URL(fileURLWithPath: "/tmp/openclaw-state/workspace", isDirectory: true))
        XCTAssertEqual(
            SoulPromptStore.openClawWorkspaceURL(
                environment: ["OPENCLAW_PROFILE": "work"], homeDirectory: home),
            home.appendingPathComponent(".openclaw/workspace-work", isDirectory: true))
        XCTAssertEqual(
            SoulPromptStore.openClawWorkspaceURL(environment: [:], homeDirectory: home),
            home.appendingPathComponent(".openclaw/workspace", isDirectory: true))
    }

    func testDetectedImportSourcesIncludeExistingHermesAndOpenClawFiles() throws {
        let hermesHome = tempDirectoryURL.appendingPathComponent(".hermes", isDirectory: true)
        let openClawWorkspace = tempDirectoryURL
            .appendingPathComponent(".openclaw/workspace", isDirectory: true)
        try FileManager.default.createDirectory(at: hermesHome, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: openClawWorkspace, withIntermediateDirectories: true)
        try Data().write(to: hermesHome.appendingPathComponent("SOUL.md"))
        try Data("OPENCLAW".utf8).write(to: openClawWorkspace.appendingPathComponent("SOUL.md"))

        let sources = SoulPromptStore.detectedImportSources(
            environment: [:], homeDirectory: tempDirectoryURL)
        XCTAssertEqual(sources.map(\.kind), [.hermes, .openClaw])
        XCTAssertEqual(sources.map(\.fileURL), [
            hermesHome.appendingPathComponent("SOUL.md"),
            openClawWorkspace.appendingPathComponent("SOUL.md")
        ])
    }

    func testImportReadSupportsArbitraryFileWithoutOverwritingIt() throws {
        let sourceURL = tempDirectoryURL.appendingPathComponent("persona.txt")
        try Data("IMPORTED".utf8).write(to: sourceURL)

        XCTAssertEqual(try SoulPromptStore.read(fileURL: sourceURL), "IMPORTED")
        XCTAssertEqual(try String(contentsOf: sourceURL), "IMPORTED")
    }

    func testCreateHermesIsAtomicAndDoesNotOverwriteUnexpectedly() throws {
        try SoulPromptStore.createHermes(content: "FIRST", hermesHome: tempDirectoryURL)
        XCTAssertEqual(
            try String(contentsOf: tempDirectoryURL.appendingPathComponent("SOUL.md")), "FIRST")

        XCTAssertThrowsError(
            try SoulPromptStore.createHermes(content: "SECOND", hermesHome: tempDirectoryURL))
        XCTAssertEqual(
            try String(contentsOf: tempDirectoryURL.appendingPathComponent("SOUL.md")), "FIRST")
    }

    func testReadHermesLeavesTheHermesFileAlone() throws {
        let hermesURL = tempDirectoryURL.appendingPathComponent("SOUL.md")
        try Data("HERMES-ORIGINAL".utf8).write(to: hermesURL)

        let imported = try SoulPromptStore.readHermes(hermesHome: tempDirectoryURL)
        XCTAssertEqual(imported, "HERMES-ORIGINAL")
        XCTAssertEqual(try String(contentsOf: hermesURL), "HERMES-ORIGINAL")
    }

    // MARK: - Library CRUD

    @MainActor
    func testAddSoulPromptSelectsTheNewRowAndSuffixesCollidingNames() {
        let model = freshSoulModel()
        let first = model.addSoulPrompt(name: "Hermes", content: "ONE")
        let second = model.addSoulPrompt(name: "Hermes", content: "TWO")

        XCTAssertEqual(first.name, "Hermes")
        XCTAssertEqual(second.name, "Hermes 2")
        XCTAssertEqual(model.selectedSoulPrompt?.id, second.id)
        XCTAssertEqual(model.soulPrompts.map(\.name), ["Hermes", "Hermes 2"])
    }

    @MainActor
    func testUpdateSoulPromptSavesTheSelectedRowContent() {
        let model = freshSoulModel()
        let soul = model.addSoulPrompt(name: "Draft", content: "OLD")

        model.updateSoulPrompt(id: soul.id, content: "NEW")

        XCTAssertEqual(model.soulPrompts.first?.content, "NEW")
    }

    @MainActor
    func testDeleteOfTheSelectedRowLeavesNothingSelected() {
        let model = freshSoulModel()
        let soul = model.addSoulPrompt(name: "Draft", content: "OLD")
        XCTAssertEqual(model.selectedSoulPromptID, soul.id)

        model.deleteSoulPrompt(soul.id)

        XCTAssertTrue(model.soulPrompts.isEmpty)
        XCTAssertNil(model.selectedSoulPromptID)
        XCTAssertEqual(model.resolvedSoulPrompt, "")
    }

    @MainActor
    func testAStaleSelectionResolvesToNoRowNeverAnotherEntry() {
        let model = freshSoulModel()
        let first = model.addSoulPrompt(name: "One", content: "ONE")
        model.addSoulPrompt(name: "Two", content: "TWO")
        model.selectedSoulPromptID = first.id
        model.deleteSoulPrompt(first.id)

        XCTAssertNil(model.selectedSoulPrompt)
    }

    // MARK: - Import into the library

    @MainActor
    func testImportCreatesANamedRowWithoutEnabling() throws {
        let sourceURL = tempDirectoryURL.appendingPathComponent("persona.md")
        try Data("IMPORTED-SOUL".utf8).write(to: sourceURL)

        let model = freshSoulModel()
        try model.importSoul(from: sourceURL)

        XCTAssertEqual(model.soulPrompts.map(\.name), ["persona"])
        XCTAssertEqual(model.soulPrompts.first?.content, "IMPORTED-SOUL")
        XCTAssertEqual(model.selectedSoulPrompt?.name, "persona")
        XCTAssertFalse(model.soulPromptEnabled, "import must not flip the enable toggle")
        XCTAssertEqual(model.resolvedSoulPrompt, "")
    }

    @MainActor
    func testReimportRefreshesTheSameNamedRowInsteadOfStacking() throws {
        let sourceURL = tempDirectoryURL.appendingPathComponent("persona.md")
        try Data("FIRST".utf8).write(to: sourceURL)
        let model = freshSoulModel()
        try model.importSoul(from: sourceURL)

        try Data("SECOND".utf8).write(to: sourceURL)
        try model.importSoul(from: sourceURL)

        XCTAssertEqual(model.soulPrompts.count, 1)
        XCTAssertEqual(model.soulPrompts.first?.content, "SECOND")
    }

    @MainActor
    func testDetectedSourceImportUsesTheHarnessDisplayName() throws {
        let hermesURL = tempDirectoryURL.appendingPathComponent("SOUL.md")
        try Data("HERMES-SOUL".utf8).write(to: hermesURL)

        let model = freshSoulModel()
        try model.importSoul(from: SoulPromptImportSource(kind: .hermes, fileURL: hermesURL))

        XCTAssertEqual(model.soulPrompts.map(\.name), ["Hermes"])
        XCTAssertEqual(try String(contentsOf: hermesURL), "HERMES-SOUL",
                       "the external file is never modified by an import")
    }

    // MARK: - Persistence and legacy migration

    func testSettingsWrittenBeforeTheSoulLibraryExistedStillDecode() throws {
        let legacy = #"{"contextTokens":0,"temperature":0.2}"#
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data(legacy.utf8))

        XCTAssertFalse(decoded.soulPromptEnabled)
        XCTAssertTrue(decoded.soulPrompts.isEmpty)
        XCTAssertEqual(decoded.activeSoulPromptID, "")
    }

    func testLegacySoulFallbackMigratesToOneEnabledSelectedEntry() throws {
        let legacy = #"{"soulPrompt":"NATIVE-SOUL"}"#
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data(legacy.utf8))

        XCTAssertEqual(decoded.soulPrompt, "", "the legacy field is retired after migrating")
        XCTAssertTrue(decoded.soulPromptEnabled, "an authored soul was an explicit choice")
        XCTAssertEqual(decoded.soulPrompts.map(\.name), ["Imported Soul"])
        XCTAssertEqual(decoded.soulPrompts.first?.content, "NATIVE-SOUL")
        XCTAssertEqual(decoded.activeSoulPromptID, decoded.soulPrompts[0].id.uuidString)
    }

    func testTheMigrationNeverReFiresOnALaterLoad() throws {
        let soul = AppSoulPrompt(name: "Hermes", content: "LIVE")
        let settings = MacAppSettings(
            soulPrompt: "",
            soulPromptEnabled: true,
            soulPrompts: [soul],
            activeSoulPromptID: soul.id.uuidString)
        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)

        XCTAssertEqual(decoded.soulPrompts.map(\.name), ["Hermes"])
        XCTAssertEqual(decoded.soulPrompts.first?.content, "LIVE")
        XCTAssertTrue(decoded.soulPromptEnabled)
        XCTAssertEqual(decoded.activeSoulPromptID, decoded.soulPrompts[0].id.uuidString)
    }

    func testSettingsRoundTripTheSoulLibrary() throws {
        let soul = AppSoulPrompt(name: "Hermes", content: "NATIVE-SOUL")
        let settings = MacAppSettings(
            soulPromptEnabled: true,
            soulPrompts: [soul],
            activeSoulPromptID: soul.id.uuidString)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self, from: JSONEncoder().encode(settings))

        XCTAssertEqual(decoded.soulPromptEnabled, true)
        XCTAssertEqual(decoded.soulPrompts, [soul])
        XCTAssertEqual(decoded.activeSoulPromptID, soul.id.uuidString)
    }

    func testAStaleSelectedSoulIDDecodesToNone() throws {
        let soul = AppSoulPrompt(name: "Hermes", content: "LIVE")
        let json = """
        {"soulPrompts":[],"soulPromptEnabled":true,
         "activeSoulPromptID":"\(soul.id.uuidString)"}
        """
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data(json.utf8))

        XCTAssertEqual(decoded.activeSoulPromptID, "")
        XCTAssertTrue(decoded.soulPrompts.isEmpty)
    }

    // MARK: - Prompt assembly

    @MainActor
    func testGlobalSoulIsAppWideAndComposesWithPersonality() {
        let personality = AppPersonality(name: "Custom", instructions: "PERSONALITY")
        let model = freshSoulModel()
        model.defaultSystemPrompt = "SYSTEM"
        model.addSoulPrompt(name: "Soul", content: "SOUL")
        model.soulPromptEnabled = true
        model.personalities = [personality]
        model.selectedPersonalityID = personality.id

        let sections = model.buildSystemPromptSections(for: nil, userPrompt: "USER")
        XCTAssertEqual(sections.map(\.section), [.userPrompt, .soul, .personality])
        XCTAssertEqual(model.appWideSystemPrompt, "SYSTEM\n\nSOUL\n\nPERSONALITY")
    }

    @MainActor
    func testADisabledSoulIsAbsentFromPromptAssembly() {
        let personality = AppPersonality(name: "Custom", instructions: "PERSONALITY")
        let model = freshSoulModel()
        model.defaultSystemPrompt = "SYSTEM"
        model.addSoulPrompt(name: "Soul", content: "SOUL")
        model.personalities = [personality]
        model.selectedPersonalityID = personality.id

        let sections = model.buildSystemPromptSections(for: nil, userPrompt: "USER")
        XCTAssertEqual(sections.map(\.section), [.userPrompt, .personality])
        XCTAssertEqual(model.appWideSystemPrompt, "SYSTEM\n\nPERSONALITY")
    }

    @MainActor
    func testSoulChangesAreReflectedInSubagentPrompt() {
        let model = freshSoulModel()
        model.addSoulPrompt(name: "Soul", content: "LIVE-SOUL")
        model.soulPromptEnabled = true
        let agent = AgentManager.shared.findAgent(name: "general-purpose")!
        let prompt = SubagentRunner.buildSystemPrompt(
            for: agent, project: nil, userPrompt: model.appWideSystemPrompt)

        XCTAssertTrue(prompt.contains("LIVE-SOUL"))
    }

    @MainActor
    func testADisabledSoulStaysOutOfTheSubagentPrompt() {
        let model = freshSoulModel()
        model.addSoulPrompt(name: "Soul", content: "LIVE-SOUL")
        let agent = AgentManager.shared.findAgent(name: "general-purpose")!
        let prompt = SubagentRunner.buildSystemPrompt(
            for: agent, project: nil, userPrompt: model.appWideSystemPrompt)

        XCTAssertFalse(prompt.contains("LIVE-SOUL"))
    }

    @MainActor
    func testGlobalSoulIsIncludedInContextAccounting() {
        let model = freshSoulModel()
        model.addSoulPrompt(name: "Soul", content: "ACCOUNTED-SOUL")
        model.soulPromptEnabled = true
        let chat = AppChat(title: "usage", messages: [
            AppChatMessage(role: .user, content: "hello")
        ])
        model.chats = [chat]
        model.selectedChatID = chat.id

        let systemPiece = model.buildEstimateParts().pieces.first { $0.label == "System prompt" }
        XCTAssertTrue(systemPiece?.content.contains("ACCOUNTED-SOUL") ?? false)
    }

    @MainActor
    func testSoulCharacterAssetLoadsFromBundle() {
        let image = SoulCharacterAssetCache.shared.soulImage
        XCTAssertNotNil(image, "SoulCharacter asset should load from bundle resources")
        if let image {
            XCTAssertGreaterThan(image.size.width, 0)
            XCTAssertGreaterThan(image.size.height, 0)
        }
        let view = TSIdlingSoulSparkView(size: 80)
        XCTAssertEqual(view.size, 80)
    }
}
