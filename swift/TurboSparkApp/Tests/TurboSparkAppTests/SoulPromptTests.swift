import Foundation
import XCTest

@testable import TurboSparkApp

final class SoulPromptTests: XCTestCase {
    private var tempDirectoryURL: URL!

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

    func testDefaultSoulIsBlank() {
        let settings = MacAppSettings()
        let resolution = SoulPromptStore.resolve(fallback: settings.soulPrompt, hermesHome: tempDirectoryURL)

        XCTAssertEqual(settings.soulPrompt, "")
        XCTAssertEqual(resolution.content, "")
        XCTAssertEqual(resolution.source, .turboSpark)
        XCTAssertFalse(resolution.hermesFileExists)
    }

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

    func testMissingHermesFileUsesNativeFallback() {
        let resolution = SoulPromptStore.resolve(
            fallback: "NATIVE-FALLBACK", hermesHome: tempDirectoryURL)

        XCTAssertEqual(resolution.content, "NATIVE-FALLBACK")
        XCTAssertEqual(resolution.source, .turboSpark)
    }

    func testExistingHermesFileWinsEvenWhenEmpty() throws {
        try FileManager.default.createDirectory(at: tempDirectoryURL, withIntermediateDirectories: true)
        try Data("HERMES-CONTENT".utf8)
            .write(to: tempDirectoryURL.appendingPathComponent("SOUL.md"))
        let content = SoulPromptStore.resolve(fallback: "NATIVE", hermesHome: tempDirectoryURL)
        XCTAssertEqual(content.content, "HERMES-CONTENT")
        XCTAssertEqual(content.source, .hermes)

        try Data().write(to: tempDirectoryURL.appendingPathComponent("SOUL.md"))
        let empty = SoulPromptStore.resolve(fallback: "NATIVE", hermesHome: tempDirectoryURL)
        XCTAssertEqual(empty.content, "")
        XCTAssertEqual(empty.source, .hermes)
        XCTAssertTrue(empty.hermesFileExists)
    }

    func testCreateAndWriteAreAtomicAndDoNotOverwriteUnexpectedly() throws {
        try SoulPromptStore.createHermes(content: "FIRST", hermesHome: tempDirectoryURL)
        XCTAssertEqual(
            try String(contentsOf: tempDirectoryURL.appendingPathComponent("SOUL.md")), "FIRST")

        XCTAssertThrowsError(
            try SoulPromptStore.createHermes(content: "SECOND", hermesHome: tempDirectoryURL))
        XCTAssertEqual(
            try String(contentsOf: tempDirectoryURL.appendingPathComponent("SOUL.md")), "FIRST")

        try SoulPromptStore.writeHermes(content: "UPDATED", hermesHome: tempDirectoryURL)
        XCTAssertEqual(
            try String(contentsOf: tempDirectoryURL.appendingPathComponent("SOUL.md")), "UPDATED")
    }

    func testImportCopiesHermesContentWithoutChangingHermes() throws {
        let hermesURL = tempDirectoryURL.appendingPathComponent("SOUL.md")
        try Data("HERMES-ORIGINAL".utf8).write(to: hermesURL)

        let imported = try SoulPromptStore.readHermes(hermesHome: tempDirectoryURL)
        XCTAssertEqual(imported, "HERMES-ORIGINAL")
        XCTAssertEqual(try String(contentsOf: hermesURL), "HERMES-ORIGINAL")
    }

    func testLegacySettingsDecodeWithBlankSoulFallback() throws {
        let legacy = #"{"contextTokens":0,"temperature":0.2}"#
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data(legacy.utf8))

        XCTAssertEqual(decoded.soulPrompt, "")
    }

    func testSettingsRoundTripSoulFallback() throws {
        let settings = MacAppSettings(soulPrompt: "NATIVE-SOUL")
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self, from: JSONEncoder().encode(settings))

        XCTAssertEqual(decoded.soulPrompt, "NATIVE-SOUL")
    }

    @MainActor
    func testGlobalSoulIsAppWideAndComposesWithPersonality() {
        let personality = AppPersonality(name: "Custom", instructions: "PERSONALITY")
        let model = AppModel()
        model.defaultSystemPrompt = "SYSTEM"
        model.soulPrompt = "SOUL"
        model.personalities = [personality]
        model.selectedPersonalityID = personality.id

        let sections = model.buildSystemPromptSections(for: nil, userPrompt: "USER")
        XCTAssertEqual(sections.map(\.section), [.environment, .userPrompt, .soul, .personality])
        XCTAssertEqual(model.appWideSystemPrompt, "SYSTEM\n\nSOUL\n\nPERSONALITY")
    }

    @MainActor
    func testSoulChangesAreReflectedInSubagentPrompt() {
        let model = AppModel()
        model.soulPrompt = "LIVE-SOUL"
        let agent = AgentManager.shared.findAgent(name: "general-purpose")!
        let prompt = SubagentRunner.buildSystemPrompt(
            for: agent, project: nil, userPrompt: model.appWideSystemPrompt)

        XCTAssertTrue(prompt.contains("LIVE-SOUL"))
    }

    @MainActor
    func testGlobalSoulIsIncludedInContextAccounting() {
        let model = AppModel()
        model.soulPrompt = "ACCOUNTED-SOUL"
        let chat = AppChat(title: "usage", messages: [
            AppChatMessage(role: .user, content: "hello")
        ])
        model.chats = [chat]
        model.selectedChatID = chat.id

        let systemPiece = model.buildEstimateParts().pieces.first { $0.label == "System prompt" }
        XCTAssertTrue(systemPiece?.content.contains("ACCOUNTED-SOUL") ?? false)
    }
}
