import XCTest
@testable import TurboSparkApp

/// Plugin hooks: discovery from an enabled plugin's hooks files and manifest
/// inline hooks, userConfig-derived option specs, and the trust gate that
/// keeps a plugin's shell commands unapproved until the user says so.
@MainActor
final class PluginHookTests: XCTestCase {
    var root: URL!
    var manager: PluginManager!
    var store: AppHookStore!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tsp-hooks-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        manager = PluginManager(
            turboSparkRoot: root, claudeRoot: root.appendingPathComponent("unused"),
            userEnableProvider: { [:] },
            projectEnableProvider: { _ in [:] },
            claudeEnableProvider: { [:] })
        store = AppHookStore()
        store.pluginProvider = { [unowned self] _ in
            manager.enabledPlugins(projectURL: nil)
        }
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    @discardableResult
    private func makePlugin(
        name: String, manifest: String, hooksJSON: String? = nil
    ) throws -> URL {
        let dir = root.appendingPathComponent(name, isDirectory: true)
        try write(manifest, to: dir.appendingPathComponent(".claude-plugin/plugin.json"))
        if let hooksJSON {
            try write(hooksJSON, to: dir.appendingPathComponent("hooks/hooks.json"))
        }
        return dir
    }

    func testEnabledPluginHooksAreDiscoveredAndNamed() throws {
        try makePlugin(
            name: "hooked",
            manifest: #"{"name": "hooked"}"#,
            hooksJSON: """
            {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo gated"}]}]}}
            """)
        store.refresh(projectDirectory: nil)
        let pluginHooks = store.hooks.filter { $0.sourceType == .plugin }
        XCTAssertEqual(pluginHooks.count, 1)
        XCTAssertEqual(pluginHooks.first?.pluginName, "hooked")
        XCTAssertEqual(pluginHooks.first?.matcher, "Bash")
        XCTAssertEqual(pluginHooks.first?.command, "echo gated")
    }

    func testDisabledPluginContributesNoHooks() throws {
        try makePlugin(
            name: "sleepy",
            manifest: #"{"name": "sleepy"}"#,
            hooksJSON: #"{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo no"}]}]}}"#)

        // Discover once with the plugin enabled, then flip the user enable
        // state through the SAME provider the discovery closure reads.
        store.refresh(projectDirectory: nil)
        XCTAssertEqual(store.hooks.filter { $0.sourceType == .plugin }.count, 1)

        manager = PluginManager(
            turboSparkRoot: root, claudeRoot: root.appendingPathComponent("unused"),
            userEnableProvider: { ["sleepy@turbospark": false] },
            projectEnableProvider: { _ in [:] },
            claudeEnableProvider: { [:] })
        store.pluginProvider = { [unowned self] _ in
            manager.enabledPlugins(projectURL: nil)
        }
        store.refresh(projectDirectory: nil)
        XCTAssertEqual(store.hooks.filter { $0.sourceType == .plugin }.count, 0)
    }

    func testInlineManifestHooksParseThroughTheSamePath() throws {
        try makePlugin(
            name: "inlined",
            manifest: """
            {"name": "inlined", "hooks": {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "echo inline"}]}]}}}
            """)
        store.refresh(projectDirectory: nil)
        let pluginHooks = store.hooks.filter { $0.sourceType == .plugin }
        XCTAssertEqual(pluginHooks.count, 1)
        XCTAssertEqual(pluginHooks.first?.command, "echo inline")
    }

    func testOptionSpecsComeFromThePluginUserConfigNotAGuess() throws {
        try makePlugin(
            name: "options",
            manifest: """
            {"name": "options", "userConfig": {"api_key": {"type": "string", "title": "API Key", "sensitive": true}, "strict": {"type": "boolean", "title": "Strict", "default": true}}}
            """,
            hooksJSON: #"{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo x"}]}]}}"#)
        store.refresh(projectDirectory: nil)
        let group = store.sourceGroups.first { $0.sourceType == .plugin }
        XCTAssertEqual(group?.pluginName, "options")
        XCTAssertEqual(Set(group?.optionSpecs.map(\.key) ?? []), ["api_key", "strict"])
        XCTAssertEqual(group?.optionSpecs.first { $0.key == "api_key" }?.isSensitive, true)
        XCTAssertEqual(group?.optionSpecs.first { $0.key == "strict" }?.defaultValue, "true")
    }

    func testPluginHooksAreUntrustedUntilApproved() throws {
        try makePlugin(
            name: "untrusted",
            manifest: #"{"name": "untrusted"}"#,
            hooksJSON: #"{"hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "rm -rf /"}]}]}}"#)
        store.refresh(projectDirectory: nil)
        let hook = store.hooks.first { $0.sourceType == .plugin }
        XCTAssertNotNil(hook)
        // THE TRUST GATE: discovered plugin hooks execute only after the
        // user approves their SHA-256 hash, exactly like hooks from a cloned
        // repository. This app's deliberate deviation from Claude Code.
        XCTAssertFalse(store.isHookTrusted(hook!))
    }

    func testShellQuotingStillHoldsForAPluginPathWithSpaces() {
        let quoted = AppHookExecutionEngine.shellQuoted("/tmp/My Plugins/p")
        XCTAssertEqual(quoted, "'/tmp/My Plugins/p'")
    }
}
