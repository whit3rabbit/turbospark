import Foundation
import XCTest
@testable import TurboSparkApp

/// ZP-18: the bundled `ios-simulator` and `android-emulator` skills.
///
/// Skill text guides the agent and enforces nothing, so the tests split the
/// work. The ENFORCED claims are pinned against the real permission code: every
/// device-control command a skill tells the agent to run must be scored high
/// risk, must not be auto-approvable, and must ask under `.auto`. The TEXT
/// claims (preflight first, no route-around, no network exposure, no
/// credentials, honest screenshot handoff) are pinned as clauses, and a
/// mutation check proves the clause scanners can fail.
final class BundledDeviceControlSkillsTests: XCTestCase {
    private let names = ["ios-simulator", "android-emulator"]

    private func skill(_ name: String) throws -> AppSkill {
        try XCTUnwrap(
            SkillManager.shared.discoverBundledSkills().first { $0.name == name },
            "\(name) must ship in TurboSparkAppResources/BundledSkills")
    }

    private func text(_ name: String) throws -> String {
        try String(contentsOf: skill(name).sourceURL, encoding: .utf8)
    }

    private static func flat(_ text: String) -> String {
        text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    override func tearDown() {
        for name in names { SkillManager.shared.setSkillEnabled(true, scope: .bundled, name: name) }
        SkillManager.shared.invalidateResolutionCache()
        super.tearDown()
    }

    // MARK: - Discovery

    func testBothSkillsShipAsBundledReadOnlyAndAreOffered() throws {
        for name in names {
            let found = try skill(name)
            XCTAssertEqual(found.scope, .bundled)
            XCTAssertFalse(found.skillDescription.isEmpty, name)
            XCTAssertFalse(SkillManager.shared.ownsSkill(found), "bundled skills are read-only")
        }
        let offered = SkillManager.shared.resolveEffectiveSkills(projectURL: nil)
        let listing = SkillManager.shared.formatSkillsWithinBudget(offered, contextWindowTokens: nil)
        for name in names { XCTAssertTrue(listing.contains("\(name) [Bundled]"), listing) }
    }

    func testUserSwitchOffHidesADeviceSkill() throws {
        SkillManager.shared.setSkillEnabled(false, scope: .bundled, name: "android-emulator")
        let offered = SkillManager.shared.resolveEffectiveSkills(projectURL: nil)
        let listing = SkillManager.shared.formatSkillsWithinBudget(offered, contextWindowTokens: nil)
        XCTAssertFalse(listing.contains("android-emulator ["), listing)
        XCTAssertTrue(listing.contains("ios-simulator [Bundled]"))
    }

    func testSkillsDoNotDependOnAPythonOrNodeRuntime() throws {
        for name in names {
            let body = try text(name).lowercased()
            for banned in ["python", "pip install", "npm install", "node -e", "node script", "pnpm ", "yarn "] {
                XCTAssertFalse(body.contains(banned), "\(name) mentions \(banned)")
            }
        }
    }

    // MARK: - Ladder position (enforced by code, not by the skill text)

    /// Every command line inside a fenced block, placeholders left in.
    static func fencedCommands(in text: String) -> [String] {
        var out: [String] = []
        var inside = false
        for line in text.components(separatedBy: "\n") {
            if line.hasPrefix("```") { inside.toggle(); continue }
            if inside, !line.trimmingCharacters(in: .whitespaces).isEmpty { out.append(line) }
        }
        return out
    }

    /// Fill `<placeholder>` tokens so the classifier sees a plausible command.
    private static func concrete(_ command: String) -> String {
        var out = command
        while let open = out.range(of: "<"), let close = out.range(of: ">", range: open.upperBound..<out.endIndex) {
            out.replaceSubrange(open.lowerBound..<close.upperBound, with: "X")
        }
        return out
    }

    private static let deviceHeads: Set<String> = [
        "xcrun", "xcodebuild", "adb", "emulator", "avdmanager", "sdkmanager", "gradlew", "open",
    ]

    func testEveryDeviceCommandIsHighRiskAndNeverAutoApprovable() throws {
        var checked = 0
        for name in names {
            for raw in Self.fencedCommands(in: try text(name)) {
                let command = Self.concrete(raw)
                guard let head = TerminalCommandClassifier.headCommand(command)
                    ?? command.split(separator: " ").first.map({ ($0 as NSString).lastPathComponent })
                else { continue }
                // Read-only probes of the host (`echo`, `ls`, `xcode-select`) may
                // run quietly. Everything that touches a device or SDK may not.
                guard Self.deviceHeads.contains(head) else { continue }
                checked += 1
                XCTAssertFalse(TerminalCommandClassifier.isAutoApprovable(command), "\(name): \(command)")
                let assessment = ToolRiskClassifier.assessRisk(
                    name: "run_command", arguments: ["command": command], projectURL: nil)
                XCTAssertEqual(assessment.level, .high, "\(name): \(command)")
            }
        }
        XCTAssertGreaterThan(checked, 20, "extraction found too few device commands")
    }

    /// The same commands must ask under Auto, the mode that promises quiet
    /// builds. Auto is where a missed allowlist entry would run silently.
    func testDeviceCommandsAskUnderAutoMode() {
        let project = AppProject(name: "p", permissions: AppProjectPermissions(mode: .auto))
        for command in [
            "xcrun simctl boot X", "xcrun simctl install X build/App.app",
            "xcodebuild -scheme S build", "adb -s emulator-5554 install -r a.apk",
            "emulator -avd Pixel -no-audio", "./gradlew assembleDebug",
        ] {
            let call = AppToolCall(
                name: "run_command", arguments: ["command": command], category: .terminal)
            let decision = AppToolPermissionEngine.evaluate(call: call, project: project, globalServers: [])
            guard case .ask = decision else { return XCTFail("\(command) must ask under auto, got \(decision)") }
        }
    }

    /// `gradle` runs the workspace's build scripts, so it asks like the other
    /// device-loop heads, and the skill still confines it to `assembleDebug`.
    func testGradleAsksAndTheSkillConfinesIt() throws {
        XCTAssertFalse(TerminalCommandClassifier.isAutoApprovable("gradle assembleDebug"))
        let body = Self.flat(try text("android-emulator"))
        XCTAssertTrue(body.contains("each call asks"))
        XCTAssertTrue(body.contains("only for `assembleDebug`, never for `installDebug`"))
        for raw in Self.fencedCommands(in: try text("android-emulator")) {
            XCTAssertFalse(raw.contains("installDebug"), raw)
        }
    }

    // MARK: - Forbidden verbs in runnable blocks

    /// Verbs the skills must never put in a command the agent is told to run:
    /// network exposure, credentials, destructive device operations.
    static let forbiddenInCommands: [String] = [
        "tcpip", "adb connect", "adb pair", " forward ", " reverse ", "adb root", "adb -s emulator-5554 root",
        "uninstall", "pm clear", "erase", "delete", "shutdown all", "devicectl", "ios-deploy",
        "allowProvisioningUpdates", "keychain", "security ", "sudo", "rm -rf", "curl", "wget",
        "-writable-system", "-http-proxy", "-grpc", "-ports ",
    ]

    static func violations(in text: String) -> [String] {
        let lines = fencedCommands(in: text)
        return forbiddenInCommands.flatMap { bad in lines.filter { $0.contains(bad) }.map { "\(bad): \($0)" } }
    }

    func testNoRunnableBlockExposesTheNetworkTouchesCredentialsOrDestroysState() throws {
        for name in names {
            XCTAssertEqual(Self.violations(in: try text(name)), [], name)
        }
    }

    /// Mutation check: the scanner must actually fail when a forbidden command
    /// is injected into a fenced block, and must ignore the same words in prose.
    func testViolationScannerFailsOnAnInjectedForbiddenCommand() throws {
        let clean = try text("android-emulator")
        XCTAssertEqual(Self.violations(in: clean), [])
        let mutated = clean.replacingOccurrences(
            of: "adb -s emulator-5554 wait-for-device",
            with: "adb -s emulator-5554 wait-for-device\nadb tcpip 5555")
        XCTAssertNotEqual(mutated, clean, "the mutation must have applied")
        XCTAssertFalse(Self.violations(in: mutated).isEmpty)
        // The prose naming the same verbs as forbidden is not a violation.
        XCTAssertTrue(clean.contains("adb tcpip"))
        let iosMutated = try text("ios-simulator").replacingOccurrences(
            of: "CODE_SIGNING_ALLOWED=NO build", with: "-allowProvisioningUpdates build")
        XCTAssertFalse(Self.violations(in: iosMutated).isEmpty)
    }

    func testAndroidCommandsAlwaysNameTheEmulatorSerial() throws {
        for raw in Self.fencedCommands(in: try text("android-emulator")) where raw.hasPrefix("adb ") {
            // `adb version` is the single host-only probe.
            if raw == "adb version" { continue }
            XCTAssertTrue(raw.hasPrefix("adb -s emulator-"), "adb without an emulator serial: \(raw)")
        }
    }

    // MARK: - Text clauses

    func testPreflightComesBeforeAnyActionAndStopsOnFailure() throws {
        for (name, firstAction, preflightHeading) in [
            ("ios-simulator", "## 2. Pick one simulator", "## 1. Preflight, before anything else"),
            ("android-emulator", "## 2. Emulator lifecycle", "## 1. Preflight, before anything else"),
        ] {
            let body = try text(name)
            let pre = try XCTUnwrap(body.range(of: preflightHeading), name)
            let act = try XCTUnwrap(body.range(of: firstAction), name)
            XCTAssertLessThan(pre.lowerBound, act.lowerBound, "\(name): preflight must precede action")
            XCTAssertTrue(Self.flat(body).contains("Do not attempt a build when preflight fails"), name)
        }
    }

    func testBothSkillsForbidRoutingAroundAPromptAndModeChanges() throws {
        for name in names {
            let body = Self.flat(try text(name))
            for clause in [
                "never route a refused command around the prompt",
                "Never ask the user to change the permission mode",
                "enforces nothing",
            ] {
                XCTAssertTrue(body.contains(clause), "\(name) lost: \(clause)")
            }
        }
    }

    func testScreenshotHandoffIsHonestAboutWhatTheModelCanSee() throws {
        for name in names {
            let body = Self.flat(try text(name))
            for clause in [
                "the PNG is NOT visible to you",
                "Do not claim you saw the screen",
                "send_user_file",
                "computer_request_access",
                "Without vision or Computer Use",
                "untrusted content",
            ] {
                XCTAssertTrue(body.contains(clause), "\(name) lost: \(clause)")
            }
        }
    }

    /// The handoff names real tools. Pin each name to the registry so a rename
    /// cannot leave the skill pointing at nothing.
    func testNamedToolsStillExist() {
        XCTAssertTrue(ComputerUseToolDefinitions.toolNames.contains("computer_screenshot"))
        XCTAssertTrue(ComputerUseToolDefinitions.toolNames.contains("computer_request_access"))
        XCTAssertEqual(AppToolCatalog.category(for: "send_user_file"), .fileRead)
    }

    func testIosSkillStatesItsInteractionLimitAndSimulatorOnlyScope() throws {
        let body = Self.flat(try text("ios-simulator"))
        XCTAssertTrue(body.contains("It cannot tap, swipe or type"))
        XCTAssertTrue(body.contains("Simulators only"))
        XCTAssertTrue(body.contains("CODE_SIGNING_ALLOWED=NO"))
    }

    func testAndroidSkillScopesToEmulatorsAndNamesTheNetworkBans() throws {
        let body = Self.flat(try text("android-emulator"))
        XCTAssertTrue(body.contains("Emulators only"))
        for verb in ["adb tcpip", "adb connect", "adb forward", "adb reverse"] {
            XCTAssertTrue(body.contains(verb), "the ban list lost \(verb)")
        }
    }
}
