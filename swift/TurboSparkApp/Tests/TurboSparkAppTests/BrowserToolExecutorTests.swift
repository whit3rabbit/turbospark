import XCTest

@testable import TurboSparkApp

final class BrowserToolExecutorTests: XCTestCase {
    func testCancellationRemainsDistinctFromBrowserFailure() async throws {
        let origin = try XCTUnwrap(BrowserOrigin(origin: "https://example.test"))
        let project = AppProject(
            name: "Cancellation",
            permissions: AppProjectPermissions(
                browser: .allow,
                browserOriginAllowlist: [origin.canonicalString]))
        let call = AppToolCall(name: "browser_snapshot", category: .browser)
        let runtime = BrowserToolRuntime(
            availability: enabledAvailability(),
            permissionContext: BrowserPermissionContext(origin: origin),
            perform: { _ in throw CancellationError() })

        let outcome = await BrowserToolExecutor.execute(call: call, in: project, runtime: runtime)
        XCTAssertEqual(outcome, .cancelled)
    }

    func testAlreadyCancelledToolDoesNotInvokeBrowserBackend() async throws {
        let origin = try XCTUnwrap(BrowserOrigin(origin: "https://example.test"))
        let project = AppProject(
            name: "Cancellation",
            permissions: AppProjectPermissions(
                browser: .allow,
                browserOriginAllowlist: [origin.canonicalString]))
        let recorder = BrowserCommandRecorder()
        let runtime = BrowserToolRuntime(
            availability: enabledAvailability(),
            permissionContext: BrowserPermissionContext(origin: origin),
            perform: { command in
                await recorder.append(command)
                throw BrowserControlError.engineCrashed
            })
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return await BrowserToolExecutor.execute(
                call: AppToolCall(name: "browser_snapshot", category: .browser),
                in: project,
                runtime: runtime)
        }

        let outcome = await task.value
        let commands = await recorder.commands()
        XCTAssertEqual(outcome, .cancelled)
        XCTAssertTrue(commands.isEmpty)
    }

    func testCatalogTrimsDisabledStaleUnsupportedAndScreenshotTools() {
        let names = Set(BrowserToolDefinitions.all.map { $0.function.name })
        XCTAssertEqual(names.count, 8)
        XCTAssertTrue(AppToolCatalog.allTools.allSatisfy { !names.contains($0.function.name) })

        let allCommands = BrowserToolAvailability(
            isEnabled: true,
            manifest: manifest(supporting: Set(BrowserControlCommandKind.allCases)))
        let visible = Set(AppToolCatalog.tools(
            for: .coder,
            browserAvailability: allCommands).map { $0.function.name })
        XCTAssertTrue(visible.contains("browser_navigate"))
        XCTAssertTrue(visible.contains("browser_snapshot"))
        XCTAssertFalse(visible.contains("browser_screenshot"), "A text-only provider must not receive screenshot tools.")

        let screenshotReady = BrowserToolAvailability(
            isEnabled: true,
            manifest: manifest(supporting: Set(BrowserControlCommandKind.allCases)),
            mediaCapability: .imageBearingToolResults(maximumEncodedByteCount: 524_288))
        XCTAssertTrue(AppToolCatalog.allTools(browserAvailability: screenshotReady)
            .contains { $0.function.name == "browser_screenshot" })
        XCTAssertEqual(screenshotReady.mediaCapability.maximumEncodedByteCount, 524_288)

        let unsupported = BrowserToolAvailability(
            isEnabled: true,
            manifest: manifest(supporting: [.navigate]))
        XCTAssertEqual(
            BrowserToolDefinitions.tools(availableFor: unsupported).map { $0.function.name },
            ["browser_navigate"])

        let stale = BrowserToolAvailability(
            isEnabled: true,
            manifest: manifest(supporting: Set(BrowserControlCommandKind.allCases), version: 0),
            mediaCapability: .imageBearingToolResults())
        XCTAssertTrue(AppToolCatalog.tools(for: .autonomous, browserAvailability: stale)
            .allSatisfy { !names.contains($0.function.name) })
    }

    func testVocabularyRegistersEveryBrowserToolAsImplementedAndBrowserScoped() {
        for name in BrowserToolDefinitions.commandKindsByToolName.keys {
            XCTAssertTrue(AppToolRegistry.isImplemented(name), "Missing executor registration for \(name).")
            XCTAssertEqual(AppToolCatalog.category(for: name), .browser)
            XCTAssertEqual(AppToolRegistry.category(for: name), .browser)
        }
    }

    func testCommandProjectionValidatesTypesEnumsTimeoutsAndExactKeys() {
        XCTAssertEqual(
            BrowserToolExecutor.command(
                for: "browser_navigate",
                arguments: ["url": "https://example.test/path", "wait_until": "committed"]),
            .navigate(url: "https://example.test/path", waitUntil: .committed, timeoutSeconds: nil))
        XCTAssertEqual(
            BrowserToolExecutor.command(for: "browser_type", arguments: [
                "reference": "e7", "text": "hello", "submit": "false",
            ]),
            .type(reference: "e7", text: "hello", submit: false, timeoutSeconds: nil))
        XCTAssertEqual(
            BrowserToolExecutor.command(for: "browser_scroll", arguments: ["direction": "down"]),
            .scroll(reference: nil, direction: .down, amount: 600))
        XCTAssertEqual(
            BrowserToolExecutor.command(for: "browser_wait", arguments: [
                "reference": "e7", "condition": "visible", "timeout_seconds": "2.5",
            ]),
            .waitFor(target: .element(reference: "e7", condition: .visible), timeoutSeconds: 2.5))

        XCTAssertNil(BrowserToolExecutor.command(for: "browser_type", arguments: [
            "reference": "e7", "text": "hello", "submit": "yes",
        ]))
        XCTAssertEqual(
            BrowserToolExecutor.command(for: "browser_type", arguments: [
                "reference": "e7", "text": "hello", "submit": "1",
            ]),
            .type(reference: "e7", text: "hello", submit: true, timeoutSeconds: nil))
        XCTAssertNil(BrowserToolExecutor.command(for: "browser_navigate", arguments: [
            "url": "https://example.test", "timeout_seconds": "61",
        ]))
        XCTAssertNil(BrowserToolExecutor.command(for: "browser_wait", arguments: [
            "load_state": "finished", "reference": "e7", "condition": "visible",
        ]))
        XCTAssertNil(BrowserToolExecutor.command(for: "browser_click", arguments: [
            "reference": "e7", "arbitrary": "forged",
        ]))
        XCTAssertNil(BrowserToolExecutor.command(for: "browser_navigate", arguments: [
            "url": "file:///etc/passwd",
        ]))
        XCTAssertNil(BrowserToolExecutor.command(for: "browser_navigate", arguments: [
            "url": "https://user:secret@example.test/",
        ]))
    }

    func testExecutorUsesExactProjectOriginAndIgnoresForgedCallCategory() async throws {
        let allowedOrigin = try XCTUnwrap(BrowserOrigin(origin: "https://allowed.example"))
        let allowedProject = AppProject(
            name: "Allowed",
            permissions: AppProjectPermissions(
                browser: .allow,
                browserOriginAllowlist: ["https://allowed.example"]))
        let otherProject = AppProject(
            name: "Other",
            permissions: AppProjectPermissions(browser: .allow))
        let recorder = BrowserCommandRecorder()
        let runtime = BrowserToolRuntime(
            availability: enabledAvailability(),
            permissionContext: BrowserPermissionContext(origin: allowedOrigin),
            perform: { command in
                await recorder.append(command)
                guard case .navigate(let url, _, _) = command else {
                    throw BrowserControlError.unsupported(command: command.kind)
                }
                let request = BrowserControlRequest(command: command)
                return BrowserControlResult(
                    request: request,
                    value: .navigated(BrowserNavigationResult(url: url, reachedState: .finished)),
                    durationMilliseconds: 5)
            })
        let forgedCategoryCall = AppToolCall(
            name: "browser_navigate",
            arguments: ["url": "https://allowed.example/account?token=private"],
            category: .fileRead,
            riskAssessment: .safe)

        let allowed = await BrowserToolExecutor.execute(
            call: forgedCategoryCall, in: allowedProject, runtime: runtime)
        guard case .completed(let result) = allowed else {
            return XCTFail("Exact project origin grant should permit the call, got \(allowed).")
        }
        let commandsAfterAllowedCall = await recorder.commands()
        XCTAssertEqual(commandsAfterAllowedCall.count, 1)
        XCTAssertEqual(
            BrowserToolExecutor.modelOutput(for: result),
            "Navigated to https://allowed.example; reached load state finished.")
        XCTAssertFalse(BrowserToolExecutor.modelOutput(for: result).contains("token=private"))

        let other = await BrowserToolExecutor.execute(
            call: forgedCategoryCall, in: otherProject, runtime: runtime)
        guard case .pendingApproval(let assessment, _) = other else {
            return XCTFail("A different project must ask, got \(other).")
        }
        XCTAssertEqual(assessment.category, .browser)
        let commandsAfterOtherProject = await recorder.commands()
        XCTAssertEqual(commandsAfterOtherProject.count, 1, "An ask must not invoke the browser backend.")
    }

    func testExplicitCurrentActionApprovalRunsOnlyTheRequestedBrowserAction() async throws {
        let origin = try XCTUnwrap(BrowserOrigin(origin: "https://example.test"))
        let project = AppProject(
            name: "Ask first",
            permissions: AppProjectPermissions(browser: .allow)
        )
        let recorder = BrowserCommandRecorder()
        let runtime = BrowserToolRuntime(
            availability: enabledAvailability(),
            permissionContext: BrowserPermissionContext(origin: origin),
            perform: { command in
                await recorder.append(command)
                guard case .navigate(let url, _, _) = command else {
                    throw BrowserControlError.unsupported(command: command.kind)
                }
                return BrowserControlResult(
                    request: BrowserControlRequest(command: command),
                    value: .navigated(BrowserNavigationResult(url: url, reachedState: .finished)),
                    durationMilliseconds: 2)
            })
        let call = AppToolCall(
            name: "browser_navigate",
            arguments: ["url": "https://example.test/page"],
            category: .browser,
            riskAssessment: .safe
        )

        let pending = await BrowserToolExecutor.execute(call: call, in: project, runtime: runtime)
        guard case .pendingApproval = pending else {
            return XCTFail("An ungranted origin should ask before execution, got \(pending).")
        }
        let commandsAfterAsk = await recorder.commands()
        XCTAssertTrue(commandsAfterAsk.isEmpty)

        let approved = await BrowserToolExecutor.execute(
            call: call,
            in: project,
            runtime: runtime,
            currentActionApproved: true
        )
        guard case .completed = approved else {
            return XCTFail("A matching one-time approval should run the action, got \(approved).")
        }
        let commandsAfterApproval = await recorder.commands()
        XCTAssertEqual(commandsAfterApproval.count, 1)
    }

    func testExecutorPassesExactOneTimePermissionContextToRuntime() async throws {
        let origin = try XCTUnwrap(BrowserOrigin(origin: "https://example.test"))
        let project = AppProject(
            name: "Ask first",
            permissions: AppProjectPermissions(browser: .allow)
        )
        let recorder = BrowserPermissionContextRecorder()
        let runtime = BrowserToolRuntime(
            availability: enabledAvailability(),
            permissionContext: BrowserPermissionContext(origin: origin),
            performWithPermissionContext: { command, context in
                await recorder.append(context)
                guard case .navigate(let url, _, _) = command else {
                    throw BrowserControlError.unsupported(command: command.kind)
                }
                return BrowserControlResult(
                    request: BrowserControlRequest(command: command),
                    value: .navigated(BrowserNavigationResult(url: url, reachedState: .finished)),
                    durationMilliseconds: 2)
            })
        let call = AppToolCall(
            name: "browser_navigate",
            arguments: ["url": "https://example.test/page"],
            category: .browser
        )

        let outcome = await BrowserToolExecutor.execute(
            call: call,
            in: project,
            runtime: runtime,
            currentActionApproved: true
        )

        guard case .completed = outcome else {
            return XCTFail("A matching one-time approval should execute, got \(outcome).")
        }
        let snapshot = await recorder.snapshot()
        let observed = try XCTUnwrap(snapshot)
        XCTAssertEqual(observed.origin, "https://example.test")
        XCTAssertTrue(observed.currentActionApproved)
    }

    func testExecutorRejectsDisabledUnsupportedAndDeniedCallsWithoutExecution() async {
        let command = AppToolCall(name: "browser_snapshot")
        let deniedProject = AppProject(
            name: "Denied",
            permissions: AppProjectPermissions(browser: .deny))
        let recorder = BrowserCommandRecorder()
        let enabledRuntime = BrowserToolRuntime(
            availability: enabledAvailability(),
            permissionContext: BrowserPermissionContext(
                origin: BrowserOrigin(origin: "https://example.test")),
            perform: { command in
                await recorder.append(command)
                throw BrowserControlError.engineCrashed
            })
        let denied = await BrowserToolExecutor.execute(
            call: command, in: deniedProject, runtime: enabledRuntime)
        guard case .denied = denied else { return XCTFail("Category deny must remain authoritative.") }

        let disabledRuntime = BrowserToolRuntime(
            availability: .disabled,
            permissionContext: nil,
            perform: { command in
                await recorder.append(command)
                throw BrowserControlError.engineCrashed
            })
        let disabled = await BrowserToolExecutor.execute(
            call: command, in: deniedProject, runtime: disabledRuntime)
        guard case .unsupported(command: .readState) = disabled else {
            return XCTFail("A forged disabled tool call must be rejected, got \(disabled).")
        }

        let unsupportedRuntime = BrowserToolRuntime(
            availability: BrowserToolAvailability(
                isEnabled: true,
                manifest: manifest(supporting: [.navigate])),
            permissionContext: nil,
            perform: { command in
                await recorder.append(command)
                throw BrowserControlError.engineCrashed
            })
        let unsupported = await BrowserToolExecutor.execute(
            call: command, in: deniedProject, runtime: unsupportedRuntime)
        guard case .unsupported(command: .readState) = unsupported else {
            return XCTFail("A forged unsupported tool call must be rejected, got \(unsupported).")
        }

        let staleRuntime = BrowserToolRuntime(
            availability: BrowserToolAvailability(
                isEnabled: true,
                manifest: manifest(supporting: Set(BrowserControlCommandKind.allCases), version: 0)),
            permissionContext: nil,
            perform: { command in
                await recorder.append(command)
                throw BrowserControlError.engineCrashed
            })
        let stale = await BrowserToolExecutor.execute(
            call: command, in: deniedProject, runtime: staleRuntime)
        guard case .unsupported(command: .readState) = stale else {
            return XCTFail("A forged stale tool call must be rejected, got \(stale).")
        }

        let screenshotRuntime = BrowserToolRuntime(
            availability: BrowserToolAvailability(
                isEnabled: true,
                manifest: manifest(supporting: Set(BrowserControlCommandKind.allCases)),
                mediaCapability: .textOnly),
            permissionContext: nil,
            perform: { command in
                await recorder.append(command)
                throw BrowserControlError.engineCrashed
            })
        let forgedScreenshot = await BrowserToolExecutor.execute(
            call: AppToolCall(name: "browser_screenshot", arguments: ["full_page": "false"]),
            in: deniedProject,
            runtime: screenshotRuntime)
        guard case .unsupported(command: .screenshot) = forgedScreenshot else {
            return XCTFail("A forged screenshot call must be refused without image support, got \(forgedScreenshot).")
        }

        let commands = await recorder.commands()
        XCTAssertTrue(commands.isEmpty)
    }

    private func enabledAvailability() -> BrowserToolAvailability {
        BrowserToolAvailability(
            isEnabled: true,
            manifest: manifest(supporting: Set(BrowserControlCommandKind.allCases)))
    }

    private func manifest(
        supporting commands: Set<BrowserControlCommandKind>,
        version: Int = BrowserControlProtocol.currentVersion
    ) -> BrowserBackendManifest {
        BrowserBackendManifest(
            backendIdentifier: "test-browser",
            commandSurfaceVersion: version,
            supportedCommands: commands)
    }
}

private actor BrowserCommandRecorder {
    private var storedCommands: [BrowserControlCommand] = []

    func append(_ command: BrowserControlCommand) {
        storedCommands.append(command)
    }

    func commands() -> [BrowserControlCommand] {
        storedCommands
    }
}

private actor BrowserPermissionContextRecorder {
    struct Snapshot: Equatable, Sendable {
        let origin: String?
        let currentActionApproved: Bool
    }

    private var stored: Snapshot?

    func append(_ context: BrowserPermissionContext) {
        stored = Snapshot(
            origin: context.origin?.canonicalString,
            currentActionApproved: context.currentActionApproved)
    }

    func snapshot() -> Snapshot? {
        stored
    }
}
