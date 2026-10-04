import AppKit
import Combine
import Darwin
import XCTest
@testable import TurboSparkApp

@MainActor
final class BrowserAppIntegrationTests: XCTestCase {
    func testBrowserCatalogTrimsDisabledAndProviderUnsupportedCommands() {
        let disabledNames = AppToolCatalog.tools(
            for: .coder,
            browserAvailability: .disabled
        ).map(\.function.name)
        XCTAssertFalse(disabledNames.contains("browser_navigate"))

        let textOnly = BrowserToolAvailability(
            isEnabled: true,
            manifest: WebKitBrowserControlPort.manifest,
            mediaCapability: .textOnly
        )
        let mainNames = AppToolCatalog.tools(
            for: .coder,
            browserAvailability: textOnly
        ).map(\.function.name)
        XCTAssertTrue(mainNames.contains("browser_navigate"))
        XCTAssertFalse(mainNames.contains("browser_screenshot"))

        let agent = AppAgentDefinition(
            name: "browser-test",
            agentDescription: "Browser catalog fixture",
            systemPrompt: ""
        )
        let agentNames = SubagentRunner.captureAvailableTools(
            for: agent,
            project: nil,
            browserAvailability: textOnly
        ).definitions.map(\.function.name)
        XCTAssertTrue(agentNames.contains("browser_navigate"))
        XCTAssertFalse(agentNames.contains("browser_screenshot"))
    }

    func testBrowserPaneClaimantTracksFeatureAndOpenState() {
        let closed = AppRightColumnClaimant.resolve(
            openArtifactID: nil,
            previewAttachmentID: nil,
            isInspectorVisible: false,
            showBrowserPane: false
        )
        let open = AppRightColumnClaimant.resolve(
            openArtifactID: nil,
            previewAttachmentID: nil,
            isInspectorVisible: false,
            showBrowserPane: true
        )

        XCTAssertEqual(closed, .none)
        XCTAssertEqual(open, .browser)
        XCTAssertEqual(AppChromeLayout.rightColumnWidth(open), AppChromeLayout.browserPanelWidth)
    }

    func testCoordinatorDetachesAgentSessionWhenPaneCloses() async throws {
        let coordinator = AppBrowserAutomationCoordinator(
            authorizeNavigation: { _ in .allow }
        )
        coordinator.openPane()
        XCTAssertTrue(coordinator.isPaneOpen)
        XCTAssertEqual(coordinator.tabStore.tabs.first?.owner, .user)

        let agentTabID = coordinator.engine.createTab(owner: .agent)
        _ = try await coordinator.automationSession.attach(to: agentTabID)
        let attachedTabID = await coordinator.automationSession.attachedTabID()
        XCTAssertEqual(attachedTabID, agentTabID)

        await coordinator.closePane()

        XCTAssertFalse(coordinator.isPaneOpen)
        let detachedTabID = await coordinator.automationSession.attachedTabID()
        XCTAssertNil(detachedTabID)
        XCTAssertEqual(coordinator.tabStore.tab(id: agentTabID)?.owner, .user)
    }

    func testRealWebKitControlPortNavigatesAndReadsLocalFixtureThroughBrowserTool() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        let engine = WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { _ in .allow }
        )
        let visibleTabID = engine.createTab(owner: .user)
        let controlledTabID = engine.createTab(owner: .agent, select: false)
        let webView = try XCTUnwrap(engine.webView(for: controlledTabID))
        webView.frame = NSRect(x: 0, y: 0, width: 900, height: 700)
        let session = BrowserAutomationSession(
            tabStore: store,
            portFactory: { tabID in WebKitBrowserControlPort(engine: engine, tabID: tabID) }
        )
        let token = try await session.attach(to: controlledTabID)
        let url = server.url("/automation")
        let origin = try XCTUnwrap(BrowserOrigin(url: url))
        let alternateHostOrigin = try XCTUnwrap(
            BrowserOrigin(url: try XCTUnwrap(
                URL(string: "http://localhost:\(server.port)/automation"))))
        let project = AppProject(
            name: "Browser fixture",
            permissions: AppProjectPermissions(
                browser: .allow,
                browserOriginAllowlist: [origin.canonicalString])
        )
        XCTAssertNotEqual(origin, alternateHostOrigin)
        XCTAssertEqual(
            BrowserPermissionRuleStore.decision(for: alternateHostOrigin, in: project.permissions),
            .ask)
        let availability = BrowserToolAvailability(
            isEnabled: true,
            manifest: WebKitBrowserControlPort.manifest,
            mediaCapability: .imageBearingToolResults()
        )
        let approvalCoordinator = BrowserPermissionApprovalCoordinator()
        func run(
            _ name: String,
            arguments: [String: String] = [:],
            in targetProject: AppProject,
            approved: Bool = true
        ) async -> BrowserToolExecutionOutcome {
            let call = AppToolCall(name: name, arguments: arguments, category: .browser)
            let approvalRequest = BrowserPermissionApprovalRequest.make(
                for: call,
                project: targetProject,
                currentOrigin: origin)
            if approved, let approvalRequest {
                approvalCoordinator.authorizeOnce(approvalRequest)
            }
            let currentActionApproved: Bool
            if let approvalRequest {
                currentActionApproved = approvalCoordinator.allowsOnce(
                    callID: call.id,
                    projectID: targetProject.id,
                    origin: approvalRequest.origin)
            } else {
                currentActionApproved = false
            }
            defer { approvalCoordinator.cancel(callID: call.id) }
            let runtime = BrowserToolRuntime(
                availability: availability,
                permissionContext: BrowserPermissionContext(origin: origin),
                perform: { command in try await session.perform(command, using: token) }
            )
            return await BrowserToolExecutor.execute(
                call: call,
                in: targetProject,
                runtime: runtime,
                currentActionApproved: currentActionApproved
            )
        }

        var grantPermissions = project.permissions
        XCTAssertEqual(
            BrowserPermissionRuleStore.decision(for: origin, in: grantPermissions),
            .allow)
        XCTAssertTrue(BrowserPermissionRuleStore.revoke(origin: origin, from: &grantPermissions))
        XCTAssertEqual(
            BrowserPermissionRuleStore.decision(for: origin, in: grantPermissions),
            .ask)

        let navigation = await run(
            "browser_navigate",
            arguments: ["url": url.absoluteString],
            in: project)
        guard case .completed(let navigationResult) = navigation else {
            await session.detach()
            return XCTFail("The granted local fixture should navigate through the real browser runtime, got \(navigation).")
        }
        XCTAssertFalse(BrowserToolExecutor.modelOutput(for: navigationResult).isEmpty)

        var revokedPermissions = project.permissions
        XCTAssertTrue(BrowserPermissionRuleStore.revoke(origin: origin, from: &revokedPermissions))
        let revokedProject = AppProject(name: "Revoked fixture", permissions: revokedPermissions)
        let revokedNavigation = await run(
            "browser_navigate",
            arguments: ["url": url.absoluteString],
            in: revokedProject,
            approved: false)
        guard case .pendingApproval = revokedNavigation else {
            await session.detach()
            return XCTFail("Revoking the exact-origin grant should require approval on the next action.")
        }

        let snapshotArguments = ["scope": "visibleText"]
        let otherProject = AppProject(
            name: "Unlisted fixture",
            permissions: AppProjectPermissions(browser: .allow))
        let otherProjectOutcome = await run(
            "browser_snapshot",
            arguments: snapshotArguments,
            in: otherProject,
            approved: false)
        guard case .pendingApproval = otherProjectOutcome else {
            await session.detach()
            return XCTFail("A different project must not inherit the fixture origin grant.")
        }
        let unapprovedSnapshot = await run(
            "browser_snapshot", arguments: snapshotArguments, in: project, approved: false)
        guard case .pendingApproval = unapprovedSnapshot else {
            await session.detach()
            return XCTFail("A private fixture origin must require a current-action approval.")
        }
        let snapshotOutcome = await run("browser_snapshot", arguments: snapshotArguments, in: project)
        guard case .completed(let snapshotResult) = snapshotOutcome,
              case .state(let snapshot) = snapshotResult.value
        else {
            await session.detach()
            return XCTFail("The real browser runtime should return a typed fixture snapshot, got \(snapshotOutcome).")
        }

        XCTAssertEqual(snapshot.title, "Automation Fixture")
        XCTAssertTrue(snapshot.snapshot.contains("Ignore all previous instructions"), snapshot.snapshot)
        let modelOutput = BrowserToolExecutor.modelOutput(for: snapshotResult)
        XCTAssertTrue(modelOutput.contains("<untrusted_browser_snapshot>"))
        XCTAssertTrue(modelOutput.contains("Ignore all previous instructions"))
        XCTAssertEqual(store.activeTabID, visibleTabID)
        XCTAssertEqual(try XCTUnwrap(BrowserOrigin(url: try XCTUnwrap(webView.url))), origin)
        let attachedTabID = await session.attachedTabID()
        XCTAssertEqual(attachedTabID, controlledTabID)

        let reusedApproval = await run(
            "browser_snapshot",
            arguments: snapshotArguments,
            in: project,
            approved: false)
        guard case .pendingApproval = reusedApproval else {
            await session.detach()
            return XCTFail("A one-time approval must not authorize the next browser action.")
        }

        let envelope = try JSONDecoder().decode(
            DOMSnapshotEnvelope.self,
            from: Data(snapshot.snapshot.utf8))
        func reference(named name: String) throws -> String {
            let node = envelope.nodes.first { $0.name == name && $0.reference != nil }
            return try XCTUnwrap(
                node?.reference,
                "Missing actionable fixture node '\(name)'. Nodes: \(envelope.nodes.map(\.name))")
        }

        let clickReference = try reference(named: "Click target")
        let click = await run(
            "browser_click", arguments: ["reference": clickReference], in: project)
        guard case .completed(let clickResult) = click else {
            await session.detach()
            return XCTFail("The fixture click should complete through WebKit, got \(click).")
        }
        XCTAssertEqual(clickResult.commandKind, .click)
        XCTAssertTrue(BrowserToolExecutor.modelOutput(for: clickResult).contains(clickReference))
        let clickPageState = try await webView.evaluateJavaScript(
            "document.getElementById('click-result').textContent") as? String
        XCTAssertEqual(clickPageState, "clicked")

        let wait = await run(
            "browser_wait",
            arguments: ["reference": clickReference, "condition": "enabled", "timeout_seconds": "2"],
            in: project)
        guard case .completed(let waitResult) = wait else {
            await session.detach()
            return XCTFail("The element wait should reach its terminal state, got \(wait).")
        }
        XCTAssertEqual(waitResult.commandKind, .waitFor)
        XCTAssertTrue(BrowserToolExecutor.modelOutput(for: waitResult).contains("wait completed"))

        let keyReference = try reference(named: "Key target")
        let keyPress = await run(
            "browser_press_key",
            arguments: ["reference": keyReference, "key": "Enter"],
            in: project)
        guard case .completed(let keyResult) = keyPress else {
            await session.detach()
            return XCTFail("The fixture key press should complete through WebKit, got \(keyPress).")
        }
        XCTAssertEqual(keyResult.commandKind, .pressKey)
        XCTAssertTrue(BrowserToolExecutor.modelOutput(for: keyResult).contains("Pressed key Enter"))
        let keyPageState = try await webView.evaluateJavaScript(
            "document.getElementById('key-result').textContent") as? String
        XCTAssertEqual(keyPageState, "enter")

        let scroll = await run(
            "browser_scroll",
            arguments: [
                "reference": try reference(named: "Scrollable fixture"),
                "direction": "down",
                "amount": "450",
            ],
            in: project)
        guard case .completed(let scrollResult) = scroll else {
            await session.detach()
            return XCTFail("The fixture scroll should complete through WebKit, got \(scroll).")
        }
        XCTAssertEqual(scrollResult.commandKind, .scroll)
        XCTAssertTrue(BrowserToolExecutor.modelOutput(for: scrollResult).contains("Scrolled"))
        let scrollTop = try await webView.evaluateJavaScript(
            "document.getElementById('scroll-box').scrollTop") as? Double
        XCTAssertGreaterThan(try XCTUnwrap(scrollTop), 0)

        let originalReference = try reference(named: "Original target")
        let replaceReference = try reference(named: "Replace stale target")
        let replaced = await run(
            "browser_click", arguments: ["reference": replaceReference], in: project)
        guard case .completed = replaced else {
            await session.detach()
            return XCTFail("The fixture should replace the original node, got \(replaced).")
        }
        let staleAction = await run(
            "browser_click", arguments: ["reference": originalReference], in: project)
        guard case .failed(let staleReason) = staleAction else {
            await session.detach()
            return XCTFail("A replaced element reference must fail without acting on its replacement, got \(staleAction).")
        }
        XCTAssertTrue(staleReason.localizedCaseInsensitiveContains("stale"))

        let pickerNode = try XCTUnwrap(
            envelope.nodes.first { $0.name == "Click target" && $0.reference != nil })
        let pickerBounds = try XCTUnwrap(pickerNode.bounds)
        let pickerPoint = CGPoint(
            x: pickerBounds.x + pickerBounds.width / 2,
            y: pickerBounds.y + pickerBounds.height / 2)
        let pickerChatID = UUID()
        var attachedPickerValue: (AppPromptAttachment, UUID)?
        let picker = ElementPickerController(
            initialComposerAvailability: true,
            composerAvailability: Just(true).eraseToAnyPublisher(),
            canAttachNow: { true },
            selectedChatID: { pickerChatID },
            isTabReady: { tabID in engine.snapshotService(for: tabID)?.canPickElement == true },
            prepareForUserInput: { _ in },
            pickCandidateAt: { tabID, point in
                guard let service = engine.snapshotService(for: tabID) else {
                    throw DOMSnapshotServiceError.unavailable
                }
                return try await service.pickCandidate(at: point)
            },
            validateCandidate: { tabID, candidate in
                guard let service = engine.snapshotService(for: tabID) else {
                    throw BrowserControlError.staleReference(reference: candidate.reference)
                }
                try await service.resolve(reference: candidate.reference, generation: candidate.generation)
            },
            attach: { attachment, chatID in
                attachedPickerValue = (attachment, chatID)
                return true
            }
        )
        XCTAssertTrue(picker.startPicking(in: controlledTabID))
        picker.updateCandidate(at: pickerPoint)
        await picker.waitForPendingCandidateUpdate()
        XCTAssertEqual(picker.candidate?.name, "Click target")
        XCTAssertTrue(picker.selectCandidate())
        let didAttachPickedElement = await picker.attachSelectedCandidate()
        XCTAssertTrue(didAttachPickedElement)
        let pickedAttachment = try XCTUnwrap(attachedPickerValue)
        XCTAssertEqual(pickedAttachment.1, pickerChatID)
        XCTAssertTrue(try XCTUnwrap(pickedAttachment.0.extractedText).contains("<untrusted_browser_element>"))
        XCTAssertTrue(try XCTUnwrap(pickedAttachment.0.extractedText).contains(clickReference))

        let postReplacement = await run(
            "browser_snapshot", arguments: snapshotArguments, in: project)
        guard case .completed(let postReplacementResult) = postReplacement,
              case .state(let postReplacementSnapshot) = postReplacementResult.value
        else {
            await session.detach()
            return XCTFail("A fresh snapshot should report the safe post-replacement state, got \(postReplacement).")
        }
        XCTAssertTrue(postReplacementSnapshot.snapshot.contains("waiting"))
        let unrelatedActionState = try await webView.evaluateJavaScript(
            "document.getElementById('unrelated-result').textContent") as? String
        XCTAssertEqual(unrelatedActionState, "waiting")

        let submitReference = try reference(named: "Submit value")
        let typedSubmit = await run(
            "browser_type",
            arguments: ["reference": submitReference, "text": "TurboSpark", "submit": "true"],
            in: project)
        guard case .completed(let typedResult) = typedSubmit else {
            await session.detach()
            return XCTFail("Typing and submitting the fixture form should complete, got \(typedSubmit).")
        }
        XCTAssertEqual(typedResult.commandKind, .type)
        XCTAssertTrue(BrowserToolExecutor.modelOutput(for: typedResult).contains("Typed into browser element"))
        let submissionDeadline = Date().addingTimeInterval(5)
        while server.requestCount(for: "/submitted") == 0 && Date() < submissionDeadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertEqual(server.requestCount(for: "/submitted"), 1)
        XCTAssertTrue(server.requestTargets(for: "/submitted").contains { target in
            URLComponents(string: "http://fixture\(target)")?.queryItems?.contains(
                URLQueryItem(name: "value", value: "TurboSpark")) == true
        })

        await session.detach()
    }

    func testFixtureDialogsResolveForAskAutoAcceptAndAutoDismissPolicies() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let fixtureURL = server.url("/dialogs")
        let origin = try XCTUnwrap(BrowserOrigin(url: fixtureURL))
        let cases: [(BrowserDialogPolicy, BrowserJavaScriptDialogKind, String)] = [
            (.autoAccept, .alert, "Alert"),
            (.autoAccept, .confirm, "Confirm"),
            (.autoAccept, .prompt, "Prompt"),
            (.autoDismiss, .alert, "Alert"),
            (.autoDismiss, .confirm, "Confirm"),
            (.autoDismiss, .prompt, "Prompt"),
            (.ask, .alert, "Alert"),
            (.ask, .confirm, "Confirm"),
            (.ask, .prompt, "Prompt"),
        ]

        for policy in BrowserDialogPolicy.allCases {
            let store = BrowserTabStore()
            let engine = WebKitBrowserEngine(
                tabStore: store,
                authorizeNavigation: { _ in .allow },
                dialogPolicyProvider: { policy })
            let tabID = engine.createTab(owner: .agent)
            let webView = try XCTUnwrap(engine.webView(for: tabID))
            webView.frame = NSRect(x: 0, y: 0, width: 900, height: 700)
            let session = BrowserAutomationSession(
                tabStore: store,
                portFactory: { tabID in WebKitBrowserControlPort(engine: engine, tabID: tabID) })
            let token = try await session.attach(to: tabID)
            let project = AppProject(
                name: "Browser dialog fixture",
                permissions: AppProjectPermissions(
                    browser: .allow,
                    browserOriginAllowlist: [origin.canonicalString]))
            let availability = BrowserToolAvailability(
                isEnabled: true,
                manifest: WebKitBrowserControlPort.manifest)
            var dialogEvents: [BrowserDialogResult] = []
            engine.onDialogResolved = { dialogEvents.append($0) }

            func run(_ name: String, arguments: [String: String] = [:]) async -> BrowserToolExecutionOutcome {
                let call = AppToolCall(name: name, arguments: arguments, category: .browser)
                let runtime = BrowserToolRuntime(
                    availability: availability,
                    permissionContext: BrowserPermissionContext(origin: origin),
                    perform: { command in try await session.perform(command, using: token) })
                return await BrowserToolExecutor.execute(
                    call: call,
                    in: project,
                    runtime: runtime,
                    currentActionApproved: true)
            }

            let navigation = await run("browser_navigate", arguments: ["url": fixtureURL.absoluteString])
            guard case .completed = navigation else {
                await session.detach()
                return XCTFail("The dialog fixture should load under \(policy), got \(navigation).")
            }
            let snapshotOutcome = await run("browser_snapshot", arguments: ["scope": "visibleText"])
            guard case .completed(let snapshotResult) = snapshotOutcome,
                  case .state(let snapshot) = snapshotResult.value
            else {
                await session.detach()
                return XCTFail("The dialog fixture should expose actionable buttons, got \(snapshotOutcome).")
            }
            let envelope = try JSONDecoder().decode(
                DOMSnapshotEnvelope.self,
                from: Data(snapshot.snapshot.utf8))
            func reference(named name: String) throws -> String {
                try XCTUnwrap(envelope.nodes.first {
                    $0.name == name && $0.reference != nil
                }?.reference)
            }

            for (casePolicy, kind, buttonName) in cases where casePolicy == policy {
                let buttonReference = try reference(named: buttonName)
                let clickTask = Task {
                    await run("browser_click", arguments: ["reference": buttonReference])
                }
                if policy == .ask {
                    for _ in 0..<200 where engine.pendingDialogs.isEmpty {
                        try await Task.sleep(nanoseconds: 10_000_000)
                    }
                    let pending = try XCTUnwrap(engine.pendingDialogs.first)
                    XCTAssertEqual(pending.kind, kind)
                    XCTAssertTrue(engine.resolvePendingDialog(
                        pending.id,
                        decision: .accept(promptText: kind == .prompt ? "approved" : nil)))
                }

                let clickOutcome = await clickTask.value
                guard case .completed(let clickResult) = clickOutcome else {
                    await session.detach()
                    return XCTFail("The \(kind) tool action should finish after policy handling, got \(clickOutcome).")
                }
                XCTAssertEqual(clickResult.commandKind, .click)
                XCTAssertTrue(BrowserToolExecutor.modelOutput(for: clickResult).contains(buttonReference))
                XCTAssertEqual(engine.pendingDialogs.count, 0)

                let event = try XCTUnwrap(dialogEvents.last)
                XCTAssertEqual(event.kind, kind)
                XCTAssertEqual(event.message, "fixture \(kind.rawValue)")
                XCTAssertEqual(
                    event.resolution,
                    policy == .autoDismiss ? .dismissed : .accepted)
                if kind == .confirm {
                    let result = try await webView.evaluateJavaScript(
                        "document.getElementById('result').textContent") as? String
                    XCTAssertEqual(result, policy == .autoDismiss ? "false" : "true")
                } else if kind == .prompt {
                    let result = try await webView.evaluateJavaScript(
                        "document.getElementById('result').textContent") as? String
                    let expected = policy == .autoDismiss ? "null" : policy == .ask ? "approved" : "seed"
                    XCTAssertEqual(result, expected)
                }
            }

            await session.detach()
        }
    }

    func testOneHundredRealWebKitSnapshotsRecordResidentMemory() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        let engine = WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { _ in .allow })
        let tabID = engine.createTab(owner: .agent)
        let webView = try XCTUnwrap(engine.webView(for: tabID))
        webView.frame = NSRect(x: 0, y: 0, width: 900, height: 700)
        let session = BrowserAutomationSession(
            tabStore: store,
            portFactory: { tabID in WebKitBrowserControlPort(engine: engine, tabID: tabID) })
        let token = try await session.attach(to: tabID)
        let url = server.url("/automation")

        let navigation = try await session.perform(
            .navigate(url: url.absoluteString, waitUntil: .finished, timeoutSeconds: 10),
            using: token)
        XCTAssertEqual(navigation.commandKind, .navigate)

        let memoryBefore = try XCTUnwrap(Self.currentResidentMemoryBytes())
        let startedAt = Date()
        var completedIterations = 0
        for iteration in 0..<100 {
            let result = try await session.perform(
                .readState(scope: .visibleText),
                using: token)
            guard case .state(let snapshot) = result.value else {
                await session.detach()
                return XCTFail("Snapshot iteration \(iteration) returned a non-state result.")
            }
            XCTAssertEqual(snapshot.title, "Automation Fixture", "iteration \(iteration)")
            XCTAssertFalse(snapshot.truncated, "iteration \(iteration)")
            completedIterations += 1
        }
        XCTAssertEqual(completedIterations, 100)
        let elapsed = Date().timeIntervalSince(startedAt)
        let memoryAfter = try XCTUnwrap(Self.currentResidentMemoryBytes())
        let deltaBytes = Int64(memoryAfter) - Int64(memoryBefore)
        print(
            "Browser snapshot loop: iterations=100, resident_before=\(memoryBefore), "
                + "resident_after=\(memoryAfter), resident_delta=\(deltaBytes), "
                + "elapsed_seconds=\(String(format: "%.3f", elapsed))")

        await session.detach()
    }

    func testCancellingClickDismissesItsNativeDialogAndRejectsLateApproval() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        let engine = WebKitBrowserEngine(
            tabStore: store, authorizeNavigation: { _ in .allow }, dialogPolicyProvider: { .ask })
        let tabID = engine.createTab(owner: .agent)
        let webView = try XCTUnwrap(engine.webView(for: tabID))
        webView.frame = NSRect(x: 0, y: 0, width: 900, height: 700)
        let session = BrowserAutomationSession(
            tabStore: store,
            portFactory: { tabID in WebKitBrowserControlPort(engine: engine, tabID: tabID) })
        let token = try await session.attach(to: tabID)
        _ = try await session.perform(
            .navigate(url: server.url("/dialogs").absoluteString, waitUntil: .finished, timeoutSeconds: 10),
            using: token)
        let state = try await session.perform(.readState(scope: .interactiveElements), using: token)
        guard case .state(let snapshot) = state.value else { return XCTFail("Expected dialog fixture state") }
        let envelope = try JSONDecoder().decode(DOMSnapshotEnvelope.self, from: Data(snapshot.snapshot.utf8))
        let reference = try XCTUnwrap(envelope.nodes.first { $0.name == "Confirm" }?.reference)
        var events: [BrowserDialogResult] = []
        engine.onDialogResolved = { events.append($0) }

        let click = Task { try await session.perform(.click(reference: reference, timeoutSeconds: 10), using: token) }
        for _ in 0..<200 where engine.pendingDialogs.isEmpty {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        let pending = try XCTUnwrap(engine.pendingDialogs.first)
        await session.cancelCurrentCommand()
        do {
            _ = try await click.value
            XCTFail("The cancelled click must fail")
        } catch is CancellationError { }

        XCTAssertTrue(engine.pendingDialogs.isEmpty)
        XCTAssertEqual(events.last?.resolution, .cancelled)
        XCTAssertFalse(engine.resolvePendingDialog(pending.id, decision: .accept(promptText: nil)))
        await session.detach()
    }

    func testNavigationTimeoutRejectsLatePolicyApprovalBeforeNetworkRequest() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        var approval: CheckedContinuation<BrowserNavigationAuthorizationDecision, Never>?
        let engine = WebKitBrowserEngine(tabStore: store, authorizeNavigation: { _ in
            await withCheckedContinuation { approval = $0 }
        })
        let tabID = engine.createTab(owner: .agent)
        let session = BrowserAutomationSession(
            tabStore: store,
            portFactory: { tabID in WebKitBrowserControlPort(engine: engine, tabID: tabID) })
        let token = try await session.attach(to: tabID)
        let navigation = Task {
            try await session.perform(
                .navigate(url: server.url("/automation").absoluteString, waitUntil: .finished, timeoutSeconds: 0.5),
                using: token)
        }
        for _ in 0..<100 where approval == nil { try await Task.sleep(nanoseconds: 10_000_000) }
        let pendingApproval = try XCTUnwrap(approval)
        do {
            _ = try await navigation.value
            XCTFail("The navigation must time out while approval is pending")
        } catch let error as BrowserControlError {
            XCTAssertEqual(error, .timeout)
        }
        pendingApproval.resume(returning: .allow)
        try await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertEqual(server.requestCount(for: "/automation"), 0)
        await session.detach()
    }

    func testScreenshotToolStoresManagedMediaAndProjectsItIntoProviderHistory() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        let engine = WebKitBrowserEngine(
            tabStore: store,
            authorizeNavigation: { _ in .allow }
        )
        let controlledTabID = engine.createTab(owner: .agent)
        let webView = try XCTUnwrap(engine.webView(for: controlledTabID))
        webView.frame = NSRect(x: 0, y: 0, width: 900, height: 700)
        engine.screenshotSnapshotter = { _ in Self.makeScreenshotImage() }
        let session = BrowserAutomationSession(
            tabStore: store,
            portFactory: { tabID in WebKitBrowserControlPort(engine: engine, tabID: tabID) }
        )
        let token = try await session.attach(to: controlledTabID)
        let fixtureURL = server.url("/automation")
        let origin = try XCTUnwrap(BrowserOrigin(url: fixtureURL))
        let project = AppProject(
            name: "Browser screenshot fixture",
            permissions: AppProjectPermissions(
                browser: .allow,
                browserOriginAllowlist: [origin.canonicalString])
        )
        let navigationCall = AppToolCall(
            name: "browser_navigate",
            arguments: ["url": fixtureURL.absoluteString],
            category: .browser
        )
        let navigationRuntime = BrowserToolRuntime(
            availability: BrowserToolAvailability(
                isEnabled: true,
                manifest: WebKitBrowserControlPort.manifest),
            permissionContext: BrowserPermissionContext(origin: origin),
            perform: { command in try await session.perform(command, using: token) }
        )
        let navigation = await BrowserToolExecutor.execute(
            call: navigationCall,
            in: project,
            runtime: navigationRuntime,
            currentActionApproved: true)
        guard case .completed = navigation else {
            await session.detach()
            return XCTFail("The local screenshot fixture should navigate, got \(navigation).")
        }

        let profileID = UUID().uuidString
        let assetRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("browser-media-\(profileID)", isDirectory: true)
        let vault = ProfileVaultStore(
            rootProvider: { assetRoot },
            profileIDProvider: { profileID },
            migrateLegacyData: false)
        let vaultSession = try XCTUnwrap(vault.prepareForLaunch())
        let assetStore = ManagedAssetStore(vault: vault)
        var storedReference: AppToolMediaReference?
        var pixelCappedReference: AppToolMediaReference?
        defer {
            if let storedReference {
                try? assetStore.release(reference: storedReference.assetReference)
            }
            if let pixelCappedReference {
                try? assetStore.release(reference: pixelCappedReference.assetReference)
            }
            try? FileManager.default.removeItem(at: assetRoot)
        }

        let screenshotCall = AppToolCall(name: "browser_screenshot", category: .browser)
        let cappedScreenshotCall = AppToolCall(name: "browser_screenshot", category: .browser)
        let pixelCappedScreenshotCall = AppToolCall(name: "browser_screenshot", category: .browser)
        let approvedCallIDs: Set<UUID> = [
            screenshotCall.id,
            cappedScreenshotCall.id,
            pixelCappedScreenshotCall.id,
        ]
        let previousRuntimeProvider = AppToolRegistry.browserToolRuntimeProvider
        let previousApprovalProvider = AppToolRegistry.browserActionApprovalProvider
        let previousAssetStoreProvider = AppToolRegistry.browserScreenshotAssetStoreProvider
        AppToolRegistry.browserToolRuntimeProvider = { call, _ in
            let mediaCapability: AppToolMediaCapability
            if call.id == cappedScreenshotCall.id {
                mediaCapability = .imageBearingToolResults(maximumEncodedByteCount: 1)
            } else if call.id == pixelCappedScreenshotCall.id {
                mediaCapability = .imageBearingToolResults(maximumPixelCount: 1)
            } else {
                mediaCapability = .imageBearingToolResults()
            }
            let availability = BrowserToolAvailability(
                isEnabled: true,
                manifest: WebKitBrowserControlPort.manifest,
                mediaCapability: mediaCapability)
            return BrowserToolRuntime(
                availability: availability,
                permissionContext: BrowserPermissionContext(origin: origin),
                perform: { command in try await session.perform(command, using: token) })
        }
        AppToolRegistry.browserActionApprovalProvider = { call, targetProject, requestedOrigin in
            approvedCallIDs.contains(call.id)
                && targetProject?.id == project.id
                && requestedOrigin == origin
        }
        AppToolRegistry.browserScreenshotAssetStoreProvider = { assetStore }
        defer {
            AppToolRegistry.browserToolRuntimeProvider = previousRuntimeProvider
            AppToolRegistry.browserActionApprovalProvider = previousApprovalProvider
            AppToolRegistry.browserScreenshotAssetStoreProvider = previousAssetStoreProvider
        }

        let screenshotResult = await AppToolRegistry.execute(call: screenshotCall, in: project)
        XCTAssertFalse(screenshotResult.isError, screenshotResult.output)
        XCTAssertEqual(screenshotResult.mediaDisposition, .kept, screenshotResult.output)
        storedReference = try XCTUnwrap(screenshotResult.mediaReferences?.first)
        let descriptor = try XCTUnwrap(
            assetStore.descriptor(for: try XCTUnwrap(storedReference).assetReference))
        XCTAssertEqual(descriptor.mimeType, "image/png")
        XCTAssertGreaterThan(descriptor.byteCount, 0)

        let history = AppToolMediaHistoryAdapter.project(
            screenshotResult.mediaReferences,
            capability: .imageBearingToolResults(),
            materialize: { try assetStore.materializedURL(for: $0) })
        XCTAssertEqual(history.omittedReferenceCount, 0)
        XCTAssertEqual(history.images.count, 1)
        if case .path(let path) = history.images[0] {
            XCTAssertTrue(FileManager.default.fileExists(atPath: path))
        } else {
            XCTFail("A provider history screenshot should be a managed image path.")
        }

        let pixelCappedResult = await AppToolRegistry.execute(call: pixelCappedScreenshotCall, in: project)
        XCTAssertFalse(pixelCappedResult.isError, pixelCappedResult.output)
        XCTAssertEqual(pixelCappedResult.mediaDisposition, .downscaled, pixelCappedResult.output)
        pixelCappedReference = try XCTUnwrap(pixelCappedResult.mediaReferences?.first)
        let pixelCapped = try XCTUnwrap(pixelCappedReference)
        XCTAssertLessThanOrEqual(
            pixelCapped.pixelWidth * pixelCapped.pixelHeight,
            1)
        let pixelCappedHistory = AppToolMediaHistoryAdapter.project(
            pixelCappedResult.mediaReferences,
            capability: .imageBearingToolResults(maximumPixelCount: 1),
            materialize: { try assetStore.materializedURL(for: $0) })
        XCTAssertEqual(pixelCappedHistory.omittedReferenceCount, 0)
        XCTAssertEqual(pixelCappedHistory.images.count, 1)
        if let image = pixelCappedHistory.images.first, case .path(let path) = image {
            XCTAssertTrue(FileManager.default.fileExists(atPath: path))
        } else {
            XCTFail("A provider-pixel-capped screenshot should remain a managed image path.")
        }

        let assetsBeforeRefusal = try vaultSession.database.allAssetMetadata()
        let cappedResult = await AppToolRegistry.execute(call: cappedScreenshotCall, in: project)
        XCTAssertTrue(cappedResult.isError)
        XCTAssertEqual(cappedResult.mediaDisposition, .refused)
        XCTAssertNil(cappedResult.mediaReferences)
        XCTAssertEqual(try vaultSession.database.allAssetMetadata(), assetsBeforeRefusal)

        await session.detach()
    }

    func testAppModelFeatureGateControlsRuntimeCatalogAndPane() async {
        let model = AppModel()
        model.browserSettings.enabled = true
        let enabledAvailability = AppToolRegistry.browserToolAvailabilityProvider?()
        XCTAssertTrue(enabledAvailability?.supports(.navigate) == true)

        model.setBrowserPaneOpen(true)
        XCTAssertTrue(model.browserPaneIsOpen)
        XCTAssertTrue(model.browserAutomationCoordinator.isPaneOpen)
        let claimant = AppRightColumnClaimant.resolve(
            openArtifactID: nil,
            previewAttachmentID: nil,
            isInspectorVisible: false,
            showBrowserPane: model.browserSettings.enabled && model.browserPaneIsOpen
        )
        XCTAssertEqual(claimant, .browser)

        model.browserSettings.enabled = false
        model.setBrowserPaneOpen(false)
        XCTAssertFalse(model.browserPaneIsOpen)
        XCTAssertFalse(AppToolRegistry.browserToolAvailabilityProvider?().supports(.navigate) ?? true)
        await Task.yield()
        model.shutdown()
    }

    func testImmediatePaneReopenSupersedesQueuedClose() async {
        let model = AppModel()
        defer { model.shutdown() }
        model.browserSettings.enabled = true
        model.setBrowserPaneOpen(true)
        model.setBrowserPaneOpen(false)
        model.setBrowserPaneOpen(true)
        for _ in 0..<20 { await Task.yield() }

        XCTAssertTrue(model.browserPaneIsOpen)
        XCTAssertTrue(model.browserAutomationCoordinator.isPaneOpen)
    }

    private static func makeScreenshotImage() -> NSImage {
        let representation = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: 32,
            pixelsHigh: 32,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0)!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: representation)
        NSColor.white.setFill()
        NSBezierPath(rect: NSRect(x: 0, y: 0, width: 32, height: 32)).fill()
        NSColor.systemPurple.setFill()
        NSBezierPath(rect: NSRect(x: 8, y: 8, width: 8, height: 8)).fill()
        NSGraphicsContext.restoreGraphicsState()

        let image = NSImage(size: NSSize(width: 32, height: 32))
        image.addRepresentation(representation)
        return image
    }

    private static func currentResidentMemoryBytes() -> UInt64? {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count)
            }
        }
        guard status == KERN_SUCCESS else { return nil }
        return info.resident_size
    }
}
