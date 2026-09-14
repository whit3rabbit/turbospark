import AppKit
import SwiftUI
import XCTest
import WebKit
@testable import TurboSparkApp

@MainActor
final class ChatSharingAndSummaryTests: XCTestCase {
    /// AppModel installs process-wide callbacks. Restore them so a UI fixture
    /// cannot change the execution policy of the next test class.
    private func makeModelFixture() -> (AppModel, () -> Void) {
        let previous0 = AppToolRegistry.activeSessionProvider
        let previous1 = AppToolRegistry.userSystemPromptProvider
        let previous2 = AppToolRegistry.subagentSamplingOptionsProvider
        let previous3 = AppToolRegistry.webToolsEnabledProvider
        let previous4 = AppToolRegistry.subagentProgressSink
        let previous5 = AppToolRegistry.backgroundAgentLauncher
        let previous6 = AppToolRegistry.backgroundAgentStopper
        let previous7 = AppToolRegistry.observationRecaller
        let previous8 = AppToolRegistry.syntextIndexingEnabled
        let previous9 = TodoWriteExecutor.onTodosUpdated
        let previous10 = TaskManager.shared.onTasksUpdated
        let previous11 = SendUserFileExecutor.onFileSent
        let previous12 = ArtifactRegistrar.onArtifactsProduced
        let previous13 = PushNotificationExecutor.onNotificationPushed
        let previous14 = AskUserQuestionExecutor.answerWaiter
        let previous15 = CronScheduler.shared.fireHandler
        let model = AppModel()
        return (model, {
            model.stopCronScheduler()
            AppToolRegistry.activeSessionProvider = previous0
            AppToolRegistry.userSystemPromptProvider = previous1
            AppToolRegistry.subagentSamplingOptionsProvider = previous2
            AppToolRegistry.webToolsEnabledProvider = previous3
            AppToolRegistry.subagentProgressSink = previous4
            AppToolRegistry.backgroundAgentLauncher = previous5
            AppToolRegistry.backgroundAgentStopper = previous6
            AppToolRegistry.observationRecaller = previous7
            AppToolRegistry.syntextIndexingEnabled = previous8
            TodoWriteExecutor.onTodosUpdated = previous9
            TaskManager.shared.onTasksUpdated = previous10
            SendUserFileExecutor.onFileSent = previous11
            ArtifactRegistrar.onArtifactsProduced = previous12
            PushNotificationExecutor.onNotificationPushed = previous13
            AskUserQuestionExecutor.answerWaiter = previous14
            CronScheduler.shared.fireHandler = previous15
        })
    }

    private func conversation() -> AppChat {
        let call = AppToolCall(name: "read_file", arguments: ["path": "guide.md"], status: .completed)
        return AppChat(title: "Project notes", messages: [
            AppChatMessage(role: .system, content: "private instructions"),
            AppChatMessage(role: .user, content: "Explain the change"),
            AppChatMessage(role: .assistant, content: "# Result\n\nThe **change** works.\n\n- First\n- Second\n\n```swift\nlet answer = 42\n```",
                reasoning: "private reasoning", toolCalls: [call],
                toolResults: [AppToolResult(callID: call.id, output: "tool evidence")],
                alternates: [AppChatMessage(role: .assistant, content: "old response")]),
        ])
    }

    func testCleanSnapshotExcludesInternalContentAndIsIndependentOfLaterChanges() throws {
        var chat = conversation()
        let snapshot = AppChatShareDocument(chat: chat, liveContent: "Partial response")
        chat.messages[1].content = "later edit"
        for format in [AppChatExportFormat.markdown, .html] {
            let text = try XCTUnwrap(String(data: snapshot.data(format: format), encoding: .utf8))
            for hidden in ["private instructions", "private reasoning", "tool evidence", "old response", "later edit"] {
                XCTAssertFalse(text.contains(hidden), hidden)
            }
            XCTAssertTrue(text.contains("Explain the change"))
            XCTAssertTrue(text.contains("Partial response"))
        }
        XCTAssertEqual(snapshot.chat.messages.count, 3)
        XCTAssertEqual(snapshot.chat.messages.last?.stopReason, "partial")
    }

    func testToolDetailsAreOptInWithoutReasoning() {
        let snapshot = AppChatShareDocument(chat: conversation(), options: .init(includeToolDetails: true))
        XCTAssertTrue(snapshot.markdown.contains("tool evidence"))
        XCTAssertTrue(snapshot.markdown.contains("guide.md"))
        XCTAssertFalse(snapshot.markdown.contains("private reasoning"))
    }

    func testWordIsARealDocumentWithMatchingText() throws {
        let snapshot = AppChatShareDocument(chat: conversation())
        let data = try snapshot.data(format: .docx)
        XCTAssertEqual(Array(data.prefix(2)), [0x50, 0x4b])
        let decoded = try NSAttributedString(data: data, options: [
            .documentType: NSAttributedString.DocumentType.officeOpenXML,
        ], documentAttributes: nil)
        XCTAssertTrue(decoded.string.contains("Explain the change"))
        XCTAssertTrue(decoded.string.contains("let answer = 42"))
        XCTAssertFalse(decoded.string.contains("private reasoning"))
    }

    func testHTMLDoesNotExecuteOrFetchMessageMarkup() {
        let chat = AppChat(title: "\"<test>", messages: [AppChatMessage(role: .assistant,
            content: "<script>fetch('https://example.invalid')</script>\n\n![remote](https://example.invalid/a.png)\n\n```\" onmouseover=\"bad\nhello\n```",
            imagePaths: ["https://example.invalid/image.png", "/missing/image.png"])])
        let snapshot = AppChatShareDocument(chat: chat)
        XCTAssertTrue(snapshot.html.contains("default-src 'none'"))
        XCTAssertFalse(snapshot.html.contains("<script>"))
        XCTAssertFalse(snapshot.html.contains("src=\"https:"))
        XCTAssertFalse(snapshot.html.contains("src=\"file:"))
        XCTAssertTrue(snapshot.html.contains("unavailable locally"))
        XCTAssertTrue(snapshot.attachments.allSatisfy { $0.png == nil })
    }

    func testExportedHTMLLoadsWithoutRemoteResources() async throws {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        let web = WKWebView(frame: NSRect(x: 0, y: 0, width: 900, height: 1000), configuration: config)
        let loaded = expectation(description: "Export loaded")
        let delegate = ExportNavigationObserver(loaded: loaded)
        web.navigationDelegate = delegate
        let document = AppChatShareDocument(chat: conversation())
        web.loadHTMLString(document.html, baseURL: nil)
        await fulfillment(of: [loaded], timeout: 15)
        let body = try await web.evaluateJavaScript("document.body.innerText") as? String
        XCTAssertTrue(body?.contains("Explain the change") == true)
        let remote = try await web.evaluateJavaScript("performance.getEntriesByType('resource').filter(x => /^https?:/.test(x.name)).length") as? Int
        XCTAssertEqual(remote, 0)
        if let path = ProcessInfo.processInfo.environment["TURBOSPARK_CHAT_UI_REVIEW_DIR"] {
            let image = try await web.takeSnapshot(configuration: nil)
            let rep = try XCTUnwrap(NSBitmapImageRep(data: XCTUnwrap(image.tiffRepresentation)))
            try XCTUnwrap(rep.representation(using: .png, properties: [:]))
                .write(to: URL(fileURLWithPath: path).appendingPathComponent("html-page.png"))
        }
    }

    func testLocalImageIsEmbeddedWithoutAFileDependency() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let url = folder.appendingPathComponent("picture.png")
        let image = NSImage(size: NSSize(width: 40, height: 30))
        image.lockFocus(); NSColor.blue.setFill(); NSRect(x: 0, y: 0, width: 40, height: 30).fill(); image.unlockFocus()
        let rep = try XCTUnwrap(NSBitmapImageRep(data: XCTUnwrap(image.tiffRepresentation)))
        try XCTUnwrap(rep.representation(using: .png, properties: [:])).write(to: url)
        let snapshot = AppChatShareDocument(chat: AppChat(title: "Image", messages: [
            AppChatMessage(role: .user, content: "An image", imagePaths: [url.path]),
        ]))
        try FileManager.default.removeItem(at: url)
        XCTAssertTrue(snapshot.html.contains("src=\"data:image/png;base64,"))
        XCTAssertNotNil(snapshot.attachments.first?.png)
        XCTAssertTrue(snapshot.markdown.contains("picture.png"))
        XCTAssertFalse(try snapshot.data(format: .docx).isEmpty)
    }

    func testSourcesDeduplicateWebResultsAndConversationLinksButNotFileOutput() throws {
        let call = AppToolCall(name: "read_url_content", arguments: ["url": "https://EXAMPLE.com"])
        let file = AppToolCall(name: "read_file")
        let messages = [AppChatMessage(role: .user, content: "[Source](https://example.com/)") ,
            AppChatMessage(role: .assistant, content: "", toolCalls: [call, file], toolResults: [
                AppToolResult(callID: call.id, output: "https://example.com/ https://example.org/docs"),
                AppToolResult(callID: file.id, output: "https://not-a-source.invalid"),
            ])]
        XCTAssertEqual(ProjectChatSummary.sources(messages: messages).map(\.id), [
            "https://example.com/", "https://example.org/docs",
        ])
        XCTAssertEqual(ProjectChatSummary.sources(messages: []), [])
    }

    func testSummaryRestoresAfterHigherPriorityPanelsClose() {
        let id = UUID()
        XCTAssertEqual(AppRightColumnClaimant.resolve(openArtifactID: id, previewAttachmentID: nil,
            isInspectorVisible: true, showProjectSummary: true), .artifact(id))
        XCTAssertEqual(AppRightColumnClaimant.resolve(openArtifactID: nil, previewAttachmentID: nil,
            isInspectorVisible: true, showProjectSummary: true), .inspector)
        XCTAssertEqual(AppRightColumnClaimant.resolve(openArtifactID: nil, previewAttachmentID: nil,
            isInspectorVisible: false, showProjectSummary: true), .projectSummary)
        XCTAssertEqual(AppRightColumnClaimant.resolve(openArtifactID: nil, previewAttachmentID: nil,
            isInspectorVisible: false, showProjectSummary: false), .none)
        XCTAssertFalse(AppRightColumnClaimant.projectSummary.isPreviewPane)
        let boundary = AppChromeLayout.primaryMinimumWidth + ProjectChatSummary.width + AppChromeLayout.dividerWidth
        XCTAssertTrue(ProjectChatSummary.canPin(availableWidth: boundary))
        XCTAssertFalse(ProjectChatSummary.canPin(availableWidth: boundary - 1))
    }

    func testProjectSummaryEligibilityAndChatSwitching() {
        XCTAssertFalse(ProjectChatSummary.isAvailable(projectID: nil, isChat: true))
        XCTAssertFalse(ProjectChatSummary.isAvailable(projectID: UUID(), isChat: false))
        XCTAssertTrue(ProjectChatSummary.isAvailable(projectID: UUID(), isChat: true))
        let (model, restore) = makeModelFixture()
        defer { restore() }
        let first = AppChat(title: "First", messages: [AppChatMessage(role: .user, content: "https://first.invalid")])
        let second = AppChat(title: "Second", messages: [AppChatMessage(role: .user, content: "https://second.invalid")])
        model.chats = [first, second]
        model.selectedChatID = first.id
        XCTAssertEqual(ProjectChatSummary.sources(messages: model.selectedTurnMessages).first?.title, "first.invalid")
        model.selectedChatID = second.id
        XCTAssertEqual(ProjectChatSummary.sources(messages: model.selectedTurnMessages).first?.title, "second.invalid")
    }

    func testCleanShareOmitsEmptyToolOnlyMessages() {
        let call = AppToolCall(name: "read_file")
        let chat = AppChat(title: "Test", messages: [AppChatMessage(role: .assistant, content: "", toolCalls: [call])])
        XCTAssertTrue(AppChatShareDocument(chat: chat).chat.messages.isEmpty)
        XCTAssertEqual(AppChatShareDocument(chat: chat, options: .init(includeToolDetails: true)).chat.messages.count, 1)
    }

    func testTasksPrioritizeCurrentWorkAndRetainCompletedRows() {
        let done = TodoItem(id: "done", content: "Done", status: "completed", activeForm: "")
        let pending = TodoItem(id: "pending", content: "Pending", status: "pending", activeForm: "")
        let active = TodoItem(id: "active", content: "Active", status: "in_progress", activeForm: "Working")
        XCTAssertEqual(ProjectChatSummary.orderedTasks([done, pending, active]).map(\.id), ["active", "pending", "done"])
    }

    func testEveryExecutableToolAndAliasHasAnIconAndLabel() throws {
        XCTAssertEqual(Set(ToolPresentation.registry.keys), AppToolRegistry.supportedToolNames)
        for name in AppToolRegistry.supportedToolNames {
            let info = ToolPresentation.resolve(name.uppercased())
            XCTAssertTrue(info.isBuiltIn, name)
            XCTAssertFalse(info.label.isEmpty, name)
            XCTAssertNotNil(BundledToolArtwork.images[info.icon], name)
        }
        XCTAssertEqual(ToolPresentation.resolve("read_file").label, "Read File")
        XCTAssertEqual(ToolPresentation.resolve("edit_file").label, "Edit File")
        XCTAssertNotEqual(ToolPresentation.resolve("read_url_content").icon, ToolPresentation.resolve("read_file").icon)
        XCTAssertEqual(ToolPresentation.resolve("mcp__demo__read_file").icon, "plug")
        XCTAssertFalse(ToolPresentation.resolve("custom_edit_tool").isBuiltIn)
    }

    func testToolErrorsOverrideCompletedStatus() {
        let call = AppToolCall(name: "read_file", status: .completed)
        XCTAssertEqual(ToolPresentation.status(call: call, result: AppToolResult(callID: call.id, output: "failed", isError: true)), .failed)
        XCTAssertEqual(ToolPresentation.status(call: call, result: nil), .completed)
    }

    func testSiteIconsUseExactBundledDomains() throws {
        XCTAssertEqual(OfflineSiteIcons.asset(for: try XCTUnwrap(URL(string: "https://github.com/org/repo"))), "site-github")
        XCTAssertNil(OfflineSiteIcons.asset(for: try XCTUnwrap(URL(string: "https://github.com.attacker.invalid"))))
        XCTAssertNil(OfflineSiteIcons.asset(for: try XCTUnwrap(URL(string: "https://unknown.invalid"))))
    }

    /// Opt-in artifacts make visual checks reproducible without launching a
    /// second app against the user's persistent profiles.
    func testRenderReviewArtifactsWhenRequested() throws {
        guard let path = ProcessInfo.processInfo.environment["TURBOSPARK_CHAT_UI_REVIEW_DIR"] else { return }
        let folder = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let document = AppChatShareDocument(chat: conversation())
        for format in [AppChatExportFormat.markdown, .html, .docx] {
            try document.data(format: format).write(to: folder.appendingPathComponent("conversation." + format.fileExtension))
        }
        let (model, restore) = makeModelFixture()
        defer { restore() }
        var chat = conversation()
        chat.projectID = UUID()
        chat.messages[1].content += "\nhttps://github.com/primer/octicons"
        chat.artifacts = [AppArtifact(chatID: chat.id, path: "/tmp/review.md", title: "Review notes", origin: .fileWrite)]
        chat.todos = [TodoItem(content: "Build offline exports", status: "completed", activeForm: ""),
            TodoItem(content: "Check the chat UI", status: "in_progress", activeForm: "Checking the chat UI")]
        model.chats = [chat]; model.selectedChatID = chat.id
        for (name, scheme, width) in [("light", ColorScheme.light, CGFloat(980)), ("dark", ColorScheme.dark, CGFloat(980)), ("narrow", ColorScheme.dark, CGFloat(600))] {
            let content = VStack(spacing: 0) {
                TopBarView(model: model, isChatSidebarVisible: false, isInspectorVisible: false,
                    toggleChatSidebar: {}, toggleInspector: {}, canPinSummary: width > 840, isSummaryVisible: width > 840)
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(["read_file", "edit_file", "run_command", "web_fetch", "mcp__demo__inspect"], id: \.self) { name in
                            ToolCallCardView(model: model, call: AppToolCall(name: name,
                                arguments: ["path": "guide.md", "command": "swift test", "url": "https://github.com/primer/octicons"],
                                status: .completed), result: nil)
                        }
                    }.padding().frame(width: width > 840 ? 640 : width - 20)
                    if width > 840 { ProjectChatSummaryView(model: model).frame(width: 320) }
                }
            }.frame(width: width, height: 540).appThemed().environment(\.colorScheme, scheme)
            let host = NSHostingView(rootView: content)
            host.frame = NSRect(x: 0, y: 0, width: width, height: 540)
            host.appearance = NSAppearance(named: scheme == .dark ? .darkAqua : .aqua)
            host.layoutSubtreeIfNeeded()
            let rep = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
            host.cacheDisplay(in: host.bounds, to: rep)
            try XCTUnwrap(rep.representation(using: .png, properties: [:]))
                .write(to: folder.appendingPathComponent("chat-" + name + ".png"))
        }
    }
}

@MainActor
private final class ExportNavigationObserver: NSObject, WKNavigationDelegate {
    let loaded: XCTestExpectation
    init(loaded: XCTestExpectation) { self.loaded = loaded }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { loaded.fulfill() }
}
