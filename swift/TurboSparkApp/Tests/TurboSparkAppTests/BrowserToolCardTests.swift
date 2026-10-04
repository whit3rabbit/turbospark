import Foundation
import XCTest
@testable import TurboSparkApp

final class BrowserToolCardTests: XCTestCase {
    func testNavigationCardShowsCanonicalOriginWithoutURLComponents() throws {
        let rawURL = "https://docs.example/private/page?token=query-secret#fragment-secret"
        let call = AppToolCall(
            name: "browser_navigate",
            arguments: ["url": rawURL],
            status: .completed)
        let result = AppToolResult(
            callID: call.id,
            output: "Dialog: dialog-secret. Page excerpt: page-secret. Typed value: typed-secret.",
            durationSeconds: 1.239)

        let presentation = BrowserToolCardPresentation(call: call, result: result)

        XCTAssertEqual(presentation.action, "Navigate")
        XCTAssertEqual(presentation.target, "https://docs.example")
        XCTAssertEqual(presentation.outcome, .completed)
        XCTAssertEqual(presentation.durationText, "1.24 s")
        let visibleFields = [presentation.action, presentation.target ?? "",
            presentation.outcome.rawValue, presentation.durationText ?? ""].joined(separator: " ")
        for forbidden in ["/private/page", "token", "query-secret", "fragment-secret",
            "dialog-secret", "page-secret", "typed-secret"]
        {
            XCTAssertFalse(visibleFields.contains(forbidden), forbidden)
        }
    }

    func testElementActionShowsOnlyValidatedOpaqueReference() throws {
        let reference = "0123456789abcdef0123456789abcdef"
        let origin = try XCTUnwrap(BrowserOrigin(origin: "https://docs.example"))
        let metadata = AppToolBrowserCardMetadata(origin: origin, elementReference: reference)
        let call = AppToolCall(
            name: "browser_type",
            arguments: ["reference": reference, "text": "typed-secret", "submit": "true"],
            status: .completed)
        let result = AppToolResult(
            callID: call.id,
            output: "Typed into browser element \(reference). Page excerpt: page-secret.",
            durationSeconds: 0.4,
            browserCardMetadata: metadata)

        let presentation = BrowserToolCardPresentation(call: call, result: result)

        XCTAssertEqual(presentation.action, "Type")
        XCTAssertEqual(presentation.target, reference)
        XCTAssertEqual(presentation.outcome, .completed)
        XCTAssertEqual(presentation.durationText, "0.40 s")
        XCTAssertFalse(presentation.target?.contains("typed-secret") == true)
        XCTAssertFalse(presentation.target?.contains("page-secret") == true)
    }

    func testBrowserResultMetadataUsesOriginAndResolvedReferenceOnly() throws {
        let origin = try XCTUnwrap(BrowserOrigin(origin: "https://docs.example"))
        let reference = "0123456789abcdef0123456789abcdef"
        let actionMetadata = BrowserToolCardMetadataBuilder.make(
            value: .clicked(BrowserElementActionResult(reference: reference)),
            fallbackOrigin: origin)
        XCTAssertEqual(actionMetadata.canonicalOrigin, "https://docs.example")
        XCTAssertEqual(actionMetadata.elementReference, reference)

        let navigationMetadata = BrowserToolCardMetadataBuilder.make(
            value: .navigated(BrowserNavigationResult(
                url: "https://docs.example/private?token=secret#fragment",
                reachedState: .finished)),
            fallbackOrigin: nil)
        XCTAssertEqual(navigationMetadata.canonicalOrigin, "https://docs.example")
        XCTAssertNil(navigationMetadata.elementReference)
    }

    func testNonBrowserCallsDoNotUseBrowserCardPresentation() {
        for name in ["read_file", "web_fetch", "custom_tool"] {
            XCTAssertFalse(BrowserToolCardPresentation.isBrowserTool(name), name)
        }
        for name in ["browser_navigate", "browser_click", "browser_type", "browser_press_key",
            "browser_scroll", "browser_screenshot", "browser_snapshot", "browser_wait"]
        {
            XCTAssertTrue(BrowserToolCardPresentation.isBrowserTool(name), name)
        }
    }

    func testBrowserMetadataRoundTripsAndLegacyResultsDecodeWithoutIt() throws {
        let origin = try XCTUnwrap(BrowserOrigin(origin: "HTTPS://Docs.Example:443"))
        let metadata = AppToolBrowserCardMetadata(
            origin: origin,
            elementReference: "0123456789abcdef0123456789abcdef")
        let result = AppToolResult(
            callID: UUID(),
            output: "private snapshot output",
            browserCardMetadata: metadata)
        let encoded = try JSONEncoder().encode(result)
        XCTAssertEqual(try JSONDecoder().decode(AppToolResult.self, from: encoded), result)

        var legacyObject = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        legacyObject.removeValue(forKey: "browserCardMetadata")
        legacyObject.removeValue(forKey: "mediaReferences")
        legacyObject.removeValue(forKey: "mediaDisposition")
        let legacyData = try JSONSerialization.data(withJSONObject: legacyObject)
        let legacyResult = try JSONDecoder().decode(AppToolResult.self, from: legacyData)
        XCTAssertNil(legacyResult.browserCardMetadata)
        XCTAssertNil(legacyResult.mediaReferences)
        XCTAssertNil(legacyResult.mediaDisposition)
    }

    func testInvalidStoredCardTargetsAreDropped() throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "canonicalOrigin": "https://docs.example/private?token=secret",
            "elementReference": "typed-secret"
        ])

        let metadata = try JSONDecoder().decode(AppToolBrowserCardMetadata.self, from: data)

        XCTAssertNil(metadata.canonicalOrigin)
        XCTAssertNil(metadata.elementReference)
        XCTAssertNil(metadata.displayTarget)
    }

    func testBrowserToolDetailsRemainBehindTheExistingChatExportOption() throws {
        let call = AppToolCall(
            name: "browser_snapshot",
            arguments: ["scope": "viewport"],
            status: .completed)
        let result = AppToolResult(callID: call.id, output: "snapshot-private-page-text")
        let chat = AppChat(title: "Browser", messages: [
            AppChatMessage(role: .assistant, content: "", toolCalls: [call], toolResults: [result])
        ])

        let defaultExport = AppChatShareDocument(chat: chat)
        let detailedExport = AppChatShareDocument(chat: chat, options: .init(includeToolDetails: true))
        XCTAssertFalse(defaultExport.markdown.contains("snapshot-private-page-text"))
        XCTAssertTrue(detailedExport.markdown.contains("snapshot-private-page-text"))
        XCTAssertTrue(detailedExport.markdown.contains("browser_snapshot"))
    }

    func testSupportReportProjectionDropsScreenshotSnapshotAndPickedContent() throws {
        let origin = try XCTUnwrap(BrowserOrigin(origin: "https://docs.example"))
        let call = AppToolCall(
            name: "browser_screenshot",
            arguments: ["full_page": "true"],
            status: .completed)
        let result = AppToolResult(
            callID: call.id,
            output: "raw-snapshot page-secret picked-element-content dialog-secret",
            durationSeconds: 0.037,
            mediaReferences: [AppToolMediaReference(
                assetReference: "managed:screenshot-secret.png",
                mimeType: "image/png",
                byteCount: 128,
                pixelWidth: 16,
                pixelHeight: 16)],
            browserCardMetadata: AppToolBrowserCardMetadata(origin: origin))

        let event = try XCTUnwrap(BrowserAutomationSupportReportAdapter.project(call: call, result: result))
        let encoded = try JSONEncoder().encode(event)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(Set(object.keys), [
            "actionType", "canonicalOrigin", "outcome", "durationMilliseconds"
        ])
        XCTAssertEqual(object["actionType"] as? String, "screenshot")
        XCTAssertEqual(object["canonicalOrigin"] as? String, "https://docs.example")
        XCTAssertEqual(object["durationMilliseconds"] as? UInt32, 37)

        let reportEntry = try XCTUnwrap(String(data: encoded, encoding: .utf8))
        for forbidden in [
            "raw-snapshot", "page-secret", "picked-element-content", "dialog-secret",
            "screenshot-secret", "image/png"
        ] {
            XCTAssertFalse(reportEntry.contains(forbidden), forbidden)
        }
    }
}
