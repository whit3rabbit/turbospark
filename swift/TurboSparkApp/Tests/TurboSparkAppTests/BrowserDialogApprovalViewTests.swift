import XCTest
@testable import TurboSparkApp

@MainActor
final class BrowserDialogApprovalViewTests: XCTestCase {
    func testAcceptRoutesPromptTextToTheOriginalRequest() throws {
        let request = makeRequest(kind: .prompt, defaultText: "guest")
        var received: (UUID, BrowserDialogDecision)?
        let actions = BrowserDialogApprovalActions(request: request) { id, decision in
            received = (id, decision)
            return true
        }

        XCTAssertTrue(actions.accept(promptResponse: "alice"))
        XCTAssertEqual(received?.0, request.id)
        XCTAssertEqual(received?.1, .accept(promptText: "alice"))
    }

    func testAcceptWithoutPromptDoesNotReturnPageProvidedDefaultText() throws {
        let request = makeRequest(kind: .confirm, defaultText: "unused")
        var receivedDecision: BrowserDialogDecision?
        let actions = BrowserDialogApprovalActions(request: request) { _, decision in
            receivedDecision = decision
            return true
        }

        XCTAssertTrue(actions.accept(promptResponse: "ignored"))
        XCTAssertEqual(receivedDecision, .accept(promptText: nil))
    }

    func testDismissAndStaleRequestAreSentToTheOriginalResolver() throws {
        let request = makeRequest(kind: .alert)
        var isPending = true
        var received: (UUID, BrowserDialogDecision)?
        let actions = BrowserDialogApprovalActions(request: request) { id, decision in
            guard isPending else { return false }
            received = (id, decision)
            return true
        }

        XCTAssertTrue(actions.dismiss())
        XCTAssertEqual(received?.0, request.id)
        XCTAssertEqual(received?.1, .dismiss)

        isPending = false
        XCTAssertFalse(actions.accept(promptResponse: "late"))
    }

    private func makeRequest(
        kind: BrowserJavaScriptDialogKind,
        defaultText: String? = nil
    ) -> BrowserDialogRequest {
        BrowserDialogRequest(
            id: UUID(),
            tabID: BrowserTabID(),
            kind: kind,
            message: "Page supplied message",
            defaultText: defaultText,
            sourceOrigin: BrowserOrigin(origin: "https://example.test")
        )
    }
}
