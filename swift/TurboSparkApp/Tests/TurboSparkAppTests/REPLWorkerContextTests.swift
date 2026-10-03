import XCTest
@testable import TurboSparkApp

final class REPLWorkerContextTests: XCTestCase {
    func testTopLevelBindingsPersistAndCompletionValueIsRendered() async {
        let worker = REPLWorkerContext()

        let first = await worker.evaluateWithThreadStatus(code: "let total = 40; total + 2")
        let second = await worker.evaluateWithThreadStatus(code: "total + 1")

        XCTAssertEqual(first.result.status, .completed)
        XCTAssertEqual(first.result.completionText, "42")
        XCTAssertTrue(first.result.sessionCreated)
        XCTAssertFalse(first.result.sessionReset)
        XCTAssertEqual(second.result.status, .completed)
        XCTAssertEqual(second.result.completionText, "41")
        XCTAssertFalse(second.result.sessionCreated)
        XCTAssertFalse(first.ranOnMainThread)
        XCTAssertFalse(second.ranOnMainThread)
    }

    func testObjectCompletionPreservesItsContents() async {
        let worker = REPLWorkerContext()

        let result = await worker.evaluate(code: "({answer: 42, items: [1, 2]})")

        XCTAssertEqual(result.status, .completed)
        XCTAssertEqual(result.completionText, #"{"answer":42,"items":[1,2]}"#)
    }

    func testConsoleAndRuntimeErrorsAreCapturedWithoutDiscardingTheContext() async {
        let worker = REPLWorkerContext()

        let failed = await worker.evaluate(
            code: "var retained = 40; console.log('value', 2); console.warn('check'); throw new Error('failure')")
        let recovered = await worker.evaluate(code: "retained + 2")

        XCTAssertEqual(failed.status, .failed)
        XCTAssertEqual(failed.outputEvents.map(\.level), [.log, .warn])
        XCTAssertEqual(failed.outputEvents.map(\.text), ["value 2", "check"])
        XCTAssertEqual(failed.consoleText, "value 2")
        XCTAssertTrue(failed.errorText?.contains("check") == true)
        XCTAssertTrue(failed.errorText?.contains("failure") == true)
        XCTAssertEqual(recovered.status, .completed)
        XCTAssertEqual(recovered.completionText, "42")
    }

    func testOneCallCannotDisableConsoleCaptureForTheNextCall() async {
        let worker = REPLWorkerContext()

        let first = await worker.evaluate(code: """
        try { console.log = () => {}; } catch (_) {}
        try { globalThis.console = {}; } catch (_) {}
        console.log("first")
        """)
        let second = await worker.evaluate(code: "console.log('still captured')")

        XCTAssertEqual(first.status, .completed)
        XCTAssertEqual(first.outputEvents.map(\.text), ["first"])
        XCTAssertEqual(second.status, .completed)
        XCTAssertEqual(second.outputEvents.map(\.text), ["still captured"])
    }

    func testMutableFormattingGlobalsCannotCorruptConsoleOrCompletionText() async {
        let worker = REPLWorkerContext()

        let result = await worker.evaluate(code: """
        JSON.stringify = value => typeof value === "object" ? "tampered" : undefined;
        String = () => "tampered string";
        console.log({line: 1}, Symbol("marker"));
        ({answer: 42})
        """)

        XCTAssertEqual(result.status, .completed)
        XCTAssertEqual(result.outputEvents.map(\.text), ["{\"line\":1} Symbol(marker)"])
        XCTAssertEqual(result.completionText, #"{"answer":42}"#)
    }
}
