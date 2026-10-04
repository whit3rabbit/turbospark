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

    // MARK: Top-level await through the production path (2.2, 3.1)

    func testTopLevelAwaitSettlesThroughTheProductionEvaluatePath() async {
        let worker = REPLWorkerContext()

        let result = await worker.evaluate(code: """
        let before = 10;
        let after = await Promise.resolve(32);
        after + before
        """)

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(result.completionText, "42")
        XCTAssertTrue(result.sessionCreated)
        XCTAssertFalse(result.sessionReset)
    }

    func testDeclarationsBeforeAndAfterAwaitPersistAcrossLaterProductionCalls() async {
        let worker = REPLWorkerContext()

        let first = await worker.evaluate(code: """
        var shared = 'v';
        let before = 'pre';
        let awaited = await Promise.resolve('mid');
        let after = 'post';
        function doubled(n) { return n * 2; }
        class Marker {}
        after
        """)
        XCTAssertEqual(first.status, .completed, first.errorText ?? "")
        XCTAssertTrue(first.sessionCreated)
        XCTAssertFalse(first.sessionReset)

        let second = await worker.evaluate(code: """
        [shared, before, awaited, after, doubled(21), Marker.name,
         Object.prototype.hasOwnProperty.call(globalThis, 'after')].join(',')
        """)
        XCTAssertEqual(second.status, .completed, second.errorText ?? "")
        XCTAssertFalse(second.sessionCreated)
        XCTAssertEqual(second.completionText, "v,pre,mid,post,42,Marker,true")
    }

    func testConsoleCaptureAggregatesThroughTheAwaitSettlementPath() async {
        let worker = REPLWorkerContext()

        let result = await worker.evaluate(code: """
        console.log('start');
        let value = await Promise.resolve(5);
        console.log('end', value);
        value
        """)

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(result.outputEvents.map(\.text), ["start", "end 5"])
        XCTAssertEqual(result.consoleText, "start\nend 5")
        XCTAssertEqual(result.completionText, "5")
    }

    func testOrdinaryScriptsKeepNativeLexicalSemanticsWithoutLowering() async {
        let worker = REPLWorkerContext()

        let declared = await worker.evaluate(code: "let u = 3; u")
        XCTAssertEqual(declared.status, .completed, declared.errorText ?? "")
        XCTAssertEqual(declared.completionText, "3")

        let lexical = await worker.evaluate(
            code: "Object.prototype.hasOwnProperty.call(globalThis, 'u')")
        XCTAssertEqual(lexical.status, .completed, lexical.errorText ?? "")
        XCTAssertEqual(
            lexical.completionText, "false",
            "ordinary scripts must keep native lexical (non-property) declarations")

        let redeclared = await worker.evaluate(code: "let u = 4")
        XCTAssertEqual(redeclared.status, .failed)
        XCTAssertTrue(
            redeclared.errorText?.contains("SyntaxError") == true,
            "unexpected redeclaration outcome: \(redeclared.errorText ?? "")")
    }

    // MARK: Parse recovery (2.4)

    func testEmptyScriptReturnsParseErrorAndKeepsTheSessionUsable() async {
        let worker = REPLWorkerContext()

        let sentinel = await worker.evaluate(code: "let sentinel = 'keep'; sentinel")
        XCTAssertEqual(sentinel.status, .completed, sentinel.errorText ?? "")

        for empty in ["", "   ", "\n\t  "] {
            let result = await worker.evaluate(code: empty)
            XCTAssertEqual(
                result.status, .parseError,
                "expected parseError for the empty script \(empty.debugDescription)")
            XCTAssertTrue(
                result.errorText?.contains("empty") == true,
                "expected an empty-script reason, got: \(result.errorText ?? "")")
            XCTAssertTrue(result.outputEvents.isEmpty)
        }

        let after = await worker.evaluate(code: "sentinel")
        XCTAssertEqual(after.status, .completed, after.errorText ?? "")
        XCTAssertEqual(after.completionText, "keep")
        XCTAssertFalse(after.sessionCreated)
    }

    func testParseErrorReturnsItsResultAndTheSessionStaysUsable() async {
        let worker = REPLWorkerContext()

        let sentinel = await worker.evaluate(code: "let sentinel = 'keep'; sentinel")
        XCTAssertEqual(sentinel.status, .completed, sentinel.errorText ?? "")

        let broken = await worker.evaluate(code: "let a = (")
        XCTAssertEqual(broken.status, .parseError)
        XCTAssertTrue(
            broken.errorText?.contains("SyntaxError") == true,
            "expected a syntax message, got: \(broken.errorText ?? "")")
        XCTAssertTrue(broken.outputEvents.isEmpty, "no evaluation must run for a parse error")

        let recovered = await worker.evaluate(code: "sentinel + '!'")
        XCTAssertEqual(recovered.status, .completed, recovered.errorText ?? "")
        XCTAssertEqual(recovered.completionText, "keep!")
    }

    // MARK: Unsupported syntax rejections (2.5)

    func testUnsupportedSyntaxReturnsDedicatedStatusAndLeavesTheSessionUnchanged() async {
        let worker = REPLWorkerContext()

        let sentinel = await worker.evaluate(code: "let sentinel = 'original'; sentinel")
        XCTAssertEqual(sentinel.status, .completed, sentinel.errorText ?? "")

        let rejected = await worker.evaluate(code: """
        let broken = await Promise.resolve(1);
        let { a } = await Promise.resolve({ a: 2 });
        """)
        XCTAssertEqual(rejected.status, .unsupportedSyntax)
        XCTAssertTrue(
            rejected.errorText?.contains("destructuring") == true,
            "expected a destructuring reason, got: \(rejected.errorText ?? "")")
        XCTAssertTrue(rejected.outputEvents.isEmpty, "no evaluation must run for rejected syntax")

        let after = await worker.evaluate(code: "sentinel + '|' + typeof broken")
        XCTAssertEqual(after.status, .completed, after.errorText ?? "")
        XCTAssertEqual(after.completionText, "original|undefined")
    }

    func testLabeledStatementWithTopLevelAwaitIsRejected() async {
        let worker = REPLWorkerContext()

        let sentinel = await worker.evaluate(code: "let sentinel = 'keep'; sentinel")
        XCTAssertEqual(sentinel.status, .completed, sentinel.errorText ?? "")

        let rejected = await worker.evaluate(code: """
        await Promise.resolve(1);
        outer: for (const value of [1, 2]) { break outer; }
        """)
        XCTAssertEqual(rejected.status, .unsupportedSyntax)
        XCTAssertTrue(
            rejected.errorText?.contains("labeled") == true,
            "expected a labeled-statement reason, got: \(rejected.errorText ?? "")")
        XCTAssertTrue(rejected.outputEvents.isEmpty)

        let after = await worker.evaluate(code: "sentinel")
        XCTAssertEqual(after.status, .completed, after.errorText ?? "")
        XCTAssertEqual(after.completionText, "keep")
    }

    func testMultiDeclaratorAwaitDeclarationIsRejectedOnTheProductionPath() async {
        let worker = REPLWorkerContext()

        let sentinel = await worker.evaluate(code: "let sentinel = 'keep'; sentinel")
        XCTAssertEqual(sentinel.status, .completed, sentinel.errorText ?? "")

        let rejected = await worker.evaluate(
            code: "let p = await Promise.resolve(1), q = 2;")
        XCTAssertEqual(rejected.status, .unsupportedSyntax)
        XCTAssertTrue(
            rejected.errorText?.contains("declarator") == true,
            "expected a declarator reason, got: \(rejected.errorText ?? "")")
        XCTAssertTrue(rejected.outputEvents.isEmpty, "no evaluation must run for rejected syntax")

        let after = await worker.evaluate(code: "sentinel + '|' + typeof p + '|' + typeof q")
        XCTAssertEqual(after.status, .completed, after.errorText ?? "")
        XCTAssertEqual(after.completionText, "keep|undefined|undefined")
    }

    func testStaticImportAndExportWithTopLevelAwaitAreRejectedOnTheProductionPath() async {
        let worker = REPLWorkerContext()

        let sentinel = await worker.evaluate(code: "let sentinel = 'keep'; sentinel")
        XCTAssertEqual(sentinel.status, .completed, sentinel.errorText ?? "")

        // Without top-level await the module syntax is simply not a classic
        // script, so it surfaces as an ordinary parse error.
        let moduleOnly = await worker.evaluate(
            code: "import helper from 'helper-module';")
        XCTAssertEqual(moduleOnly.status, .parseError)
        XCTAssertTrue(
            moduleOnly.errorText?.contains("SyntaxError") == true,
            "expected a syntax message, got: \(moduleOnly.errorText ?? "")")

        let imported = await worker.evaluate(
            code: "import helper from 'helper-module'; await helper;")
        XCTAssertEqual(imported.status, .unsupportedSyntax)
        XCTAssertTrue(
            imported.errorText?.contains("import") == true,
            "expected an import reason, got: \(imported.errorText ?? "")")
        XCTAssertTrue(imported.outputEvents.isEmpty, "no evaluation must run for rejected syntax")

        let exported = await worker.evaluate(
            code: "export const value = await Promise.resolve(1);")
        XCTAssertEqual(exported.status, .unsupportedSyntax)
        XCTAssertTrue(
            exported.errorText?.contains("export") == true,
            "expected an export reason, got: \(exported.errorText ?? "")")
        XCTAssertTrue(exported.outputEvents.isEmpty)

        let after = await worker.evaluate(code: "sentinel + '|' + typeof helper + '|' + typeof value")
        XCTAssertEqual(after.status, .completed, after.errorText ?? "")
        XCTAssertEqual(after.completionText, "keep|undefined|undefined")
    }

    func testTopLevelReturnWithTopLevelAwaitIsRejected() async {
        let worker = REPLWorkerContext()

        let sentinel = await worker.evaluate(code: "let sentinel = 'keep'; sentinel")
        XCTAssertEqual(sentinel.status, .completed, sentinel.errorText ?? "")

        let rejected = await worker.evaluate(code: """
        let value = await Promise.resolve(2);
        return value;
        """)
        XCTAssertEqual(rejected.status, .unsupportedSyntax)
        XCTAssertTrue(
            rejected.errorText?.contains("return") == true,
            "expected a top-level-return reason, got: \(rejected.errorText ?? "")")
        XCTAssertTrue(rejected.outputEvents.isEmpty)

        let after = await worker.evaluate(code: "sentinel")
        XCTAssertEqual(after.status, .completed, after.errorText ?? "")
        XCTAssertEqual(after.completionText, "keep")
    }

    // MARK: Failure and settlement bounds (2.3)

    func testRejectedAwaitedPromiseFailsAndKeepsPreAwaitBindings() async {
        let worker = REPLWorkerContext()

        let setup = await worker.evaluate(code: "let kept = 7")
        XCTAssertEqual(setup.status, .completed, setup.errorText ?? "")

        let rejected = await worker.evaluate(code: """
        let kept2 = 8;
        await Promise.reject(new Error('boom'));
        let never = 9;
        """)
        XCTAssertEqual(rejected.status, .failed)
        XCTAssertTrue(
            rejected.errorText?.contains("boom") == true,
            "expected the rejection text, got: \(rejected.errorText ?? "")")

        let kept = await worker.evaluate(code: "kept + '|' + kept2 + '|' + typeof never")
        XCTAssertEqual(kept.status, .completed, kept.errorText ?? "")
        XCTAssertEqual(kept.completionText, "7|8|undefined")
    }

    func testAwaitForeverReturnsTimedOutAndTheContextRemainsUsable() async {
        let worker = REPLWorkerContext()

        let result = await worker.evaluate(
            code: "await new Promise(() => {})",
            settlementTimeout: 0.5)

        XCTAssertEqual(result.status, .timedOut)
        // The in-context settlement deadline cannot terminate the context;
        // worker termination on timeout belongs to the supervisor (task 3.2).
        XCTAssertFalse(result.sessionReset)

        let alive = await worker.evaluate(code: "1 + 1")
        XCTAssertEqual(alive.status, .completed, alive.errorText ?? "")
        XCTAssertEqual(alive.completionText, "2")
    }
}
