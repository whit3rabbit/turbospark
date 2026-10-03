import Foundation
import JavaScriptCore
import XCTest
@testable import TurboSparkApp

/// Standing evidence for task 1.5 of the js-repl-tool spec.
///
/// Section 1 records what the public JavaScriptCore API on macOS does with
/// top-level await and promise-valued completions when scripts are evaluated
/// as classic scripts through JSContext. The public SDK headers expose only
/// evaluateScript and evaluateScript:withSourceURL: (no module evaluation
/// method, no public JSScript header), so classic script evaluation is the
/// only native route available to this target.
///
/// Sections 2 and 3 exercise the isolated bounded lowering prototype
/// (REPLTopLevelAwaitPrototype) that task 1.6 will either adopt or revise.
final class REPLTopLevelAwaitProofTests: XCTestCase {
    // MARK: Section 1: native public-API behavior

    func testClassicScriptSyntaxCheckRejectsTopLevelAwait() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let rejected = await prototype.classicSyntaxProbe(
            code: "let a = await Promise.resolve(1)")

        XCTAssertFalse(rejected.ok, "top-level await must fail the classic script syntax check")
        let message = rejected.message ?? ""
        XCTAssertTrue(message.contains("SyntaxError"), "unexpected probe message: \(message)")
        XCTAssertFalse(message.isEmpty)

        let control = await prototype.classicSyntaxProbe(
            code: "let a = Promise.resolve(1)")

        XCTAssertTrue(control.ok, control.message ?? "")
    }

    func testNativeEvaluationReturnsPromiseObjectInsteadOfSettledValue() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let result = await prototype.evaluate(code: "Promise.resolve(7)")

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        // The renderer JSON-stringifies the value: a promise object has no
        // enumerable fields, so the settled value 7 never reaches the caller.
        XCTAssertEqual(result.completionText, "{}")
        XCTAssertNotEqual(result.completionText, "7")
    }

    func testNativeEvaluationDrainsMicrotasksBeforeReturning() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let scheduled = await prototype.evaluate(code: """
        let p = Promise.resolve(9);
        p.then(value => { globalThis.proofThenRan = value; });
        'done'
        """)
        XCTAssertEqual(scheduled.status, .completed, scheduled.errorText ?? "")

        let observed = await prototype.evaluate(code: "proofThenRan")

        XCTAssertEqual(observed.status, .completed, observed.errorText ?? "")
        XCTAssertEqual(observed.completionText, "9")
    }

    func testDeferredPromiseCAPIConstructsAndResolvesFromSwift() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let marker: String = await prototype.runProbe { context in
            var resolve: JSObjectRef?
            var reject: JSObjectRef?
            guard let promiseRef = JSObjectMakeDeferredPromise(
                context.jsGlobalContextRef, &resolve, &reject, nil)
            else { return "no deferred promise" }
            guard let resolve else { return "no resolve function" }

            context.setObject(
                JSValue(jsValueRef: promiseRef, in: context),
                forKeyedSubscript: "__proofDeferred" as NSString)
            context.evaluateScript("""
            globalThis.__proofDeferredResult = 'pending';
            __proofDeferred.then(value => {
              globalThis.__proofDeferredResult = 'resolved:' + value;
            });
            """)

            var payload = JSValueMakeNumber(context.jsGlobalContextRef, 11)
            var exception: JSValueRef?
            JSObjectCallAsFunction(
                context.jsGlobalContextRef, resolve, nil, 1, &payload, &exception)
            if let exception {
                let text = JSValue(jsValueRef: exception, in: context).toString() ?? "unknown"
                return "resolve threw: \(text)"
            }

            context.evaluateScript("0")
            return context.evaluateScript("__proofDeferredResult")?.toString() ?? "nil"
        }

        XCTAssertEqual(marker, "resolved:11")
    }

    func testDynamicImportCannotServeAsAModuleRoute() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let result = await prototype.evaluate(code: """
        globalThis.proofImportOutcome = 'unset';
        try {
          const loaded = await import('data:text/javascript,export default 1');
          globalThis.proofImportOutcome = 'loaded:' + String(loaded);
        } catch (error) {
          globalThis.proofImportOutcome = 'rejected:' + String(error);
        }
        'evaluated'
        """)

        let outcome = await prototype.evaluate(code: "proofImportOutcome")

        switch result.status {
        case .completed:
            XCTAssertEqual(outcome.status, .completed, outcome.errorText ?? "")
            let text = outcome.completionText ?? ""
            XCTAssertFalse(
                text.hasPrefix("loaded"),
                "dynamic import must not load a module in this kernel: \(text)")
        case .parseError, .failed:
            break
        default:
            XCTFail("unexpected status for the dynamic import probe: \(result.status)")
        }
    }

    // MARK: Section 2: bounded lowering proof

    func testLoweredTopLevelAwaitSettlesAndReturnsAwaitedCompletion() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let result = await prototype.evaluate(code: """
        let before = 10;
        let after = await Promise.resolve(32);
        after + before
        """)

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(result.completionText, "42")
        XCTAssertFalse(result.sessionReset)
    }

    func testDeclarationsBeforeAndAfterAwaitPersistAcrossLaterCalls() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let first = await prototype.evaluate(code: """
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

        let second = await prototype.evaluate(code: """
        [shared, before, awaited, after, doubled(21), Marker.name,
         Object.prototype.hasOwnProperty.call(globalThis, 'after')].join(',')
        """)
        XCTAssertEqual(second.status, .completed, second.errorText ?? "")
        XCTAssertFalse(second.sessionCreated)
        XCTAssertEqual(
            second.completionText,
            "v,pre,mid,post,42,Marker,true")
    }

    func testHostDelayedPromiseSettlesAwaitWithinDeadline() async {
        let prototype = REPLTopLevelAwaitPrototype()
        let started = Date()

        let result = await prototype.evaluate(
            code: """
            let slow = await __replProofDelayedResolve(120, 41);
            slow + 1
            """,
            settlementTimeout: 5)

        let elapsed = Date().timeIntervalSince(started)
        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(result.completionText, "42")
        XCTAssertGreaterThanOrEqual(elapsed, 0.11)

        let stored = await prototype.evaluate(code: "slow")
        XCTAssertEqual(stored.status, .completed, stored.errorText ?? "")
        XCTAssertEqual(stored.completionText, "41")
    }

    func testOrdinaryScriptsStayOnTheNativePathWithoutLowering() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let declared = await prototype.evaluate(code: "let u = 3; u")
        XCTAssertEqual(declared.status, .completed, declared.errorText ?? "")
        XCTAssertEqual(declared.completionText, "3")

        let lexical = await prototype.evaluate(
            code: "Object.prototype.hasOwnProperty.call(globalThis, 'u')")
        XCTAssertEqual(lexical.status, .completed, lexical.errorText ?? "")
        XCTAssertEqual(
            lexical.completionText, "false",
            "ordinary scripts must keep native lexical (non-property) declarations")
    }

    func testNativeLexicalRedeclarationAcrossCallsStillThrows() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let first = await prototype.evaluate(code: "let r = 1")
        XCTAssertEqual(first.status, .completed, first.errorText ?? "")

        let second = await prototype.evaluate(code: "let r = 2")
        XCTAssertEqual(second.status, .failed)
        XCTAssertTrue(
            second.errorText?.contains("SyntaxError") == true,
            "unexpected redeclaration outcome: \(second.errorText ?? "")")
    }

    func testLoweredFinalDeclarationCompletesAsUndefined() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let result = await prototype.evaluate(code: "let a = await Promise.resolve(1)")

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(result.completionText, "undefined")
    }

    func testRejectedAwaitedPromiseFailsAndKeepsPreAwaitBindings() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let result = await prototype.evaluate(code: """
        let kept = 7;
        await Promise.reject(new Error('boom'));
        let never = 8;
        """)

        XCTAssertEqual(result.status, .failed)
        XCTAssertTrue(
            result.errorText?.contains("boom") == true,
            "expected the rejection text, got: \(result.errorText ?? "")")

        let kept = await prototype.evaluate(code: "kept")
        XCTAssertEqual(kept.status, .completed, kept.errorText ?? "")
        XCTAssertEqual(kept.completionText, "7")

        let never = await prototype.evaluate(code: "typeof never")
        XCTAssertEqual(never.status, .completed, never.errorText ?? "")
        XCTAssertEqual(never.completionText, "undefined")
    }

    // MARK: Section 3: bounded subset rejection and recorded limits

    func testDestructuringDeclarationIsRejectedWithoutChangingSessionState() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let sentinel = await prototype.evaluate(code: "let sentinel = 'original'; sentinel")
        XCTAssertEqual(sentinel.status, .completed, sentinel.errorText ?? "")
        XCTAssertEqual(sentinel.completionText, "original")

        let rejected = await prototype.evaluate(code: """
        let broken = await Promise.resolve(1);
        let { a } = await Promise.resolve({ a: 2 });
        """)
        XCTAssertEqual(rejected.status, .unsupportedSyntax)
        XCTAssertTrue(
            rejected.errorText?.contains("destructuring") == true,
            "expected a destructuring reason, got: \(rejected.errorText ?? "")")
        XCTAssertTrue(rejected.outputEvents.isEmpty, "no evaluation must run for rejected syntax")

        let after = await prototype.evaluate(code: "sentinel + '|' + typeof broken")
        XCTAssertEqual(after.status, .completed, after.errorText ?? "")
        XCTAssertEqual(after.completionText, "original|undefined")
    }

    func testMultiDeclaratorAwaitDeclarationIsRejected() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let result = await prototype.evaluate(
            code: "let p = await Promise.resolve(1), q = 2;")

        XCTAssertEqual(result.status, .unsupportedSyntax)
        XCTAssertTrue(
            result.errorText?.contains("declarator") == true,
            "expected a declarator reason, got: \(result.errorText ?? "")")

        let absent = await prototype.evaluate(code: "typeof p + '|' + typeof q")
        XCTAssertEqual(absent.status, .completed, absent.errorText ?? "")
        XCTAssertEqual(absent.completionText, "undefined|undefined")
    }

    func testStaticModuleImportIsRejectedWithoutEvaluation() async {
        let withoutAwait = REPLTopLevelAwaitPrototype()
        let classic = await withoutAwait.evaluate(
            code: "import helper from 'helper-module';")
        // Without top-level await the script is simply not a classic script.
        XCTAssertEqual(classic.status, .parseError)
        XCTAssertTrue(classic.errorText?.contains("SyntaxError") == true)

        let withAwait = REPLTopLevelAwaitPrototype()
        let bounded = await withAwait.evaluate(
            code: "import helper from 'helper-module'; await helper;")
        XCTAssertEqual(bounded.status, .unsupportedSyntax)
        XCTAssertTrue(
            bounded.errorText?.contains("import") == true,
            "expected an import reason, got: \(bounded.errorText ?? "")")
    }

    func testAwaitForeverTimesOutAndContextRemainsUsable() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let result = await prototype.evaluate(
            code: "await new Promise(() => {})",
            settlementTimeout: 0.5)

        XCTAssertEqual(result.status, .timedOut)
        // The in-process prototype cannot terminate its context; the
        // production path terminates the worker process instead (task 3.2).
        XCTAssertFalse(result.sessionReset)

        let alive = await prototype.evaluate(code: "1 + 1")
        XCTAssertEqual(alive.status, .completed, alive.errorText ?? "")
        XCTAssertEqual(alive.completionText, "2")
    }

    func testLoweredAndNativeLexicalBindingsOfTheSameNameDiverge() async {
        let prototype = REPLTopLevelAwaitPrototype()

        let native = await prototype.evaluate(code: "let x = 1; x")
        XCTAssertEqual(native.status, .completed, native.errorText ?? "")
        XCTAssertEqual(native.completionText, "1")

        let lowered = await prototype.evaluate(code: "let x = await Promise.resolve(5); x")
        XCTAssertEqual(lowered.status, .completed, lowered.errorText ?? "")
        // Recorded limit for task 1.6: the lowering persists the value as a
        // global property, while the native lexical binding still shadows it.
        XCTAssertEqual(lowered.completionText, "1")

        let property = await prototype.evaluate(code: "globalThis.x")
        XCTAssertEqual(property.status, .completed, property.errorText ?? "")
        XCTAssertEqual(property.completionText, "5")
    }
}
