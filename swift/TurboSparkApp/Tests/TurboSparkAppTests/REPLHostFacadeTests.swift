import JavaScriptCore
import XCTest
@testable import TurboSparkApp

/// Task 2.1: console capture order across levels (6.1), the output cap
/// with the truncation flag (6.3), the warn/error error channel, and the
/// frozen facade surface that tampering cannot weaken for later calls
/// (5.6).
final class REPLHostFacadeTests: XCTestCase {
    private let harness = REPLFacadeHarness()

    // MARK: Facade installation on a bare context (6.1, 5.6)

    func testConsoleCapturesEveryLevelInCrossLevelCallOrder() {
        harness.evaluate("""
        console.debug("d");
        console.log("l");
        console.warn("w");
        console.error("e");
        console.info("i");
        """)

        XCTAssertEqual(
            harness.events.map(\.level),
            [.debug, .log, .warn, .error, .info],
            "outputEvents must preserve exact cross-level call order (6.1)")
        XCTAssertEqual(harness.events.map(\.text), ["d", "l", "w", "e", "i"])
        XCTAssertTrue(harness.exceptions.isEmpty)
    }

    func testConsoleSurfaceMembersAreNonWritableFrozenAndUndeletable() {
        let probes: [(String, String)] = [
            ("console object is frozen", "Object.isFrozen(console)"),
            ("console object is not extensible", "!Object.isExtensible(console)"),
            (
                "every console method is a function with a non-writable "
                    + "non-configurable descriptor",
                """
                (() => {
                    const levels = ['log', 'info', 'debug', 'warn', 'error'];
                    return levels.every(name => {
                        const descriptor = Object.getOwnPropertyDescriptor(console, name);
                        return descriptor !== undefined
                            && typeof descriptor.value === 'function'
                            && descriptor.writable === false
                            && descriptor.configurable === false;
                    });
                })()
                """),
            (
                "console method functions themselves are frozen",
                """
                (() => {
                    const levels = ['log', 'info', 'debug', 'warn', 'error'];
                    return levels.every(name => Object.isFrozen(console[name]));
                })()
                """),
            (
                "global console binding is non-writable and non-configurable",
                """
                (() => {
                    const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'console');
                    return descriptor !== undefined
                        && descriptor.writable === false
                        && descriptor.configurable === false;
                })()
                """),
            (
                "sloppy delete of a console method fails",
                "!(delete console.log)"),
            (
                "the installation sink is not left on the global",
                """
                !Object.prototype.hasOwnProperty.call(globalThis, '__turbosparkConsoleSink')
                """)
        ]

        for (description, script) in probes {
            let passed = harness.evaluate(script)?.toBool() ?? false
            XCTAssertTrue(passed, "expected true: \(description)")
        }
        XCTAssertTrue(harness.events.isEmpty, "probes must not capture output")
        XCTAssertTrue(harness.exceptions.isEmpty)
    }

    func testStrictReassignmentThrowsAndNoNewMembersCanBeDefined() {
        harness.evaluate(#""use strict"; console.log = () => {};"#)
        harness.evaluate(#""use strict"; globalThis.console = {};"#)
        harness.evaluate(#""use strict"; delete console.warn;"#)
        harness.evaluate("Object.defineProperty(console, 'extra', { value: 1 });")

        XCTAssertEqual(
            harness.exceptions.count, 4,
            "every tampering vector must throw: \(harness.exceptions)")

        harness.evaluate("console.log('intact'); console.error('routed');")
        XCTAssertEqual(
            harness.events.map(\.text), ["intact", "routed"],
            "the original members must still capture after tampering attempts")
    }

    func testSloppyReassignmentFailsSilentlyAndOriginalMembersKeepCapturing() {
        let survived = harness.evaluate("""
        console.log = () => {};
        globalThis.console = {};
        delete console.log;
        "survived"
        """)

        XCTAssertEqual(survived?.toString(), "survived")
        XCTAssertTrue(
            harness.exceptions.isEmpty,
            "sloppy-mode tampering fails silently rather than throwing")

        harness.evaluate("console.log('captured after silent failures')")
        XCTAssertEqual(harness.events.map(\.text), ["captured after silent failures"])
    }

    // MARK: Worker results: error channel, cap, truncation flag (6.1, 6.3)

    func testWarnAndErrorLevelsFeedTheErrorChannelInCallOrder() async {
        let worker = REPLWorkerContext()

        let result = await worker.evaluate(code: """
        console.info("i");
        console.debug("d");
        console.error("e");
        console.warn("w");
        """)

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(result.outputEvents.map(\.level), [.info, .debug, .error, .warn])
        XCTAssertEqual(result.consoleText, "i\nd")
        XCTAssertEqual(result.errorText, "e\nw")
        XCTAssertFalse(result.truncated)
    }

    func testCapturedOutputPastTheCapIsCompactedHeadAndTailAndFlaggedTruncated() async {
        // Cap 30 keeps a 20-character head and a 7-character tail of the
        // 54-character captured stream; 27 characters are removed.
        let worker = REPLWorkerContext(limits: REPLLimits(maximumOutputCharacters: 30))

        let result = await worker.evaluate(code: """
        console.log("A".repeat(10));
        console.warn("B".repeat(10));
        console.log("C".repeat(10));
        console.error("D".repeat(10));
        console.log("E".repeat(10));
        """)

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertTrue(
            result.truncated,
            "captured output past the cap must set the truncation flag (6.3)")
        XCTAssertEqual(result.outputEvents, [
            REPLTextOutputEvent(level: .log, text: "AAAAAAAAAA"),
            REPLTextOutputEvent(level: .warn, text: "BBBBBBBBB"),
            REPLTextOutputEvent(level: .log, text: "... [27 chars truncated] ..."),
            REPLTextOutputEvent(level: .log, text: "EEEEEEE")
        ])
        XCTAssertEqual(
            result.consoleText,
            "AAAAAAAAAA\n... [27 chars truncated] ...\nEEEEEEE")
        XCTAssertEqual(
            result.errorText, "BBBBBBBBB",
            "a warn event sliced by the cut still feeds the error channel")
    }

    func testCapturedOutputWithinTheCapKeepsEveryEventUntruncated() async {
        let worker = REPLWorkerContext(limits: REPLLimits(maximumOutputCharacters: 100))

        let result = await worker.evaluate(code: """
        console.log("A".repeat(10));
        console.warn("B".repeat(10));
        console.log("C".repeat(10));
        console.error("D".repeat(10));
        console.log("E".repeat(10));
        """)

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertFalse(result.truncated)
        XCTAssertEqual(result.outputEvents.count, 5)
        XCTAssertEqual(result.outputEvents.map(\.level), [.log, .warn, .log, .error, .log])
        XCTAssertEqual(result.consoleText, "AAAAAAAAAA\nCCCCCCCCCC\nEEEEEEEEEE")
        XCTAssertEqual(result.errorText, "BBBBBBBBBB\nDDDDDDDDDD")
    }

    func testSingleOversizedEventIsCompactedWithinItsOwnText() async {
        // Cap 40 keeps a 26-character head and a 10-character tail of the
        // 200-character single event; 164 characters are removed.
        let worker = REPLWorkerContext(limits: REPLLimits(maximumOutputCharacters: 40))

        let result = await worker.evaluate(code: #"console.error("x".repeat(200))"#)

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertTrue(result.truncated)
        XCTAssertEqual(result.outputEvents.map(\.level), [.error, .log, .error])
        XCTAssertEqual(result.outputEvents.map(\.text), [
            String(repeating: "x", count: 26),
            "... [164 chars truncated] ...",
            String(repeating: "x", count: 10)
        ])
    }

    func testStrictReassignmentFailsTheCallAndTheNextCallUsesTheOriginalMember() async {
        let worker = REPLWorkerContext()

        let tampered = await worker.evaluate(
            code: #""use strict"; console.error = () => {};"#)
        XCTAssertEqual(tampered.status, .failed)
        XCTAssertTrue(
            tampered.errorText?.contains("TypeError") == true,
            "expected a TypeError from strict reassignment, got: "
                + (tampered.errorText ?? ""))

        let followUp = await worker.evaluate(code: "console.error('original intact')")
        XCTAssertEqual(followUp.status, .completed, followUp.errorText ?? "")
        XCTAssertEqual(
            followUp.outputEvents,
            [REPLTextOutputEvent(level: .error, text: "original intact")])
        XCTAssertEqual(followUp.errorText, "original intact")
        XCTAssertFalse(followUp.truncated)
    }
}

/// A bare JavaScriptCore context with the production facade installed and
/// the sink wired to observable buffers.
private final class REPLFacadeHarness: @unchecked Sendable {
    let context: JSContext
    private let lock = NSLock()
    private var eventStorage: [REPLTextOutputEvent] = []
    private var exceptionStorage: [String] = []

    init() {
        guard let context = JSContext() else {
            fatalError("JavaScriptCore could not create a facade test context")
        }
        self.context = context
        let harness = self
        context.exceptionHandler = { _, exception in
            harness.recordException(exception?.toString() ?? "unknown exception")
        }
        REPLHostFacade(output: { event in harness.recordEvent(event) })
            .installConsole(into: context)
    }

    @discardableResult
    func evaluate(_ script: String) -> JSValue? {
        context.evaluateScript(script)
    }

    var events: [REPLTextOutputEvent] {
        lock.lock()
        defer { lock.unlock() }
        return eventStorage
    }

    var exceptions: [String] {
        lock.lock()
        defer { lock.unlock() }
        return exceptionStorage
    }

    private func recordEvent(_ event: REPLTextOutputEvent) {
        lock.lock()
        eventStorage.append(event)
        lock.unlock()
    }

    private func recordException(_ message: String) {
        lock.lock()
        exceptionStorage.append(message)
        lock.unlock()
    }
}
