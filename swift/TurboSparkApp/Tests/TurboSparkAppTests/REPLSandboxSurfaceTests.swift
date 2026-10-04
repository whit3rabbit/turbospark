import JavaScriptCore
import XCTest
@testable import TurboSparkApp

/// Task 2.4: the deny-by-default sandbox surface (5.1, 5.2, 5.6). These
/// are standing refusal-path tests so the negative surface cannot regress
/// silently: no network, process, or module-loading global exists; dynamic
/// import rejects; constructor chains and prototype retargeting reach no
/// host capability beyond the deliberate console and repl surface. The
/// engine itself provides nothing, so absence is asserted rather than
/// implemented; this file does not claim protection from script-engine
/// vulnerabilities or an operating-system sandbox.
final class REPLSandboxSurfaceTests: XCTestCase {
    // MARK: Helpers

    private func makeArtifactDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("repl-sandbox-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    private func makeWorker(artifacts: URL) -> REPLWorkerContext {
        // No file channel: the fail-closed broker denies every file
        // operation, which is the posture requirement 5.5 demands for an
        // ungranted capability.
        REPLWorkerContext(
            limits: REPLLimits(),
            configuration: REPLSessionConfiguration(artifactDirectory: artifacts))
    }

    private func evaluate(_ worker: REPLWorkerContext, _ code: String) async -> REPLCallResult {
        await worker.evaluate(code: code)
    }

    private func globalNames(_ worker: REPLWorkerContext) async -> Set<String> {
        let result = await evaluate(worker, "Object.getOwnPropertyNames(globalThis).join(',')")
        precondition(result.status == .completed, result.errorText ?? "probe failed")
        return Set((result.completionText ?? "").split(separator: ",").map(String.init))
    }

    // MARK: No network, process, or module-loading surface (5.1, 5.2)

    func testGlobalExposesNoNetworkProcessOrModuleLoadingSurface() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        let result = await evaluate(worker, """
        [
            "fetch", "XMLHttpRequest", "WebSocket", "Request", "Response",
            "process", "require", "module", "exports", "Buffer", "Deno",
            "globalThis.process"
        ].map(name => name + ":" + eval("typeof " + name)).join(",")
            + "|repl:" + typeof repl
            + "|console:" + typeof console
        """)

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        let surface = result.completionText ?? ""
        for absent in [
            "fetch:undefined", "XMLHttpRequest:undefined", "WebSocket:undefined",
            "Request:undefined", "Response:undefined", "process:undefined",
            "require:undefined", "module:undefined", "exports:undefined",
            "Buffer:undefined", "Deno:undefined"
        ] {
            XCTAssertTrue(
                surface.contains(absent),
                "the sandbox must not expose \(absent), got: \(surface)")
        }
        XCTAssertTrue(surface.contains("repl:object"), "the deliberate repl surface exists")
        XCTAssertTrue(surface.contains("console:object"), "the deliberate console surface exists")
    }

    func testDynamicImportRejectsInsteadOfLoadingModules() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        let armed = await evaluate(worker, """
        globalThis.importProbe = "pending";
        import("./sneaky.js").then(
            value => { globalThis.importProbe = "resolved:" + value; },
            error => { globalThis.importProbe = "rejected:" + error.message; }
        );
        "armed"
        """)
        XCTAssertEqual(armed.status, .completed, armed.errorText ?? "")

        let outcome = await evaluate(worker, "globalThis.importProbe")
        XCTAssertEqual(outcome.status, .completed, outcome.errorText ?? "")
        XCTAssertTrue(
            (outcome.completionText ?? "").hasPrefix("rejected:"),
            "dynamic import must reject rather than load a module, got: "
                + (outcome.completionText ?? ""))
        XCTAssertNotEqual(outcome.completionText, "pending",
                          "the import probe must settle, not hang")
    }

    // MARK: The host adds exactly the deliberate surface (5.2, 5.6)

    func testHostAddedGlobalsAreExactlyConsoleAndRepl() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        guard let bare = JSContext() else {
            return XCTFail("JavaScriptCore could not create a bare reference context")
        }
        let bareNamesText: String
        if let bareProbe = bare.evaluateScript(
            "Object.getOwnPropertyNames(globalThis).join(',')")
        {
            bareNamesText = bareProbe.toString()
        } else {
            bareNamesText = ""
        }
        let bareNames = Set(bareNamesText.split(separator: ",").map(String.init))

        let workerNames = await globalNames(worker)

        XCTAssertTrue(bareNames.isSubset(of: workerNames),
                      "the worker must not remove engine globals, removed: "
                          + bareNames.subtracting(workerNames).sorted().joined(separator: ","))
        // The macOS JavaScriptCore already ships an enumerable `console`
        // in a bare context, so it may or may not appear as an addition.
        // Everything the host adds beyond the deliberate surface must be
        // one of the two documented non-enumerable internal bridges the
        // lowered wrapper settles through; they expose no capability and
        // are pinned tamper-proof below.
        let allowedAdditions: Set<String> = [
            "console", "repl", "__turbosparkReplRender", "__turbosparkReplSettled"
        ]
        let added = workerNames.subtracting(bareNames).sorted()
        XCTAssertTrue(
            added.allSatisfy(allowedAdditions.contains),
            "the only host-added globals must be the deliberate surface and "
                + "the documented internal bridges, got: " + added.joined(separator: ","))
        XCTAssertTrue(added.contains("repl"), "the deliberate repl surface must be added")

        let bridges = await evaluate(worker, """
        (() => {
            const report = [];
            for (const name of ["__turbosparkReplRender", "__turbosparkReplSettled"]) {
                const descriptor = Object.getOwnPropertyDescriptor(globalThis, name);
                report.push(
                    name
                        + ":" + (descriptor === undefined ? "absent" : "present")
                        + ":enumerable=" + (descriptor ? descriptor.enumerable : "n/a")
                        + ":writable=" + (descriptor ? descriptor.writable : "n/a")
                        + ":configurable=" + (descriptor ? descriptor.configurable : "n/a")
                        + ":" + (descriptor ? typeof descriptor.value : "n/a"));
            }
            return report.join("|");
        })()
        """)
        XCTAssertEqual(bridges.status, .completed, bridges.errorText ?? "")
        for bridge in (bridges.completionText ?? "").split(separator: "|") {
            XCTAssertTrue(
                bridge.hasSuffix(":enumerable=false:writable=false:configurable=false:function"),
                "internal bridges must stay non-enumerable, sealed functions: \(bridge)")
        }
    }

    func testReplExposesExactlyTheFileAndImageCapabilities() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        let result = await evaluate(worker, "Object.keys(repl).sort().join(',')")

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(
            result.completionText, "emitImage,fs",
            "repl must expose exactly the declared capabilities")
    }

    // MARK: Constructor chains reach no host capability (5.6)

    func testConstructorChainsReachNoHostCapability() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        let result = await evaluate(worker, """
        (() => {
            const FunctionConstructor = ({}).constructor.constructor;
            const constructedGlobal = FunctionConstructor("return this")();
            const throughConsole = console.log.constructor("return this")();
            const names = [
                "fetch", "XMLHttpRequest", "WebSocket", "process",
                "require", "module", "Buffer"
            ];
            const reachable = [];
            for (const name of names) {
                if (typeof constructedGlobal[name] !== "undefined") reachable.push(name);
                if (typeof throughConsole[name] !== "undefined") reachable.push("console:" + name);
            }
            return reachable.join(",")
                + "|constructed-repl:" + (typeof constructedGlobal.repl)
                + "|constructed-console:" + (typeof constructedGlobal.console);
        })()
        """)

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        let outcome = result.completionText ?? ""
        XCTAssertTrue(
            outcome.hasPrefix("|"),
            "no probed host capability may be reachable through constructor chains, got: "
                + outcome)
        XCTAssertTrue(outcome.contains("constructed-repl:object"))
        XCTAssertTrue(outcome.contains("constructed-console:object"))
    }

    // MARK: Prototype retargeting cannot swap the surface (5.6)

    func testPrototypeRetargetingIsRefusedAndOriginalsSurvive() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        // JavaScriptCore refuses __proto__ assignment on a frozen object
        // with a TypeError even in sloppy mode; either way the retargeting
        // must not succeed.
        let sloppy = await evaluate(worker, """
        globalThis.protoProbe = [];
        const attempts = [
            () => { console.__proto__ = { log: () => "hijacked" }; return "succeeded"; },
            () => { Object.setPrototypeOf(console, { log: () => "hijacked" }); return "succeeded"; }
        ];
        for (const attempt of attempts) {
            try { globalThis.protoProbe.push(attempt()); }
            catch (error) { globalThis.protoProbe.push("refused"); }
        }
        globalThis.protoProbe.join(",")
        """)
        XCTAssertEqual(sloppy.status, .completed, sloppy.errorText ?? "")
        XCTAssertEqual(
            sloppy.completionText, "refused,refused",
            "every prototype retargeting vector must be refused, got: "
                + (sloppy.completionText ?? ""))

        // The prototype was never swapped.
        let unchanged = await evaluate(
            worker, "Object.getPrototypeOf(console) === Object.prototype")
        XCTAssertEqual(unchanged.status, .completed, unchanged.errorText ?? "")
        XCTAssertEqual(unchanged.completionText, "true",
                       "the frozen console prototype must survive retargeting attempts")

        // Strict-mode __proto__ assignment throws, failing only that call.
        let strict = await evaluate(
            worker, #""use strict"; console.__proto__ = { log: () => {} };"#)
        XCTAssertEqual(
            strict.status, .failed,
            "strict-mode prototype retargeting must fail the call")
        XCTAssertTrue(
            (strict.errorText ?? "").contains("prototype") || (strict.errorText ?? "").contains("__proto__")
                || (strict.errorText ?? "").contains("readonly"),
            "the failure must identify the retargeting, got: " + (strict.errorText ?? ""))

        // The original members still capture afterwards.
        let after = await evaluate(
            worker, #"console.log("still-capturing"); "ok";"#)
        XCTAssertEqual(after.status, .completed, after.errorText ?? "")
        XCTAssertEqual(after.consoleText, "still-capturing",
                       "the original console must keep capturing after tampering")
        XCTAssertEqual(after.completionText, "ok")
    }
}
