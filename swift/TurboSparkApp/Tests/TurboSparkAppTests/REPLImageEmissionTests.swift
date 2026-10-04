import JavaScriptCore
import XCTest
@testable import TurboSparkApp

/// Task 2.3: the image emission interface. Valid PNG and JPEG payloads
/// land as files in the per-chat artifact directory and are referenced in
/// the call result (6.2); oversized, wrong-magic, and undecodable payloads
/// reject with explanatory errors while the rest of the result survives
/// (6.4). Payloads are validated by PNG or JPEG magic bytes, never by the
/// claimed label, and the file name never derives from the label.
final class REPLImageEmissionTests: XCTestCase {
    private static let pngMagic: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    private static let jpegMagic: [UInt8] = [0xFF, 0xD8, 0xFF, 0xE0]

    // MARK: Helpers

    private func makeArtifactDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("repl-image-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    /// Base64 of a PNG signature plus `payloadCount` filler bytes.
    private func pngBase64(payloadCount: Int = 24) -> String {
        Self.base64(Self.pngMagic + Array(repeating: 0x54, count: payloadCount))
    }

    /// Base64 of a JPEG signature plus `payloadCount` filler bytes.
    private func jpegBase64(payloadCount: Int = 24) -> String {
        Self.base64(Self.jpegMagic + Array(repeating: 0x46, count: payloadCount))
    }

    private static func base64(_ bytes: [UInt8]) -> String {
        Data(bytes).base64EncodedString()
    }

    private func expectedBytes(magic: [UInt8], payloadCount: Int, filler: UInt8) -> Data {
        Data(magic + Array(repeating: filler, count: payloadCount))
    }

    private func makeWorker(
        artifacts: URL,
        maximumImageBytes: Int = 20 * 1_024 * 1_024
    ) -> REPLWorkerContext {
        REPLWorkerContext(
            limits: REPLLimits(maximumImageBytes: maximumImageBytes),
            configuration: REPLSessionConfiguration(artifactDirectory: artifacts))
    }

    private func filesInDirectory(_ url: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
    }

    // MARK: Valid emissions land as files with references (6.2)

    func testValidPngLandsAsFileWithLabelAndReferenceInResult() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)
        let expected = expectedBytes(magic: Self.pngMagic, payloadCount: 24, filler: 0x54)

        let result = await worker.evaluate(
            code: "repl.emitImage(\"\(pngBase64())\", \"chart\")")

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(result.images.count, 1, "the emitted image must be referenced in the result")
        let image = try XCTUnwrap(result.images.first)
        XCTAssertEqual(image.label, "chart")
        XCTAssertEqual(
            image.fileURL.pathExtension,
            "png",
            "the extension must come from the detected magic bytes")
        XCTAssertEqual(
            result.completionText,
            image.fileURL.path,
            "repl.emitImage returns the file reference")
        XCTAssertTrue(
            image.fileURL.path.hasPrefix(artifacts.path + "/"),
            "the file must land inside the per-chat artifact directory, got \(image.fileURL.path)")
        XCTAssertEqual(
            try Data(contentsOf: image.fileURL),
            expected,
            "the written file must keep the emitted bytes exactly")
    }

    func testValidJpegLandsAsFileWithReferenceInResult() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)
        let expected = expectedBytes(magic: Self.jpegMagic, payloadCount: 24, filler: 0x46)

        let result = await worker.evaluate(
            code: "repl.emitImage(\"\(jpegBase64())\", \"photo\")")

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(result.images.count, 1)
        let image = try XCTUnwrap(result.images.first)
        XCTAssertEqual(image.label, "photo")
        XCTAssertEqual(
            image.fileURL.pathExtension, "jpg", "a JPEG payload keeps a JPEG extension")
        XCTAssertEqual(result.completionText, image.fileURL.path)
        XCTAssertTrue(image.fileURL.path.hasPrefix(artifacts.path + "/"))
        XCTAssertEqual(try Data(contentsOf: image.fileURL), expected)
    }

    func testEmissionWithoutALabelKeepsANilLabel() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        let result = await worker.evaluate(code: "repl.emitImage(\"\(pngBase64())\")")

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(result.images.count, 1)
        XCTAssertNil(result.images.first?.label, "an omitted label stays nil")
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.images.first?.fileURL.path ?? ""))
    }

    func testEmissionWorksThroughTheTopLevelAwaitPath() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        let result = await worker.evaluate(code: """
        const path = repl.emitImage("\(pngBase64())", "awaited");
        await Promise.resolve(1);
        path
        """)

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(result.images.count, 1)
        XCTAssertEqual(result.images.first?.label, "awaited")
        XCTAssertEqual(result.completionText, result.images.first?.fileURL.path)
    }

    // MARK: The call result survives later failures and rejections (6.4)

    func testImageEmittedBeforeALaterScriptErrorSurvivesInTheResult() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        let result = await worker.evaluate(code: """
        console.log("progress");
        repl.emitImage("\(pngBase64())", "early");
        throw new Error("later failure")
        """)

        XCTAssertEqual(result.status, .failed)
        XCTAssertTrue(
            result.errorText?.contains("later failure") == true,
            "the later error must surface, got: \(result.errorText ?? "")")
        XCTAssertEqual(result.consoleText, "progress")
        XCTAssertEqual(
            result.images.count, 1,
            "a valid image emitted before the error must stay in the result")
        XCTAssertEqual(result.images.first?.label, "early")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: result.images.first?.fileURL.path ?? ""),
            "the emitted file must still exist on disk")
    }

    func testOverCapPayloadRejectsWithExplanatoryErrorAndTheRestSurvives() async throws {
        let artifacts = try makeArtifactDirectory()
        // Cap 64: the kept image is 32 bytes, the rejected image is 65.
        let worker = makeWorker(artifacts: artifacts, maximumImageBytes: 64)

        let result = await worker.evaluate(code: """
        console.log("kept");
        repl.emitImage("\(pngBase64(payloadCount: 24))", "kept-image");
        repl.emitImage("\(pngBase64(payloadCount: 57))", "over-cap");
        """)

        XCTAssertEqual(result.status, .failed, "the uncaught rejection must fail the call")
        XCTAssertTrue(
            result.errorText?.contains("exceeds") == true,
            "the rejection must explain the cap, got: \(result.errorText ?? "")")
        XCTAssertTrue(
            result.errorText?.contains("65") == true,
            "the rejection must name the payload size, got: \(result.errorText ?? "")")
        XCTAssertTrue(
            result.errorText?.contains("64") == true,
            "the rejection must name the limit, got: \(result.errorText ?? "")")
        XCTAssertEqual(
            result.consoleText, "kept", "earlier console output must survive the rejection")
        XCTAssertEqual(
            result.images.count, 1, "the earlier valid image must survive the rejection")
        XCTAssertEqual(result.images.first?.label, "kept-image")
        XCTAssertEqual(
            try filesInDirectory(artifacts).count, 1,
            "only the valid image may land on disk")
    }

    func testGiantPayloadIsRejectedBeforeItsBytesAreDecoded() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts, maximumImageBytes: 64)

        let result = await worker.evaluate(
            code: "repl.emitImage(\"\(pngBase64(payloadCount: 2_000))\", \"giant\")")

        XCTAssertEqual(result.status, .failed)
        XCTAssertTrue(
            result.errorText?.contains("exceeds") == true,
            "an oversized payload must reject without decoding, got: \(result.errorText ?? "")")
        XCTAssertTrue(result.images.isEmpty)
        XCTAssertTrue(try filesInDirectory(artifacts).isEmpty)
    }

    func testWrongMagicBytesRejectEvenWhenTheLabelClaimsAnImageFormat() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)
        let textBytes = Self.base64(Array("this is definitely not an image".utf8))
        let gifBytes = Self.base64(Array("GIF89a".utf8) + Array(repeating: 0x00, count: 16))

        let textResult = await worker.evaluate(
            code: "repl.emitImage(\"\(textBytes)\", \"notes.png\")")
        XCTAssertEqual(
            textResult.status, .failed, "text bytes must reject regardless of the claimed label")
        XCTAssertTrue(
            textResult.errorText?.contains("PNG") == true
                && textResult.errorText?.contains("JPEG") == true,
            "the rejection must name the accepted magic formats, got: \(textResult.errorText ?? "")")

        let gifResult = await worker.evaluate(
            code: "repl.emitImage(\"\(gifBytes)\", \"diagram.jpeg\")")
        XCTAssertEqual(
            gifResult.status, .failed, "a GIF header must reject even labeled as jpeg")
        XCTAssertTrue(gifResult.errorText?.contains("magic") == true)

        XCTAssertTrue(
            textResult.images.isEmpty && gifResult.images.isEmpty,
            "rejected payloads must produce no image references")
        XCTAssertTrue(
            try filesInDirectory(artifacts).isEmpty,
            "rejected payloads must not write files")
    }

    func testBase64DecodingFailureRejectsWithAnExplanatoryError() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        let result = await worker.evaluate(
            code: "repl.emitImage(\"!!!not base64!!!\", \"broken\")")

        XCTAssertEqual(result.status, .failed)
        XCTAssertTrue(
            result.errorText?.contains("base64") == true,
            "a decoding failure must explain itself, got: \(result.errorText ?? "")")
        XCTAssertTrue(result.images.isEmpty)
        XCTAssertTrue(try filesInDirectory(artifacts).isEmpty)
    }

    // MARK: The label never reaches the file path

    func testTraversalLabelCannotEscapeTheArtifactDirectory() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        let result = await worker.evaluate(
            code: "repl.emitImage(\"\(pngBase64())\", \"../../escape.png\")")

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(
            result.images.count, 1, "the emission itself is valid; only the label is hostile")
        let image = try XCTUnwrap(result.images.first)
        XCTAssertEqual(
            image.label, "../../escape.png", "the label rides along verbatim")
        XCTAssertEqual(
            image.fileURL.lastPathComponent.hasPrefix("repl-image-"),
            true,
            "the file name is generated, never derived from the label")
        XCTAssertTrue(
            image.fileURL.path.hasPrefix(artifacts.path + "/"),
            "the file must stay inside the artifact directory")
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: artifacts.deletingLastPathComponent()
                    .appendingPathComponent("escape.png").path),
            "a traversal label must not create a file outside the artifact directory")
        XCTAssertEqual(try filesInDirectory(artifacts).count, 1)
    }

    // MARK: Result hygiene across calls

    func testImagesDoNotLeakAcrossCalls() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        let first = await worker.evaluate(
            code: "repl.emitImage(\"\(pngBase64())\", \"first\")")
        let second = await worker.evaluate(code: "console.log('no images here')")

        XCTAssertEqual(first.status, .completed, first.errorText ?? "")
        XCTAssertEqual(first.images.count, 1)
        XCTAssertEqual(second.status, .completed, second.errorText ?? "")
        XCTAssertTrue(
            second.images.isEmpty, "each call's result carries only its own images")
    }

    // MARK: The frozen surface follows the 2.1 conventions

    func testEmitImageSurfaceIsFrozenOnTheSealedReplObject() {
        let harness = ImageEmitterHarness()

        let probes: [(String, String)] = [
            ("repl exposes exactly emitImage when only the image piece is installed",
             "Object.keys(repl).join(',') === 'emitImage'"),
            ("repl object is frozen", "Object.isFrozen(repl)"),
            ("repl object is not extensible", "!Object.isExtensible(repl)"),
            (
                "emitImage is a frozen function with a non-writable non-configurable "
                    + "descriptor",
                """
                (() => {
                    const descriptor = Object.getOwnPropertyDescriptor(repl, 'emitImage');
                    return descriptor !== undefined
                        && typeof descriptor.value === 'function'
                        && descriptor.writable === false
                        && descriptor.configurable === false
                        && Object.isFrozen(descriptor.value);
                })()
                """),
            (
                "the global repl binding is non-writable and non-configurable",
                """
                (() => {
                    const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'repl');
                    return descriptor !== undefined
                        && descriptor.writable === false
                        && descriptor.configurable === false;
                })()
                """),
            (
                "the installation bridge is not left on the global",
                """
                !Object.prototype.hasOwnProperty.call(globalThis, '__turbosparkEmitImage')
                """
            )
        ]

        for (description, script) in probes {
            let passed = harness.evaluate(script)?.toBool() ?? false
            XCTAssertTrue(passed, "expected true: \(description)")
        }
        XCTAssertTrue(harness.exceptions.isEmpty, "probes must not throw: \(harness.exceptions)")
    }

    func testTamperingWithTheEmitImageSurfaceThrowsAndTheOriginalSurvives() {
        let harness = ImageEmitterHarness()

        harness.evaluate(#""use strict"; repl.emitImage = () => {};"#)
        harness.evaluate(#""use strict"; globalThis.repl = {};"#)
        harness.evaluate("delete repl.emitImage;")

        XCTAssertEqual(
            harness.exceptions.count, 2,
            "strict reassignment must throw (delete fails silently), got: \(harness.exceptions)")

        harness.evaluate("""
        globalThis.emission = "__pending__";
        try {
            const path = repl.emitImage("\(pngBase64())", "after-tampering");
            globalThis.emission = "ok:" + (path.indexOf("/") === 0);
        } catch (error) {
            globalThis.emission = "err:" + error.message;
        }
        """)
        XCTAssertEqual(
            harness.awaitGlobal("emission"),
            "ok:true",
            "the frozen original must still emit after tampering attempts")
        XCTAssertEqual(harness.images.count, 1)
    }

    func testCaughtRejectionSurfacesTheExplanatoryMessageAsAnOrdinaryError() {
        let harness = ImageEmitterHarness()

        harness.evaluate("""
        globalThis.message = "__pending__";
        try {
            repl.emitImage("!!!not base64!!!", "broken");
            globalThis.message = "unexpectedly accepted";
        } catch (error) {
            globalThis.message = error.message;
        }
        """)

        XCTAssertEqual(
            harness.awaitGlobal("message"),
            "repl.emitImage rejected the image: the data is not valid base64.",
            "scripts must be able to catch the rejection and read its explanation")
        XCTAssertTrue(harness.images.isEmpty)
    }

    // MARK: The worker installs the complete sealed surface

    func testWorkerSurfaceExposesFrozenFsAndEmitImageTogether() async throws {
        let artifacts = try makeArtifactDirectory()
        let worker = makeWorker(artifacts: artifacts)

        let result = await worker.evaluate(code: """
        Object.keys(repl).sort().join(',')
            + '|' + Object.isFrozen(repl)
            + '|' + Object.isFrozen(repl.fs)
            + '|' + Object.isFrozen(repl.emitImage)
        """)

        XCTAssertEqual(result.status, .completed, result.errorText ?? "")
        XCTAssertEqual(
            result.completionText,
            "emitImage,fs|true|true|true",
            "the worker's sealed repl object carries both capabilities, frozen")
    }
}

/// A bare JavaScriptCore context with only the image emitter installed, for
/// surface and rejection probes that do not need the worker evaluation path.
private final class ImageEmitterHarness: @unchecked Sendable {
    let context: JSContext
    private let lock = NSLock()
    private var imageStorage: [REPLEmittedImage] = []
    private var exceptionStorage: [String] = []

    init(
        artifactDirectory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("repl-image-harness-\(UUID().uuidString)", isDirectory: true)
    ) {
        guard let context = JSContext() else {
            fatalError("JavaScriptCore could not create an image emitter test context")
        }
        self.context = context
        let harness = self
        context.exceptionHandler = { _, exception in
            harness.recordException(exception?.toString() ?? "unknown exception")
        }
        REPLHostFacade(output: { _ in })
            .installImageEmitter(
                into: context,
                config: REPLSessionConfiguration(artifactDirectory: artifactDirectory),
                limits: REPLLimits()) { image in
                harness.recordImage(image)
            }
    }

    @discardableResult
    func evaluate(_ script: String) -> JSValue? {
        context.evaluateScript(script)
    }

    /// Polls a global string until it leaves the pending sentinel, the same
    /// way a later REPL call observes promise settlement.
    func awaitGlobal(_ name: String) -> String? {
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if let value = context.evaluateScript("globalThis.\(name)")?.toString(),
                value != "__pending__" {
                return value
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
        return nil
    }

    var images: [REPLEmittedImage] {
        lock.lock()
        defer { lock.unlock() }
        return imageStorage
    }

    var exceptions: [String] {
        lock.lock()
        defer { lock.unlock() }
        return exceptionStorage
    }

    private func recordImage(_ image: REPLEmittedImage) {
        lock.lock()
        imageStorage.append(image)
        lock.unlock()
    }

    private func recordException(_ message: String) {
        lock.lock()
        exceptionStorage.append(message)
        lock.unlock()
    }
}
