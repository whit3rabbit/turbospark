import Foundation
import XCTest

@testable import TurboSparkApp

/// v0.6.0 shipped no `mlx-swift_Cmlx.bundle`, so the first Z-Image generation
/// reached MLX's default error handler, which calls `exit(-1)`: the app quit
/// after "Loading tokenizer" with no crash report. The image sessions now
/// look for the kernels first, the way MLX does, and throw instead.
final class MLXKernelLibraryTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-kernel-library-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    /// The v0.6.0 layout: `Contents/MacOS` and `Contents/Resources` with no
    /// kernel library anywhere.
    private func appLayout() throws -> (app: URL, macOS: URL, resources: URL) {
        let app = root.appendingPathComponent("TurboSpark.app")
        let macOS = app.appendingPathComponent("Contents/MacOS")
        let resources = app.appendingPathComponent("Contents/Resources")
        for dir in [macOS, resources] {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return (app, macOS, resources)
    }

    private func touch(_ url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("metallib".utf8).write(to: url)
    }

    private func locate(_ layout: (app: URL, macOS: URL, resources: URL)) -> URL? {
        MLXKernelLibrary.locate(
            executableDirectory: layout.macOS,
            mainBundleURL: layout.app,
            bundleResourceURLs: [layout.resources],
            frameworkResourceURLs: [],
            currentDirectory: root.appendingPathComponent("cwd")
        )
    }

    func testABundleWithoutTheKernelsIsRefused() throws {
        let layout = try appLayout()
        XCTAssertNil(locate(layout))
        XCTAssertTrue(MLXKernelLibraryError.missing.localizedDescription.contains("mlx-swift_Cmlx.bundle"))
        // An empty Cmlx bundle is still no kernels.
        try FileManager.default.createDirectory(
            at: layout.resources.appendingPathComponent("mlx-swift_Cmlx.bundle/Contents/Resources"),
            withIntermediateDirectories: true)
        XCTAssertNil(locate(layout))
    }

    /// Where a swiftbuild release puts the kernels, copied into Resources by
    /// `make-app-bundle.sh`'s `*.bundle` glob.
    func testTheSwiftPMBundleInResourcesIsFound() throws {
        let layout = try appLayout()
        let metallib = layout.resources
            .appendingPathComponent("mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
        try touch(metallib)
        XCTAssertEqual(locate(layout)?.resolvingSymlinksInPath(), metallib.resolvingSymlinksInPath())
    }

    /// This test process links MLX too. A swiftbuild test build carries the
    /// kernels, so the check must pass here, or it would refuse working builds.
    func testThisTestProcessHasItsKernels() throws {
        XCTAssertNotNil(MLXKernelLibrary.locateForCurrentProcess())
        XCTAssertNoThrow(try MLXKernelLibrary.requireForCurrentProcess())
    }

    func testAColocatedLibraryIsFoundFirst() throws {
        let layout = try appLayout()
        try touch(layout.resources
            .appendingPathComponent("mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"))
        let colocated = layout.macOS.appendingPathComponent("mlx.metallib")
        try touch(colocated)
        XCTAssertEqual(locate(layout), colocated)
    }

    func testAFrameworkOrWorkingDirectoryLibraryIsFound() throws {
        let layout = try appLayout()
        let framework = root.appendingPathComponent("Cmlx.framework/Resources")
        try touch(framework.appendingPathComponent("default.metallib"))
        XCTAssertNotNil(MLXKernelLibrary.locate(
            executableDirectory: layout.macOS, mainBundleURL: layout.app,
            bundleResourceURLs: [layout.resources], frameworkResourceURLs: [framework],
            currentDirectory: root))
        let cwd = root.appendingPathComponent("cwd")
        try touch(cwd.appendingPathComponent("default.metallib"))
        XCTAssertEqual(locate(layout), cwd.appendingPathComponent("default.metallib"))
    }
}
