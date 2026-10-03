// swift-tools-version: 5.9
import PackageDescription
import Foundation

// Development uses the sibling checkout. Release packaging sets one exact
// revision for both this Swift client and its bundled daemon.
let openKindRevision = ProcessInfo.processInfo.environment["OPENKIND_RELEASE_REVISION"]
let openKindURL = ProcessInfo.processInfo.environment["OPENKIND_PACKAGE_URL"]
    ?? "https://github.com/whit3rabbit/opendecision.git"
let openKindIdentity = URL(string: openKindURL)?.deletingPathExtension().lastPathComponent.lowercased()
    ?? "opendecision"
let openKindDependency: Package.Dependency = openKindRevision.map {
    .package(url: openKindURL, revision: $0)
} ?? .package(path: "../../../openkind")

let package = Package(
    name: "TurboSparkApp",
    defaultLocalization: "en",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../TurboSpark"),
        openKindDependency,
        .package(path: "Vendor/ZImage"),
        .package(path: "Vendor/QwenImage"),
        .package(url: "https://github.com/sqlcipher/SQLCipher.swift.git", from: "4.10.0"),
        .package(url: "https://github.com/gonzalezreal/swift-markdown-ui", from: "2.4.0"),
        .package(url: "https://github.com/whit3rabbit/syntext", exact: "2.5.0"),
        .package(url: "https://github.com/sparkle-project/Sparkle.git", from: "2.9.6")
    ],
    targets: [
        .executableTarget(
            name: "TurboSparkApp",
            dependencies: [
                .product(name: "TurboSpark", package: "TurboSpark"),
                .product(name: "OpenKind", package: openKindRevision == nil ? "openkind" : openKindIdentity),
                .product(name: "ZImage", package: "zimage"),
                .product(name: "QwenImage", package: "qwenimage"),
                .product(name: "SQLCipher", package: "SQLCipher.swift"),
                .product(name: "MarkdownUI", package: "swift-markdown-ui"),
                .product(name: "Syntext", package: "syntext"),
                .product(name: "Sparkle", package: "Sparkle")
            ],
            resources: [
                .process("Resources")
            ],
            cSettings: [
                .define("SQLITE_HAS_CODEC")
            ],
            linkerSettings: [
                .linkedFramework("JavaScriptCore"),
                // EVERY CONSUMER OF TurboSpark HAS TO REPEAT THIS, and that
                // is a SwiftPM limitation rather than a mistake here: a
                // library search path in `unsafeFlags` is resolved against
                // the root of the package being BUILT, not the package that
                // declared it. So TurboSpark's own `-LSources/CTurboSpark`
                // is correct when its tests link and wrong for everyone
                // else, and the path below is this package's view of the
                // same directory.
                .unsafeFlags([
                    "-L../TurboSpark/Sources/CTurboSpark",
                    "-Xlinker", "-force_load",
                    "-Xlinker", "../TurboSpark/Sources/CTurboSpark/libturbospark_ffi.a",
                ])
            ]
        ),
        .executableTarget(
            name: "ZImageMLXBenchmark",
            dependencies: [.product(name: "ZImage", package: "zimage")],
            path: "Benchmarks/ZImageMLXBenchmark"
        ),
        .executableTarget(
            name: "QwenImageMLXBenchmark",
            dependencies: [.product(name: "QwenImage", package: "qwenimage")],
            path: "Benchmarks/QwenImageMLXBenchmark"
        ),
        .target(
            name: "DOMSnapshotFixtures",
            path: "Tests/TurboSparkAppTests/Fixtures/DOMSnapshotService",
            resources: [
                .process("forms.html"),
                .process("dialogs.html"),
                .process("dynamic-rerender.html"),
                .process("credential-fields.html")
            ]
        ),
        .testTarget(
            name: "TurboSparkAppTests",
            dependencies: [
                "TurboSparkApp",
                "DOMSnapshotFixtures",
                .product(name: "TurboSpark", package: "TurboSpark"),
                .product(name: "SQLCipher", package: "SQLCipher.swift"),
                .product(name: "Syntext", package: "syntext")
            ],
            exclude: [
                "Fixtures/DOMSnapshotService"
            ],
            cSettings: [
                .define("SQLITE_HAS_CODEC")
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-L../TurboSpark/Sources/CTurboSpark",
                    "-Xlinker", "-force_load",
                    "-Xlinker", "../TurboSpark/Sources/CTurboSpark/libturbospark_ffi.a",
                ])
            ]
        )
    ]
)
