// swift-tools-version: 5.9
import PackageDescription
import Foundation

// Development uses the sibling checkout. Release packaging pins this Swift
// client to the source commit that published its paired daemon crate.
let openKindRevision = ProcessInfo.processInfo.environment["OPENKIND_RELEASE_REVISION"]
let openKindURL = ProcessInfo.processInfo.environment["OPENKIND_PACKAGE_URL"]
    ?? "https://github.com/whit3rabbit/openkind.git"
let openKindIdentity = URL(string: openKindURL)?.deletingPathExtension().lastPathComponent.lowercased()
    ?? "openkind"
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
        .package(path: "Vendor/Qwen3TTS"),
        .package(url: "https://github.com/sqlcipher/SQLCipher.swift.git", from: "4.10.0"),
        .package(url: "https://github.com/gonzalezreal/swift-markdown-ui", from: "2.4.0"),
        // Exact is safe from 2.7.1 on: the tag pins its own xcframework zip.
        // Tags through v2.7.0 pinned the PREVIOUS release's zip (`v2.5.0`
        // linked the 2.4.0 binary), so never pin one of those exactly; use
        // its `swift-vX.Y.Z` commit by revision. See docs/SYNTEXT.md section 6.
        .package(url: "https://github.com/whit3rabbit/syntext", exact: "2.7.1"),
        .package(url: "https://github.com/sparkle-project/Sparkle.git", from: "2.9.6"),
        // Exact on purpose. From v0.15.6 FluidAudio links a Rust staticlib
        // (NemoTextProcessing): a duplicate _rust_eh_personality and a fourth
        // compact-unwind personality, which this app's link cannot take.
        // Read docs/FLUIDAUDIO.md before changing the version.
        .package(url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.15.5")
    ],
    targets: [
        // The app's resources live in their own target so the lookup can be
        // hand-written (see AppResources.swift). The executable target must
        // declare none, or SwiftPM generates a `Bundle.module` that collides
        // with the forwarding one in Theme/ResourceBundle.swift.
        .target(
            name: "TurboSparkAppResources",
            resources: [
                .process("Resources")
            ]
        ),
        .executableTarget(
            name: "TurboSparkApp",
            dependencies: [
                "TurboSparkAppResources",
                .product(name: "TurboSpark", package: "TurboSpark"),
                .product(name: "OpenKind", package: openKindRevision == nil ? "openkind" : openKindIdentity),
                .product(name: "ZImage", package: "zimage"),
                .product(name: "QwenImage", package: "qwenimage"),
                .product(name: "Qwen3TTS", package: "qwen3tts"),
                .product(name: "SQLCipher", package: "SQLCipher.swift"),
                .product(name: "MarkdownUI", package: "swift-markdown-ui"),
                .product(name: "Syntext", package: "syntext"),
                .product(name: "SyntextContext", package: "syntext"),
                .product(name: "Sparkle", package: "Sparkle"),
                .product(name: "FluidAudio", package: "FluidAudio")
            ],
            exclude: [
                "Tools/Codemode/README.md"
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
                    "-Xlinker", "-rpath",
                    "-Xlinker", "@executable_path/../Frameworks",
                    // A bare SwiftPM executable has no Info.plist, so AppKit
                    // pinned the menu bar to English whatever the language
                    // picker wrote to AppleLanguages. The embedded copy
                    // declares the catalog's languages (see the file).
                    // Editing it needs a relink: SwiftPM does not track it.
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Localization/Info.plist",
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
                .product(name: "Syntext", package: "syntext"),
                .product(name: "SyntextContext", package: "syntext"),
                .product(name: "FluidAudio", package: "FluidAudio")
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
