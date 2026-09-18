// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "TurboSparkApp",
    defaultLocalization: "en",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../TurboSpark"),
        .package(url: "https://github.com/gonzalezreal/swift-markdown-ui", from: "2.4.0"),
        .package(url: "https://github.com/whit3rabbit/syntext", exact: "2.5.0")
    ],
    targets: [
        .executableTarget(
            name: "TurboSparkApp",
            dependencies: [
                .product(name: "TurboSpark", package: "TurboSpark"),
                .product(name: "MarkdownUI", package: "swift-markdown-ui"),
                .product(name: "Syntext", package: "syntext")
            ],
            resources: [
                .process("Resources")
            ],
            linkerSettings: [
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
        .testTarget(
            name: "TurboSparkAppTests",
            dependencies: [
                "TurboSparkApp",
                .product(name: "TurboSpark", package: "TurboSpark"),
                .product(name: "Syntext", package: "syntext")
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
