// swift-tools-version: 6.0
// VoiceIsolatorMLX, vendored from starkdmi/VoiceIsolatorMLX (see UPSTREAM.md):
// the library target only, on the app's mlx-swift 0.30.x line.

import PackageDescription

let package = Package(
    name: "VoiceIsolatorMLX",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "VoiceIsolatorMLX", targets: ["VoiceIsolatorMLX"]),
    ],
    dependencies: [
        // Same line as Vendor/Qwen3TTS, so the app resolves one mlx-swift.
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.6")),
    ],
    targets: [
        .target(
            name: "VoiceIsolatorMLX",
            dependencies: [.product(name: "MLX", package: "mlx-swift")]
        ),
        .testTarget(
            name: "VoiceIsolatorMLXTests",
            dependencies: ["VoiceIsolatorMLX"]
        ),
    ]
)
