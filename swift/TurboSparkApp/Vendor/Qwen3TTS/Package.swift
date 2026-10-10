// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Qwen3TTSTurboSpark",
    platforms: [.macOS(.v14)],
    products: [.library(name: "Qwen3TTS", targets: ["Qwen3TTS"]), .executable(name: "Qwen3TTSSmoke", targets: ["Qwen3TTSSmoke"])],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.6")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "2.30.6"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.9"))
    ],
    targets: [
        .target(name: "Qwen3TTSCore", dependencies: [.product(name: "MLX", package: "mlx-swift")]),
        .target(name: "Qwen3TTSCodecs", dependencies: ["Qwen3TTSCore",
            .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXLMCommon", package: "mlx-swift-lm")]),
        .target(name: "Qwen3TTS", dependencies: ["Qwen3TTSCore", "Qwen3TTSCodecs",
            .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            .product(name: "Tokenizers", package: "swift-transformers")]),
        .executableTarget(name: "Qwen3TTSSmoke", dependencies: ["Qwen3TTS", .product(name: "MLX", package: "mlx-swift")]),
        .testTarget(name: "Qwen3TTSTests", dependencies: ["Qwen3TTS"], resources: [.process("Fixtures")])
    ]
)
