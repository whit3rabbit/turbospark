// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ZImageTurboSpark",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "ZImage", targets: ["ZImage"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.6")),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.7.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.9")),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.6.4")
    ],
    targets: [
        .target(
            name: "ZImage",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Logging", package: "swift-log")
            ],
            path: "Sources/ZImage"
        )
    ]
)
