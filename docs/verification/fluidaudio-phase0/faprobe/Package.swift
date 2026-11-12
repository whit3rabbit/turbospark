// swift-tools-version: 5.9
import PackageDescription

// Phase 0 spike harness: FluidAudio 0.15.5 offline behavior and per-family
// loads from pinned install layouts. Same tools version and platform floor
// as swift/TurboSparkApp. Not part of the repository.
let package = Package(
    name: "FAProbe",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", exact: "0.15.5")
    ],
    targets: [
        .executableTarget(
            name: "FAProbe",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")]
        )
    ]
)
