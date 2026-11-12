import Foundation

/// Finds MLX's compiled Metal kernels before any MLX work starts.
///
/// When MLX cannot load its kernel library it calls mlx-c's default error
/// handler, which prints to stdout and calls `exit(-1)`. In a bundled app
/// that is a silent quit with no crash report. SwiftPM's native build
/// system does not compile mlx-swift's Metal shaders (the swiftbuild system
/// does), so a native build ships no `mlx-swift_Cmlx.bundle`. v0.6.0 shipped
/// that way and quit on the first Z-Image generation, right after "Loading
/// tokenizer". Checking first turns that into an error the Images screen
/// can show.
///
/// The search mirrors `load_default_library` in mlx-swift's
/// `mlx/backend/metal/device.cpp`, in its order.
enum MLXKernelLibrary {
    static let swiftPMBundleName = "mlx-swift_Cmlx"

    /// The first kernel library MLX would load, or nil when it would find none.
    static func locate(
        executableDirectory: URL?,
        mainBundleURL: URL,
        bundleResourceURLs: [URL],
        frameworkResourceURLs: [URL],
        currentDirectory: URL
    ) -> URL? {
        let fileManager = FileManager.default
        var candidates: [URL] = []
        if let executableDirectory {
            candidates.append(executableDirectory.appendingPathComponent("mlx.metallib"))
            candidates.append(executableDirectory.appendingPathComponent("Resources/mlx.metallib"))
        }
        // `try_load_bundle`: `<url>/mlx-swift_Cmlx.bundle`, read as a bundle.
        for url in [mainBundleURL] + bundleResourceURLs {
            let path = url.appendingPathComponent(swiftPMBundleName + ".bundle").path
            if let resources = Bundle(path: path)?.resourceURL {
                candidates.append(resources.appendingPathComponent("default.metallib"))
            }
        }
        for url in frameworkResourceURLs {
            candidates.append(url.appendingPathComponent("default.metallib"))
        }
        if let executableDirectory {
            candidates.append(executableDirectory.appendingPathComponent("Resources/default.metallib"))
        }
        // `METAL_PATH` is the bare "default.metallib", resolved against the
        // working directory.
        candidates.append(currentDirectory.appendingPathComponent("default.metallib"))
        return candidates.first { fileManager.isReadableFile(atPath: $0.path) }
    }

    /// `locate` for this process.
    static func locateForCurrentProcess() -> URL? {
        let frameworks = Bundle.allFrameworks
            .filter { $0.bundleIdentifier == swiftPMBundleName }
            .compactMap(\.resourceURL)
        return locate(
            executableDirectory: Bundle.main.executableURL?.deletingLastPathComponent(),
            mainBundleURL: Bundle.main.bundleURL,
            bundleResourceURLs: Bundle.allBundles.compactMap(\.resourceURL),
            frameworkResourceURLs: frameworks,
            currentDirectory: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        )
    }

    /// Throws `MLXKernelLibraryError.missing` when MLX would exit the process.
    static func requireForCurrentProcess() throws {
        if locateForCurrentProcess() == nil { throw MLXKernelLibraryError.missing }
    }
}

enum MLXKernelLibraryError: LocalizedError, Equatable {
    case missing

    var errorDescription: String? {
        """
        This build of TurboSpark is missing MLX's GPU kernels \
        (mlx-swift_Cmlx.bundle), so Swift MLX image models cannot run. \
        Reinstall TurboSpark from a release that includes them.
        """
    }
}
