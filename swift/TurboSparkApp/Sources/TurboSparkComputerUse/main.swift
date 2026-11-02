import AppKit
import ComputerUseKit

// TurboSpark Computer Use: owns the Accessibility and Screen Recording grants
// for TurboSpark, and acts only for the app bundle it ships inside. The app
// launches it with `--session-dir <dir>`; `--self-test` prints the manifest
// for scripts/verify-app-bundle.sh and exits.

let arguments = CommandLine.arguments

if arguments.contains("--self-test") {
    let data = (try? JSONEncoder().encode(HelperService.manifest)) ?? Data()
    print(String(data: data, encoding: .utf8) ?? "")
    exit(0)
}

guard let flag = arguments.firstIndex(of: "--session-dir"), flag + 1 < arguments.count else {
    FileHandle.standardError.write(Data("usage: TurboSparkComputerUse --session-dir <dir>\n".utf8))
    exit(64)
}

let application = NSApplication.shared
application.setActivationPolicy(.accessory)
let controller = MainActor.assumeIsolated {
    HelperController(sessionDirectory: URL(fileURLWithPath: arguments[flag + 1]))
}
MainActor.assumeIsolated { controller.start() }
application.run()
