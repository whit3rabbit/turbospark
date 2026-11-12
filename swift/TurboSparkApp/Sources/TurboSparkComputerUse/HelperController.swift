import AppKit
import ComputerUseKit

/// The menu bar item is the user's sign that something can drive their Mac,
/// and its Stop item is the kill switch that needs no keyboard.
@MainActor
final class HelperController {
    private let service = HelperService()
    private let server: HelperServer
    private var statusItem: NSStatusItem?
    private var stopItem: NSMenuItem?

    init(sessionDirectory: URL) {
        server = HelperServer(sessionDirectory: sessionDirectory, service: service)
    }

    func start() {
        installStatusItem()
        service.onStoppedChange = { [weak self] stopped in self?.update(stopped: stopped) }
        service.installKillSwitch()
        server.start()
    }

    private func installStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        item.button?.image = NSImage(systemSymbolName: "cursorarrow.rays", accessibilityDescription: "TurboSpark Computer Use")
        let menu = NSMenu()
        let title = NSMenuItem(title: "TurboSpark can control granted apps", action: nil, keyEquivalent: "")
        title.isEnabled = false
        menu.addItem(title)
        menu.addItem(.separator())
        let stop = NSMenuItem(title: "Stop Agent Control", action: #selector(stopFromMenu), keyEquivalent: "")
        stop.target = self
        menu.addItem(stop)
        let quit = NSMenuItem(title: "Quit Computer Use Helper", action: #selector(quitFromMenu), keyEquivalent: "")
        quit.target = self
        menu.addItem(quit)
        item.menu = menu
        statusItem = item
        stopItem = stop
    }

    private func update(stopped: Bool) {
        stopItem?.title = stopped ? "Agent Control Stopped" : "Stop Agent Control"
        stopItem?.isEnabled = !stopped
        statusItem?.button?.image = NSImage(
            systemSymbolName: stopped ? "hand.raised.fill" : "cursorarrow.rays",
            accessibilityDescription: "TurboSpark Computer Use")
    }

    @objc private func stopFromMenu() { service.stop() }
    @objc private func quitFromMenu() { exit(0) }
}
