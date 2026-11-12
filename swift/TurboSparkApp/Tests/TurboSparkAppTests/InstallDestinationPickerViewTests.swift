import AppKit
import Foundation
import SwiftUI
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.9 (requirements 2.1, 2.2 and 2.5): the destination menu is drawn only
/// when there is a choice, drawn with the note when the default library is away,
/// and never drawn by the surfaces that install embedding or OCR models.
///
/// A hosted SwiftUI tree exposes no accessibility nodes in a unit test, and the
/// AppKit classes behind a menu differ between macOS releases, so the menu is
/// found by what it does to the layout: it takes height, and with nothing to
/// choose it takes none. The words and the rows are pinned by
/// `InstallDestinationModelTests`, which the menu draws and nothing else.
@MainActor
final class InstallDestinationPickerViewTests: LibraryStateTestCase {
    private var model: AppModel!
    private var state: LibraryState!

    /// A menu row is well over this tall; nothing else on the screen is here.
    private let menuFloor: CGFloat = 16

    private func bootstrap(
        libraries: [ModelLibrary], thisMacFree: UInt64? = 100_000_000_000
    ) async {
        service.with { $0.libraries = libraries }
        state = makeState()
        await state.refresh()
        model = await makeModel(using: state)
        model.thisMacFreeSpaceReader = { thisMacFree }
        addTeardownBlock { @MainActor [model] in model?.stopCronScheduler() }
    }

    private func host(
        width: CGFloat, dark: Bool = false, rtl: Bool = false, dynamicType: DynamicTypeSize? = nil,
        locale: String = "en"
    ) async -> NSHostingView<AnyView> {
        // The words follow the in-app language picker, which is a default, not the
        // environment locale: set it for the render.
        let key = AppLanguage.storageKey
        let previous = UserDefaults.standard.string(forKey: key)
        UserDefaults.standard.set(locale, forKey: key)
        defer {
            if let previous { UserDefaults.standard.set(previous, forKey: key) } else {
                UserDefaults.standard.removeObject(forKey: key)
            }
        }
        var view = AnyView(
            InstallDestinationPicker(model: model, selection: .constant(nil))
                .padding(16)
                .appThemed()
                .frame(width: width)
                .environment(\.colorScheme, dark ? .dark : .light)
                .environment(\.layoutDirection, rtl ? .rightToLeft : .leftToRight)
                .environment(\.locale, Locale(identifier: locale)))
        if let dynamicType { view = AnyView(view.environment(\.dynamicTypeSize, dynamicType)) }
        let hosted = NSHostingView(rootView: view)
        hosted.frame = NSRect(x: 0, y: 0, width: width, height: 400)
        let window = NSWindow(contentRect: hosted.frame, styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = hosted
        for _ in 0..<4 {
            try? await Task.sleep(nanoseconds: 40_000_000)
            Self.turnRunLoop(for: 0.02)
            hosted.layoutSubtreeIfNeeded()
        }
        return hosted
    }

    /// The menu's own height: the host's fitting height without the 16-point
    /// padding on each side.
    private func menuHeight(_ host: NSView) -> CGFloat { host.fittingSize.height - 32 }

    // MARK: Requirement 2.1

    func testNoMenuIsDrawnWhenThisMacIsTheOnlyDestination() async {
        await bootstrap(libraries: [LibraryFixtures.internalLibrary])
        let hosted = await host(width: 420)
        XCTAssertLessThan(menuHeight(hosted), menuFloor, "one destination is no choice, so no menu")
    }

    func testNoMenuIsDrawnForALibraryThatIsAway() async {
        await bootstrap(libraries: [
            LibraryFixtures.internalLibrary, LibraryFixtures.external("spare-id", name: "Spare", available: false),
        ])
        let hosted = await host(width: 420)
        XCTAssertLessThan(menuHeight(hosted), menuFloor)
    }

    func testTheMenuIsDrawnWithTwoDestinationsAtEveryWidth() async {
        await bootstrap(libraries: [LibraryFixtures.internalLibrary, t7])
        for width: CGFloat in [420, 760] {
            let hosted = await host(width: width)
            XCTAssertGreaterThan(menuHeight(hosted), menuFloor, "width \(width)")
        }
    }

    func testTheMenuHoldsInDarkRightToLeftAndTheLargestTextSize() async {
        await bootstrap(libraries: [LibraryFixtures.internalLibrary, t7])
        let plain = await host(width: 420)
        let dark = await host(width: 420, dark: true)
        let arabic = await host(width: 420, rtl: true, locale: "ar")
        let large = await host(width: 420, dynamicType: .accessibility5)
        for (name, hosted) in [("dark", dark), ("arabic", arabic), ("large", large)] {
            XCTAssertGreaterThan(menuHeight(hosted), menuFloor, name)
            XCTAssertLessThan(menuHeight(hosted), 400, "\(name) fits the pane instead of overflowing")
        }
        XCTAssertGreaterThan(
            menuHeight(large), menuHeight(plain) - 1, "the largest text is not smaller than the default")
    }

    // MARK: Requirement 2.2

    func testTheNoteIsDrawnWhenTheDefaultLibraryIsAway() async {
        // Exactly one library is the default: here it is the one that is away.
        let thisMac = ModelLibrary(
            id: "internal", name: "This Mac", root: "/Users/test/.turbospark/models", available: true,
            modelCount: 2, isDefault: false, isInternal: true)
        await bootstrap(libraries: [
            thisMac, LibraryFixtures.external("spare-id", name: "Spare", available: false, isDefault: true),
        ])
        let hosted = await host(width: 420)
        XCTAssertGreaterThan(
            menuHeight(hosted), menuFloor, "the install goes to This Mac, and the user is told why")
        XCTAssertNotNil(state.destinationDefault.notConnectedLibraryName)
    }

    // MARK: Requirement 2.5

    func testTheEmbeddingAndOcrSurfacesDoNotDrawTheMenu() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
        var embedding = 0
        let walker = FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil)
        while let url = walker?.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            let code = MoveStoreRetirementTests.codeWithoutComments(
                try String(contentsOf: url, encoding: .utf8))
            // Any file that queues an embedding or OCR download is home-only.
            if code.contains("enqueueEmbeddingDownload") || code.contains("EmbeddingCatalog") {
                embedding += 1
                XCTAssertFalse(
                    code.contains("InstallDestinationPicker"), "\(url.lastPathComponent) installs home-only models")
            }
        }
        XCTAssertGreaterThan(embedding, 0, "the scan found no embedding surface: is it looking at the right files?")
    }

    /// Sync on purpose: `RunLoop.run(until:)` is unavailable in async code,
    /// and the hosted SwiftUI view needs the main run loop to turn.
    private static func turnRunLoop(for seconds: TimeInterval) {
        RunLoop.current.run(until: Date().addingTimeInterval(seconds))
    }
}
