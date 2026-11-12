import AppKit
import Foundation
import SwiftUI
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.1 and 6.2: the libraries section builds, renders without crashing at
/// a narrow and a wide settings width, sits above the storage section, is in
/// the settings search catalog, and never writes the app error.
@MainActor
final class ModelLibrariesSectionViewTests: LibraryStateTestCase {
    /// `swift/TurboSparkApp`.
    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var sourcesRoot: URL { packageRoot.appendingPathComponent("Sources/TurboSparkApp") }

    private func source(_ relative: String) throws -> String {
        try String(contentsOf: sourcesRoot.appendingPathComponent(relative), encoding: .utf8)
    }

    // MARK: - Rendering

    private func populatedState() async -> LibraryState {
        let offline = ModelLibrary(
            id: "away-id", name: "Archive Disk With A Fairly Long Library Name", root: nil, available: false,
            modelCount: 7, lastSeen: "2026-10-01")
        let slow = ModelLibrary(
            id: "slow-id", name: "Old USB", root: "/Volumes/Old USB/TurboSpark", available: true, modelCount: 1)
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.t7, slow, offline]
        }
        let state = makeState()
        await state.refresh()
        state.recordMeasuredSpeed(412, for: t7.id)
        state.setPendingMovedStore(PendingMovedStore(root: "/Volumes/Old/Previous Store", name: "Previous Store"))
        return state
    }

    private func renderedHost(_ model: AppModel, width: CGFloat, dark: Bool = false) -> NSHostingView<AnyView> {
        let content = ModelLibrariesSectionView(model: model)
            .frame(width: width)
            .environment(\.colorScheme, dark ? .dark : .light)
        let host = NSHostingView(rootView: AnyView(content))
        host.frame = NSRect(x: 0, y: 0, width: width, height: 10)
        let window = NSWindow(contentRect: host.frame, styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        return host
    }

    func testTheSectionRendersAtNarrowAndWideWidths() async {
        let state = await populatedState()
        let model = await makeModel(using: state)
        for width in [420.0, 760.0] {
            let host = renderedHost(model, width: width)
            let size = host.fittingSize
            XCTAssertEqual(size.width, width, accuracy: 1, "width \(width)")
            // Five rows (This Mac, two connected, one away, the placeholder)
            // cannot fit in less than this without clipping.
            XCTAssertGreaterThan(size.height, 400, "width \(width): \(size.height)")
            host.frame = NSRect(origin: .zero, size: size)
            host.layoutSubtreeIfNeeded()
            guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
                return XCTFail("no bitmap at \(width)")
            }
            host.cacheDisplay(in: host.bounds, to: bitmap)
            XCTAssertTrue(hasMoreThanOneColor(bitmap), "the section drew nothing at \(width)")
        }
    }

    func testTheSectionRendersInDarkAndWithNoLibrariesLoadedYet() async {
        let loaded = await populatedState()
        let model = await makeModel(using: loaded)
        let dark = renderedHost(model, width: 600, dark: true)
        XCTAssertGreaterThan(dark.fittingSize.height, 300)

        let empty = makeState()
        let emptyModel = await makeModel(using: empty)
        let host = renderedHost(emptyModel, width: 600)
        XCTAssertGreaterThan(host.fittingSize.height, 40, "the heading and Add Library still draw")
    }

    func testTheSectionGrowsWhenTheTextSizeIsExtraLarge() async {
        let state = await populatedState()
        let model = await makeModel(using: state)
        let normal = renderedHost(model, width: 420).fittingSize.height
        let content = ModelLibrariesSectionView(model: model)
            .frame(width: 420)
            .environment(\.dynamicTypeSize, .accessibility3)
        let host = NSHostingView(rootView: AnyView(content))
        host.frame = NSRect(x: 0, y: 0, width: 420, height: 10)
        let window = NSWindow(contentRect: host.frame, styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        XCTAssertGreaterThanOrEqual(host.fittingSize.height, normal)
        XCTAssertEqual(host.fittingSize.width, 420, accuracy: 1, "large text must not widen the section")
    }

    private func hasMoreThanOneColor(_ bitmap: NSBitmapImageRep) -> Bool {
        guard let data = bitmap.bitmapData else { return false }
        let count = bitmap.bytesPerRow * bitmap.pixelsHigh
        guard count > 8 else { return false }
        let first = data[0]
        for index in 0..<min(count, 400_000) where data[index] != first { return true }
        return false
    }

    // MARK: - Where it sits

    func testThePaneListsLibrariesAboveTheStorageSection() throws {
        let pane = try source("Components/ModelsSettingsPaneView.swift")
        let body = try XCTUnwrap(pane.range(of: "public var body"))
        let rest = pane[body.upperBound...]
        let libraries = try XCTUnwrap(rest.range(of: "ModelLibrariesSectionView(model: model)"))
        let storage = try XCTUnwrap(rest.range(of: "turboSparkStorageSection"))
        XCTAssertLessThan(libraries.lowerBound, storage.lowerBound)
        XCTAssertFalse(pane.contains("Move Store"))
    }

    // MARK: - The settings search catalog

    func testTheNewControlsAreInTheSettingsSearchCatalog() {
        let titles = SettingsControlCatalog.entries.filter { $0.pane == .models }.map(\.title)
        XCTAssertTrue(titles.contains("Model libraries"), "\(titles)")
        XCTAssertTrue(titles.contains("Add Library\u{2026}"), "\(titles)")
    }

    // MARK: - It never raises an app error

    func testTheLibraryUIFilesNeverWriteTheAppErrorOrPostAToast() throws {
        let files = [
            "State/ModelLibrariesController.swift", "State/ModelLibraryRowModel.swift",
            "State/LibraryActionText.swift", "Components/ModelLibrariesSectionView.swift",
            "Components/ModelLibraryRowView.swift",
        ]
        for file in files {
            let code = MoveStoreRetirementTests.codeWithoutComments(try source(file))
            XCTAssertFalse(code.contains("model.error"), "\(file) writes or reads the app error")
            XCTAssertFalse(code.contains(".error = "), "\(file) assigns an error")
            XCTAssertFalse(code.contains("showToast"), "\(file) posts a toast")
            XCTAssertFalse(code.contains("activeToast"), "\(file) touches the toast")
        }
    }

    func testTheLibraryUIFilesNeverUseAppKitColorsOrFixedSizes() throws {
        let files = [
            "Components/ModelLibrariesSectionView.swift", "Components/ModelLibraryRowView.swift",
        ]
        for file in files {
            let code = MoveStoreRetirementTests.codeWithoutComments(try source(file))
            XCTAssertFalse(code.contains("nsColor"), "\(file) uses a system chrome color")
            XCTAssertFalse(code.contains(".font(.system"), "\(file) sizes text by number")
            XCTAssertFalse(code.contains(".font(.caption"), "\(file) sizes text by role without the theme")
            XCTAssertFalse(code.contains("withAnimation"), "\(file) animates without Reduce Motion")
        }
    }
}
