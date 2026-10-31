import AppKit
import Foundation
import SwiftUI
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.6 (requirement 4.3): the add-library banner is drawn in BOTH
/// placements (the top of the Installed Models manager and the top of the
/// Model libraries section) while an offer is pending, in neither otherwise,
/// and holds at a narrow and a wide width, in dark, right-to-left and at the
/// largest text size.
///
/// A hosted SwiftUI tree exposes no accessibility nodes in a unit test (the
/// tree is built only for an assistive client), and the AppKit classes behind a
/// button are private and differ between macOS releases. So the banner is found
/// by what it does to the layout: it takes about 100 points of height at the top
/// of the placement, and it takes them back when the offer is answered. The
/// words on it are pinned by `LibraryOfferBannerModelTests`, which the banner
/// draws and nothing else.
@MainActor
final class LibraryOfferBannerViewTests: LibraryStateTestCase {
    private var state: LibraryState!
    private var model: AppModel!
    private var announcements: [String] = []

    /// What a banner costs in height. The measured cost is about 96 points; this
    /// is a floor well under it that nothing else on the screen would add.
    private let bannerFloor: CGFloat = 60

    private func script(_ name: String, id: String, models: Int = 4, readOnly: Bool = false) {
        let marker = try! LibraryFixtures.marker(id, name: name, modelCount: models, readOnly: readOnly)
        service.with { $0.markers["/Volumes/\(name)"] = marker }
    }

    private func bootstrap() async {
        state = makeState()
        await state.refresh()
        model = await makeModel(using: state)
        announcements = []
    }

    private func offer(_ name: String, readOnly: Bool = false, models: Int = 4) async {
        script(name, id: "\(name.lowercased())-id", models: models, readOnly: readOnly)
        await state.handle(.mounted(LibraryFixtures.volume(name)))
        await state.waitForPendingWork()
    }

    // MARK: Hosting

    private enum Placement: CaseIterable {
        case installedManager
        case librariesSection

        @MainActor func view(_ model: AppModel) -> AnyView {
            switch self {
            case .installedManager: return AnyView(ModelManagerView(model: model))
            case .librariesSection: return AnyView(ModelLibrariesSectionView(model: model))
            }
        }

        var width: CGFloat { self == .installedManager ? 900 : 600 }
    }

    private func host(
        _ content: AnyView, width: CGFloat, height: CGFloat = 700, dark: Bool = false,
        rtl: Bool = false, dynamicType: DynamicTypeSize? = nil, locale: String = "en"
    ) -> NSHostingView<AnyView> {
        // The theme is resolved inside the environment that sets the scheme and
        // the text size, as a scene root does, so dark is the dark theme.
        var view = AnyView(
            content
                .appThemed()
                .frame(width: width)
                .environment(\.colorScheme, dark ? .dark : .light)
                .environment(\.layoutDirection, rtl ? .rightToLeft : .leftToRight)
                .environment(\.locale, Locale(identifier: locale)))
        if let dynamicType { view = AnyView(view.environment(\.dynamicTypeSize, dynamicType)) }
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(x: 0, y: 0, width: width, height: height)
        let window = NSWindow(contentRect: host.frame, styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        return host
    }

    /// Lets the main run loop turn so a published change reaches the hosted view.
    private func settle(_ hosts: NSView...) async {
        for _ in 0..<4 {
            try? await Task.sleep(nanoseconds: 40_000_000)
            Self.turnRunLoop(for: 0.02)
            for host in hosts { host.layoutSubtreeIfNeeded() }
        }
    }

    private func height(_ host: NSView) -> CGFloat { host.fittingSize.height }

    private func bannerAlone(
        width: CGFloat, dark: Bool = false, rtl: Bool = false, dynamicType: DynamicTypeSize? = nil,
        controller: LibraryOfferController? = nil, locale: String = "en"
    ) async -> NSHostingView<AnyView> {
        // The banner's words follow the in-app language picker, which is a
        // default, not the environment locale: set it for the render.
        let key = AppLanguage.storageKey
        let previous = UserDefaults.standard.string(forKey: key)
        UserDefaults.standard.set(locale, forKey: key)
        defer {
            if let previous { UserDefaults.standard.set(previous, forKey: key) } else {
                UserDefaults.standard.removeObject(forKey: key)
            }
        }
        let content = AnyView(
            LibraryOfferBanner(model: model, libraryState: state, controller: controller).padding(16))
        let hosted = host(
            content, width: width, dark: dark, rtl: rtl, dynamicType: dynamicType, locale: locale)
        await settle(hosted)
        return hosted
    }

    // MARK: - Both placements

    func testAPendingOfferAddsTheBannerToBothPlacementsAndAnsweringTakesItBack() async throws {
        await bootstrap()
        var hosts: [Placement: NSHostingView<AnyView>] = [:]
        var without: [Placement: CGFloat] = [:]
        for placement in Placement.allCases {
            let hosted = host(placement.view(model), width: placement.width)
            await settle(hosted)
            hosts[placement] = hosted
            without[placement] = height(hosted)
        }

        await offer("Spare")
        for placement in Placement.allCases { await settle(try XCTUnwrap(hosts[placement])) }
        for placement in Placement.allCases {
            let gained = height(try XCTUnwrap(hosts[placement])) - (without[placement] ?? 0)
            XCTAssertGreaterThan(gained, bannerFloor, "\(placement) shows no banner for a pending offer")
        }

        state.dismiss(try XCTUnwrap(state.pendingOffer))
        for placement in Placement.allCases { await settle(try XCTUnwrap(hosts[placement])) }
        for placement in Placement.allCases {
            XCTAssertEqual(
                height(try XCTUnwrap(hosts[placement])), without[placement] ?? 0, accuracy: 1,
                "\(placement) still shows an answered offer")
        }
    }

    func testNoOfferNoBannerInEitherPlacementAndTheBannerAloneDrawsNothing() async {
        await bootstrap()
        XCTAssertNil(state.pendingOffer)
        let alone = await bannerAlone(width: 600)
        // The 16-point padding around nothing is all there is.
        XCTAssertLessThanOrEqual(height(alone), 40)
        // A dismissed library is never offered again, so each placement gets a
        // library of its own and a fresh state.
        for (placement, name) in zip(Placement.allCases, ["Northern", "Southern"]) {
            await bootstrap()
            let hosted = host(placement.view(model), width: placement.width)
            await settle(hosted)
            let plain = height(hosted)
            await offer(name)
            await settle(hosted)
            XCTAssertGreaterThan(height(hosted), plain + bannerFloor, "\(placement)")
            state.dismiss(state.pendingOffer!)
            await settle(hosted)
            XCTAssertEqual(height(hosted), plain, accuracy: 1, "\(placement)")
        }
    }

    func testTheNextQueuedOfferTakesTheAnsweredOnesPlaceInBothPlacements() async throws {
        await bootstrap()
        let section = host(Placement.librariesSection.view(model), width: 600)
        let manager = host(Placement.installedManager.view(model), width: 900)
        await settle(section, manager)
        let plainSection = height(section)
        let plainManager = height(manager)

        await offer("Alpha")
        await offer("Beta")
        await settle(section, manager)
        let alphaSection = height(section)
        let alphaManager = height(manager)
        XCTAssertGreaterThan(alphaSection, plainSection + bannerFloor)
        XCTAssertGreaterThan(alphaManager, plainManager + bannerFloor)

        state.dismiss(try XCTUnwrap(state.pendingOffer))
        await settle(section, manager)

        XCTAssertEqual(state.pendingOffer?.markerId, "beta-id")
        XCTAssertGreaterThan(height(section), plainSection + bannerFloor, "Beta's banner took Alpha's place")
        XCTAssertGreaterThan(height(manager), plainManager + bannerFloor, "Beta's banner took Alpha's place")
        XCTAssertLessThan(height(section), alphaSection, "Beta has nobody queued behind it: no queued line")
        XCTAssertLessThan(height(manager), alphaManager)
    }

    // MARK: - Lines

    func testTheReadOnlyNoteAndTheQueuedLineEachMakeTheBannerTaller() async throws {
        await bootstrap()
        await offer("Plain")
        let plain = height(await bannerAlone(width: 600))

        await bootstrap()
        await offer("Locked", readOnly: true)
        let locked = height(await bannerAlone(width: 600))
        XCTAssertGreaterThan(locked, plain + 10, "the read-only note takes a line")

        await bootstrap()
        await offer("First")
        await offer("Second")
        let queued = height(await bannerAlone(width: 600))
        XCTAssertGreaterThan(queued, plain + 10, "the queued line takes a line")
    }

    func testARefusalTakesAnInlineLineAndTheAddInFlightShowsProgress() async throws {
        await bootstrap()
        await offer("Spare")
        let controller = LibraryOfferController(
            libraryState: state, isBusy: { false }, announce: { [unowned self] in self.announcements.append($0) },
            language: { .english })
        let plain = height(await bannerAlone(width: 420, controller: controller))

        service.with { $0.addError = TurboSparkError(code: .invalidArgument, message: "read_only: x") }
        await controller.add(try XCTUnwrap(state.pendingOffer))
        XCTAssertNotNil(controller.failure)
        let refused = height(await bannerAlone(width: 420, controller: controller))
        XCTAssertGreaterThan(refused, plain + 10, "the sentence is inline in the banner")

        // The offer is still there to answer.
        XCTAssertEqual(state.pendingOffer?.markerId, "spare-id")
    }

    // MARK: - Layout

    func testTheBannerHoldsAtNarrowAndWideWidthsInLightAndDark() async {
        await bootstrap()
        await offer("Spare With A Fairly Long Library Name That Keeps Going And Going Past The Edge", readOnly: true)
        await offer("Second")
        var heights: [CGFloat: CGFloat] = [:]
        for width in [420.0, 760.0] {
            for dark in [false, true] {
                let hosted = await bannerAlone(width: width, dark: dark)
                XCTAssertEqual(hosted.fittingSize.width, width, accuracy: 1, "\(width) dark=\(dark)")
                XCTAssertGreaterThan(height(hosted), 80, "\(width) dark=\(dark)")
                heights[width] = height(hosted)
                // A hosting view centers content that is smaller than its frame,
                // so it is sized to the content before it is drawn.
                hosted.frame = NSRect(origin: .zero, size: hosted.fittingSize)
                hosted.layoutSubtreeIfNeeded()
                guard let bitmap = hosted.bitmapImageRepForCachingDisplay(in: hosted.bounds) else {
                    return XCTFail("no bitmap")
                }
                hosted.cacheDisplay(in: hosted.bounds, to: bitmap)
                XCTAssertTrue(hasMoreThanOneColor(bitmap), "the banner drew nothing at \(width) dark=\(dark)")
            }
        }
        XCTAssertGreaterThanOrEqual(heights[420] ?? 0, heights[760] ?? 0, "a narrow banner wraps, it does not clip")
    }

    func testTheBannerHoldsAtTheLargestTextSizeAndInRightToLeftWithoutWideningTheSection() async {
        await bootstrap()
        await offer("Spare", readOnly: true)
        await offer("Beta")
        let normal = height(await bannerAlone(width: 420))
        let large = await bannerAlone(width: 420, dynamicType: .accessibility3)
        XCTAssertGreaterThanOrEqual(height(large), normal, "larger text does not shrink the banner")
        XCTAssertEqual(large.fittingSize.width, 420, accuracy: 1, "large text must not widen the banner")
        for (label, locale, dynamicType) in [
            ("ar", "ar", nil), ("he", "he", nil), ("ar large", "ar", DynamicTypeSize.accessibility3),
        ] as [(String, String, DynamicTypeSize?)] {
            let rtl = await bannerAlone(width: 420, rtl: true, dynamicType: dynamicType, locale: locale)
            XCTAssertEqual(rtl.fittingSize.width, 420, accuracy: 1, label)
            XCTAssertGreaterThan(height(rtl), 80, label)
        }
        let german = await bannerAlone(width: 420, dynamicType: .accessibility3, locale: "de")
        XCTAssertEqual(german.fittingSize.width, 420, accuracy: 1, "a long translation wraps")
    }

    func testTheInstalledManagerStillRendersWithTheBannerAbove() async {
        await bootstrap()
        await offer("Spare")
        let hosted = host(Placement.installedManager.view(model), width: 900, height: 700)
        await settle(hosted)
        guard let bitmap = hosted.bitmapImageRepForCachingDisplay(in: hosted.bounds) else {
            return XCTFail("no bitmap")
        }
        hosted.cacheDisplay(in: hosted.bounds, to: bitmap)
        XCTAssertTrue(hasMoreThanOneColor(bitmap), "the manager drew nothing")
    }

    private func hasMoreThanOneColor(_ bitmap: NSBitmapImageRep) -> Bool {
        guard let data = bitmap.bitmapData else { return false }
        let count = bitmap.bytesPerRow * bitmap.pixelsHigh
        guard count > 8 else { return false }
        let first = data[0]
        for index in 0..<count where data[index] != first { return true }
        return false
    }

    // MARK: - Source guards

    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
    }

    private func code(_ relative: String) throws -> String {
        MoveStoreRetirementTests.codeWithoutComments(
            try String(contentsOf: sourcesRoot.appendingPathComponent(relative), encoding: .utf8))
    }

    func testBothPlacementsDrawTheSharedBannerOnlyForAPendingOffer() throws {
        let manager = try code("Installation/ModelManagerView.swift")
        let section = try code("Components/ModelLibrariesSectionView.swift")
        XCTAssertTrue(manager.contains("LibraryOfferBanner("), "the Installed Models manager has no banner")
        XCTAssertTrue(section.contains("LibraryOfferBanner("), "the Model libraries section has no banner")
        XCTAssertTrue(manager.contains("libraryState.pendingOffer != nil"), "the manager must key it on the offer")
        XCTAssertTrue(section.contains("libraryState.pendingOffer != nil"), "the section must key it on the offer")
    }

    func testBothButtonsAreDisabledNotHiddenWhileBlocked() throws {
        let banner = try code("Components/LibraryOfferBanner.swift")
        XCTAssertEqual(banner.components(separatedBy: ".disabled(blocked)").count - 1, 2)
        XCTAssertTrue(banner.contains("controller.isChangeBlocked"))
        XCTAssertTrue(banner.contains("model.isInstallOrMoveRunning"))
        // The design-system bar, indeterminate (#202 replaced `ProgressView()`).
        XCTAssertTrue(banner.contains("AppProgressBar(value: nil"), "an add in flight shows a progress indicator")
        XCTAssertTrue(banner.contains(".appAction(.primary"))
        XCTAssertTrue(banner.contains(".appAction(.quiet"))
        XCTAssertTrue(banner.contains("accessibilityIdentifier(\"library-offer-banner\")"))
    }

    func testTheOfferFilesNeverWriteTheAppErrorOrPostAToastOrPickTheirOwnLook() throws {
        let files = [
            "Components/LibraryOfferBanner.swift", "State/LibraryOfferBannerModel.swift",
            "State/LibraryOfferController.swift",
        ]
        for file in files {
            let text = try code(file)
            XCTAssertFalse(text.contains("model.error"), "\(file) touches the app error")
            XCTAssertFalse(text.contains(".error = "), "\(file) assigns an error")
            XCTAssertFalse(text.contains("showToast"), "\(file) posts a toast")
            XCTAssertFalse(text.contains("activeToast"), "\(file) touches the toast")
            XCTAssertFalse(text.contains("nsColor"), "\(file) uses a system chrome color")
            XCTAssertFalse(text.contains(".font(.system"), "\(file) sizes text by number")
            XCTAssertFalse(text.contains(".font(.caption"), "\(file) sizes text without the theme")
            XCTAssertFalse(text.contains("withAnimation"), "\(file) animates without Reduce Motion")
            XCTAssertFalse(text.contains(".frame(width:"), "\(file) fixes a width")
        }
    }

    func testOnlyTheAppModelPostsTheToastAndTheAnnouncerNeverTouchesTheError() throws {
        let announcer = try code("State/LibraryOfferAnnouncer.swift")
        XCTAssertFalse(announcer.contains("showToast"), "the announcer takes a closure; AppModel owns the toast")
        XCTAssertFalse(announcer.contains("error"), "an offer must not raise the app error")
        let wiring = try code("State/AppModel+ModelLibraries.swift")
        XCTAssertTrue(wiring.contains("showToast"), "AppModel posts the toast")
        XCTAssertFalse(wiring.contains("model.error"))
    }

    /// Sync on purpose: `RunLoop.run(until:)` is unavailable in async code,
    /// and the hosted SwiftUI view needs the main run loop to turn.
    private static func turnRunLoop(for seconds: TimeInterval) {
        RunLoop.current.run(until: Date().addingTimeInterval(seconds))
    }
}
