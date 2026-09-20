import AppKit
import SwiftUI
import XCTest
import TurboSpark
@testable import TurboSparkApp

@MainActor
final class ImageModelOnboardingTests: XCTestCase {
    private func model() throws -> AppModel {
        let model = AppModel()
        model.imageModels = []
        model.imageModelPathText = ""
        model.imageCatalog = try JSONDecoder().decode([ImageCatalogEntry].self, from: Data(
            #"[{"alias":"z-image-turbo-mlx-2bit","modelID":"fixture/Z-Image","revision":"pinned","quantization":"2-bit"}]"#.utf8))
        return model
    }

    func testSetupRequiresImageNavigationAndNoSelectedOrInstalledModel() throws {
        let model = try model()
        for section in AppModel.AppNavigationSection.allCases {
            model.activeSection = section
            XCTAssertEqual(model.shouldRecommendImageModel, section == .images)
        }
        model.activeSection = .images
        model.imageModelPathText = "/models/side-loaded.image.gturbo"
        XCTAssertFalse(model.shouldRecommendImageModel)
        model.imageModelPathText = ""
        model.imageModels = [ImageInstalledModel(
            alias: "z-image-turbo", modelID: "fixture/Z-Image", revision: "pinned",
            path: "/models/image", width: 1024, height: 1024, schedulerSteps: 9)]
        XCTAssertFalse(model.shouldRecommendImageModel)
        model.imageModels = []
        model.imageCatalog = []
        XCTAssertFalse(model.shouldRecommendImageModel)
    }

    func testStartupDoesNotConsumeSetupAndOpeningImagesDoes() async throws {
        let model = try model()
        model.activeSection = .files
        let suiteName = "ImageModelOnboardingTests-" + UUID().uuidString
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        let previousShutdown = AppShutdownCoordinator.shared.onTerminate
        defer {
            defaults.removePersistentDomain(forName: suiteName)
            AppShutdownCoordinator.shared.onTerminate = previousShutdown
        }
        let host = NSHostingView(rootView: RootView(model: model).defaultAppStorage(defaults))
        host.frame = NSRect(x: 0, y: 0, width: 1320, height: 900)
        host.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertFalse(defaults.bool(forKey: "TurboSpark.imageModelRecommendationSeen"))

        // Catalog loading can finish after the user opens the workspace.
        let sources = model.imageCatalog
        model.imageCatalog = []
        model.activeSection = .images
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertFalse(defaults.bool(forKey: "TurboSpark.imageModelRecommendationSeen"))
        model.imageCatalog = sources
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertTrue(defaults.bool(forKey: "TurboSpark.imageModelRecommendationSeen"))
        XCTAssertFalse(model.isInstallingImageModel, "Presenting setup must not start a download")
        withExtendedLifetime(host) {}
    }
}
