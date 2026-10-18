import AppKit
import SwiftUI
import TurboSpark

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Welcome and initial setup view displayed when no local models are detected.
/// Scans system hardware, displays memory limits, highlights MoE streaming advantages,
/// and presents hardware-tailored recommendations.
@MainActor
struct ModelInstallView: View {
    @ObservedObject var model: AppModel
    @State private var recommendations: [ModelRecommendation] = []
    @State private var isLoadingRecommendations = true
    @State private var recommendationProgress: ModelRecommendationProbeProgress?
    @State private var recommendationError: String?
    @State private var selectedFilter: RecommendationFilter = .recommended
    /// The library the user picked for the install. Nil follows the default
    /// (requirement 2.2). Per install: it is not remembered across launches.
    @State private var destination: String?

    enum RecommendationFilter: String, CaseIterable, Identifiable {
        case recommended = "Best Fit (MoE & Fast)"
        case all = "All Catalog Models"

        var id: String { rawValue }
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerSection
                if ModelStorageManager.isLMStudioDirectoryPresent(customPath: model.lmStudioDirectory) {
                    lmStudioDetectedBanner
                }
                HardwareScanCard(telemetry: model.telemetry)
                MoESpotlightCard()

                if model.isInstallingModel {
                    ActiveInstallCard(model: model)
                }

                recommendationsSection
                utilityActionsRow
            }
            // Wide enough that the hardware stats and recommendation rows use
            // the pane instead of sitting in a narrow strip; this screen only
            // shows for an empty chat with nothing installed.
            .frame(maxWidth: 920)
            .padding(.horizontal, 24)
            .padding(.vertical, 32)
            .frame(maxWidth: .infinity)
        }
        // Keep native buttons and segmented controls on the app's live font
        // family and size even when their labels do not declare a role.
        .themedFont(.small)
        .task(id: model.fitRecommendationConfigurationID) {
            await loadRecommendations()
        }
    }

    // MARK: - Header & Identity

    private var lmStudioDetectedBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: "shippingbox.fill")
                .themedFont(.title2)
                .foregroundStyle(Color.accentColor)

            VStack(alignment: .leading, spacing: 2) {
                Text("LM Studio Library Detected", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                Text("Found models at ~/.lmstudio/models. TurboSpark can run your LM Studio models directly without copying files.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }

            Spacer()

            Button {
                model.enableLMStudioDetection = true
                model.persistSettings()
                model.refreshModels()
                model.showToast(AppStrings.localized("Scanned LM Studio models folder"), style: .success)
            } label: {
                Text("Scan LM Studio", bundle: .module)
            }
            .appAction(.secondary, size: .small)
        }
        .padding(12)
        .background(Color.accentColor.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.accentColor.opacity(0.3), lineWidth: 1))
    }

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "bolt.horizontal.circle.fill")
                .themedFont(.display)
                .foregroundStyle(.appAccent)
                .help(Text(verbatim: "TurboSpark"))
                .accessibilityHidden(true)

            Text("Welcome to TurboSpark", bundle: .module)
                .themedFont(.title, weight: .bold)
                .accessibilityAddTraits(.isHeader)

            Text("No local models found in ~/.turbospark. Choose a recommended model below to start.", bundle: .module)
                .themedFont(.base)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
        }
    }

    // MARK: - Recommendations Section

    private var recommendationsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Recommended Models for Your Mac", bundle: .module)
                        .themedFont(.base, weight: .semibold)
                    Group {
                        if let chip = model.telemetry?.chip {
                            Text("Ranked by compatibility with your \(chip) memory budget.", bundle: .module)
                        } else {
                            Text("Ranked by compatibility with this Mac's memory budget.", bundle: .module)
                        }
                    }
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                Picker(selection: $selectedFilter) {
                    ForEach(RecommendationFilter.allCases) { f in
                        recommendationFilterLabel(f).tag(f)
                    }
                } label: { Text("Filter", bundle: .module) }
                .pickerStyle(.segmented)
                .labelsHidden()
                .themedFont(.small)
                .fixedSize()
                .accessibilityLabel(Text("Recommendations filter", bundle: .module))
            }

            // Where an install from this list goes: only when there is a choice
            // (requirement 2.1). A drive with no models yet is a library too.
            InstallDestinationPicker(model: model, selection: $destination)

            if isLoadingRecommendations {
                ModelRecommendationLoadingView(progress: recommendationProgress)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
            } else if let recommendationError {
                VStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle")
                        .themedFont(.title2)
                        .foregroundStyle(.orange)
                    Text(recommendationError)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .multilineTextAlignment(.center)
                    Button {
                        Task { await loadRecommendations() }
                    } label: {
                        Text("Refresh", bundle: .module)
                    }
                    .appAction(.secondary, size: .small)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
            } else if displayedRecommendations.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .themedFont(.title2)
                        .foregroundStyle(.quaternary)
                    Text("No recommended models", bundle: .module)
                        .themedFont(.base, weight: .medium)
                    Button {
                        selectedFilter = .all
                    } label: {
                        Text("All models", bundle: .module)
                    }
                    .appAction(.secondary, size: .small)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
            } else {
                VStack(spacing: 10) {
                    ForEach(displayedRecommendations) { rec in
                        ModelRecommendationRow(
                            recommendation: rec, model: model,
                            library: model.installDestinations.libraryForInstall(selection: destination))
                    }
                }
            }
        }
    }

    // MARK: - Bottom Utility Actions

    private var utilityActionsRow: some View {
        HStack(spacing: 12) {
            Button {
                model.refreshModels()
                Task { await loadRecommendations() }
            } label: {
                Label { Text("Rescan Storage", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
            }
            .appAction(.secondary)
            .help(Text("Rescans local model directories and reloads recommendations", bundle: .module))

            Button {
                ModelLocationPicker.choose(for: model)
            } label: {
                Label { Text("Choose Existing Model Folder...", bundle: .module) } icon: { Image(systemName: "folder") }
            }
            .appAction(.secondary)
            .help(Text("Select an existing model directory", bundle: .module))

            Spacer()

            Button {
                model.openModelHub()
            } label: {
                Label { Text("Browse Full Model Hub", bundle: .module) } icon: { Image(systemName: "square.grid.2x2") }
            }
            .appAction(.secondary)
            .help(Text("Open the model catalog", bundle: .module))
        }
        .padding(.top, 4)
    }

    // MARK: - Helpers

    private var displayedRecommendations: [ModelRecommendation] {
        if selectedFilter == .recommended {
            // Prioritize runnable MoE and high-performance models
            return recommendations.filter { $0.runs }
        }
        return recommendations
    }

    @ViewBuilder
    private func recommendationFilterLabel(_ filter: RecommendationFilter) -> some View {
        switch filter {
        case .recommended:
            Text("Recommended", bundle: .module)
        case .all:
            Text("All models", bundle: .module)
        }
    }

    private func loadRecommendations() async {
        let configurationID = model.fitRecommendationConfigurationID
        isLoadingRecommendations = true
        recommendationProgress = nil
        recommendationError = nil
        do {
            let rows = try await model.loadFitRecommendations { completed, total in
                guard configurationID == model.fitRecommendationConfigurationID else { return }
                recommendationProgress = ModelRecommendationProbeProgress(
                    completed: completed,
                    total: total)
            }
            guard !Task.isCancelled,
                  configurationID == model.fitRecommendationConfigurationID else { return }
            recommendations = rows
            recommendationProgress = nil
            isLoadingRecommendations = false
        } catch {
            guard !Task.isCancelled,
                  configurationID == model.fitRecommendationConfigurationID else { return }
            recommendations = []
            recommendationProgress = nil
            recommendationError = error.localizedDescription
            isLoadingRecommendations = false
        }
    }
}

struct ModelRecommendationProbeProgress: Equatable {
    let completed: UInt32
    let total: UInt32

    var fraction: Double {
        guard total > 0 else { return 0 }
        return min(1, Double(completed) / Double(total))
    }
}

struct ModelRecommendationLoadingView: View {
    let progress: ModelRecommendationProbeProgress?

    var body: some View {
        VStack(spacing: 8) {
            if let progress, progress.total > 0 {
                ProgressView(value: progress.fraction)
                    .progressViewStyle(.linear)
                    .accessibilityValue(MetricFormat.percent(progress.fraction * 100))
                Text(MetricFormat.percent(progress.fraction * 100))
                    .themedFont(.tiny)
                    .monospacedDigit()
                    .foregroundStyle(.appSecondary)
            } else {
                ProgressView()
                    .progressViewStyle(.linear)
            }
            Text("Calculating hardware fit recommendations...", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
        .frame(maxWidth: 360)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text("Calculating hardware fit recommendations...", bundle: .module))
    }
}
