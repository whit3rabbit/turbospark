import SwiftUI

/// OC-07: the welcome tour, a popup card over a dimmed main window (RootView
/// mounts it only while `FirstRunCoordinator.isActive`, which now happens only
/// from Help > Show Welcome Tour). Every step is
/// skippable, Skip Setup is always in the header, Back is always allowed, and
/// focus lands on the primary action of each step. No step requests a macOS
/// permission, a credential or the network.
struct FirstRunFlowView: View {
    @ObservedObject var coordinator: FirstRunCoordinator
    @Environment(\.appTheme) private var theme
    @FocusState private var primaryFocused: Bool

    private var stepNumber: Int { coordinator.index + 1 }
    private var stepCount: Int { coordinator.steps.count }

    var body: some View {
        ZStack {
            // Dimmed scrim: swallows clicks so the window behind is inert,
            // but the tour is a card over it, not a full-screen takeover.
            Rectangle()
                .fill(Color.black.opacity(0.35))
                .ignoresSafeArea()
                .contentShape(Rectangle())
                .onTapGesture {}
            ScrollView {
                VStack(spacing: 28) {
                    header
                    stepBody
                        .frame(maxWidth: .infinity)
                        .id(coordinator.current)
                        .transition(.opacity)
                    footer
                }
                .padding(28)
            }
            .frame(width: (560 * AppMetrics.textScale(theme)).rounded())
            .frame(maxHeight: (640 * AppMetrics.textScale(theme)).rounded())
            .fixedSize(horizontal: false, vertical: true)
            .background(.appPage, in: RoundedRectangle(cornerRadius: 14))
            .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.appBorder))
            .shadow(color: .black.opacity(0.25), radius: 24, y: 8)
            .padding(24)
        }
        .accessibilityAddTraits(.isModal)
        .tsAnimation(TSMotion.pane, value: coordinator.current)
        // Focus moves to the primary action whenever the step changes, so
        // Return always advances and a keyboard user never starts on the
        // wrong control.
        .onChange(of: coordinator.current, initial: true) { _, _ in
            primaryFocused = true
            announceProgress()
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text("Step \(stepNumber) of \(stepCount)", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 0)
            Button {
                coordinator.skipAll()
            } label: {
                Text("Skip Setup", bundle: .module)
            }
            .appAction(.quiet, size: .small)
            // Escape ends the flow the same way the visible button does.
            .keyboardShortcut(.cancelAction)
        }
    }

    @ViewBuilder
    private var stepBody: some View {
        switch coordinator.current {
        case .welcome:
            FirstRunWelcomeStep()
        case .firstModel:
            FirstRunModelStep(browseModels: coordinator.browseModels)
        case .primer:
            FirstRunPrimerStep()
        case .importHandoff, .none:
            // `.importHandoff` is never in a step list until the import
            // wizard exists (`FirstRunFlow.importWizardAvailable`), and then
            // it gets a real view here.
            EmptyView()
        }
    }

    private var footer: some View {
        HStack(spacing: AppMetrics.actionSpacing) {
            if coordinator.canGoBack {
                Button {
                    coordinator.back()
                } label: {
                    Text("Back", bundle: .module)
                }
                .appAction(.secondary)
            }
            Spacer(minLength: 0)
            if !coordinator.isLastStep {
                Button {
                    coordinator.skipStep()
                } label: {
                    Text("Skip", bundle: .module)
                }
                .appAction(.quiet)
            }
            Button {
                coordinator.advance()
            } label: {
                if coordinator.isLastStep {
                    Text("Get Started", bundle: .module)
                } else {
                    Text("Continue", bundle: .module)
                }
            }
            .appAction(.primary)
            .keyboardShortcut(.defaultAction)
            .focused($primaryFocused)
        }
    }

    private func announceProgress() {
        guard coordinator.current != nil else { return }
        let message = AppStrings.localized("Step \(stepNumber) of \(stepCount)")
        _ = AccessibilityNotification.Announcement.post(.init(message))
    }
}
