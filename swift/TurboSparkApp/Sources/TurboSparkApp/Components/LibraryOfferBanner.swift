import SwiftUI

/// The offer to add a library found on a drive that just connected (task 6.6,
/// requirements 4.3 and 4.4).
///
/// It is drawn at the top of the Installed Models manager AND at the top of the
/// Model libraries section, from the same state, so answering in one makes the
/// other disappear on the next publish. It is the pending offer only; the rest
/// wait behind it and the banner says how many.
///
/// **IT OBSERVES `LibraryState` DIRECTLY.** `AppModel` does not publish it (a
/// mount would redraw every screen), so each placement holds the state as its own
/// `@ObservedObject` and this view takes it the same way.
///
/// **IT DRAWS A `LibraryOfferBannerModel` AND NOTHING ELSE,** so a table test
/// pins every word, and VoiceOver reads the same strings a sighted user does. The
/// library's name is untrusted: the model escaped and cut it, and this only draws
/// it (`Text(verbatim:)`, never a key).
///
/// **A REFUSAL STAYS INSIDE THE BANNER,** as a glyph and one sentence, and the
/// offer stays. It is never the app error and never a toast; the one toast that
/// announces an offer is `LibraryOfferAnnouncer`'s.
///
/// Isolated explicitly: its helpers call each other, and only `body` is isolated
/// by the protocol on the macOS 14 SDK.
@MainActor
struct LibraryOfferBanner: View {
    @ObservedObject var model: AppModel
    @ObservedObject var libraryState: LibraryState
    /// A test passes its own, wired to a recording announcer, so it can drive a
    /// refusal or an add in flight. The app lets the card build the real one.
    var controller: LibraryOfferController?

    var body: some View {
        if let offer = libraryState.pendingOffer {
            // Keyed by the offer, so the next one starts with no sentence about
            // the last one's refusal and no add in flight.
            LibraryOfferBannerCard(
                model: model, libraryState: libraryState, offer: offer,
                queuedCount: libraryState.queuedOffers.count, controller: controller
            )
            .id(offer.markerId)
        }
    }
}

@MainActor
private struct LibraryOfferBannerCard: View {
    @ObservedObject var model: AppModel
    @ObservedObject var libraryState: LibraryState
    @StateObject private var controller: LibraryOfferController
    let offer: LibraryOffer
    let queuedCount: Int

    /// Read so the strings built from `AppStrings` follow the language picker.
    @Environment(\.locale) private var locale

    init(
        model: AppModel, libraryState: LibraryState, offer: LibraryOffer, queuedCount: Int,
        controller: LibraryOfferController?
    ) {
        self.model = model
        self.libraryState = libraryState
        self.offer = offer
        self.queuedCount = queuedCount
        _controller = StateObject(
            wrappedValue: controller
                ?? LibraryOfferController(
                    libraryState: libraryState,
                    isBusy: { [weak model] in model?.isInstallOrMoveRunning ?? false }))
    }

    private var language: AppLanguage {
        _ = locale
        return AppStrings.currentLanguage
    }

    var body: some View {
        let row = LibraryOfferBannerModel(offer: offer, queuedCount: queuedCount, language: language)
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: LibraryOfferGlyph.drive)
                    .themedFont(.title3)
                    .foregroundStyle(.appAccent)
                    .frame(minWidth: 24)
                    .accessibilityHidden(true)
                summary(row)
            }
            if let failure = controller.failure {
                failureLine(failure)
            }
            actions
        }
        .padding(10)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(.appAccent, lineWidth: 1)
        )
        // The text is one element and the two buttons are real controls inside
        // it, in reading order: VoiceOver and Full Keyboard Access reach both.
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("library-offer-banner")
    }

    // MARK: - Text

    private func summary(_ row: LibraryOfferBannerModel) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(verbatim: row.title)
                .themedFont(.base, weight: .semibold)
                .foregroundStyle(.appText)
                .fixedSize(horizontal: false, vertical: true)
            Text(verbatim: row.countText)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            if let note = row.readOnlyText {
                noteLine(glyph: LibraryOfferGlyph.readOnly, tint: AnyShapeStyle(AppThemeColor.appSecondary), text: note)
            }
            if let queued = row.queuedText {
                Text(verbatim: queued)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: row.accessibilityLabel))
        .accessibilityAddTraits(.isHeader)
    }

    /// A glyph and a sentence, so the line survives color blindness and
    /// Differentiate Without Color.
    private func noteLine(glyph: String, tint: AnyShapeStyle, text: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: glyph).foregroundStyle(tint).accessibilityHidden(true)
            Text(verbatim: text)
                .themedFont(.small)
                .foregroundStyle(.appText)
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private func failureLine(_ sentence: String) -> some View {
        noteLine(glyph: LibraryOfferGlyph.failure, tint: AnyShapeStyle(AppAttention.color), text: sentence)
            .accessibilityElement(children: .combine)
    }

    // MARK: - Actions

    /// Add and Not now side by side while they fit, one under the other when
    /// they do not (a narrow pane, a long translation, the largest text size).
    /// An `HStack` mirrors in a right-to-left language by itself.
    private var actions: some View {
        let busy = model.isInstallOrMoveRunning
        let reason = LibraryOfferBannerModel.disabledReason(
            adding: controller.isAdding, busy: busy, language: language)
        return ViewThatFits(in: .horizontal) {
            HStack(spacing: AppMetrics.actionSpacing) { actionItems(blocked: controller.isChangeBlocked, reason: reason) }
            VStack(alignment: .leading, spacing: 6) { actionItems(blocked: controller.isChangeBlocked, reason: reason) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder private func actionItems(blocked: Bool, reason: String?) -> some View {
        addButton(blocked: blocked, reason: reason)
        notNowButton(blocked: blocked, reason: reason)
        if controller.isAdding {
            HStack(spacing: 8) {
                // A range, not a fixed width (design-system ProgressBar guide):
                // the bar may give way to the wrapping sentence beside it.
                AppProgressBar(value: nil, label: Text("Adding the library\u{2026}", bundle: .module), size: .compact)
                    .frame(minWidth: 40, idealWidth: 80, maxWidth: 80)
                // Wraps instead of truncating: a long translation at the
                // largest text size would otherwise cut the sentence to a word.
                Text("Adding the library\u{2026}", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .accessibilityElement(children: .combine)
        }
    }

    private func addButton(blocked: Bool, reason: String?) -> some View {
        Button {
            Task { await controller.add(offer) }
        } label: {
            Label {
                Text("Add Library", bundle: .module)
            } icon: {
                Image(systemName: "plus")
            }
            // The theme sizes the label, so it follows the text-size setting and
            // matches the borderless button beside it (a small bordered control
            // would otherwise draw a smaller system font than that one).
            .themedFont(.small)
        }
        .appAction(.primary, size: .small)
        .disabled(blocked)
        .modifier(DisabledReason(reason: reason))
    }

    private func notNowButton(blocked: Bool, reason: String?) -> some View {
        let help = LibraryOfferBannerModel.notNowHelp(language: language)
        return Button {
            controller.notNow(offer)
        } label: {
            Text("Not now", bundle: .module)
                .themedFont(.small)
        }
        .appAction(.quiet, size: .small)
        .disabled(blocked)
        // A disabled control says why; an enabled one says what it does.
        .help(Text(verbatim: reason ?? help))
        .accessibilityHint(Text(verbatim: help))
    }
}

/// The tooltip of a disabled button: why it is disabled. Nothing while enabled.
private struct DisabledReason: ViewModifier {
    let reason: String?

    func body(content: Content) -> some View {
        if let reason {
            content.help(Text(verbatim: reason))
        } else {
            content
        }
    }
}
