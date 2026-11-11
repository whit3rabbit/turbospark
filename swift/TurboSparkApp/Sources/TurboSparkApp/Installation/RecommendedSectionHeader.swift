import SwiftUI
import TurboSpark

/// The heading over one fit group of the Recommended tab.
///
/// The title says what the group means for the person using it; the subtitle
/// is `ModelFitPresentation`'s own sentence for the verdict, so a group reads
/// the same here as in a card's fit pill (swift Gotcha 22: one mapping, not a
/// restatement).
struct RecommendedSectionHeader: View {
    let verdict: ModelRecommendation.FitVerdict

    var body: some View {
        AppSectionHeader(
            title: title,
            subtitle: Text(LocalizedStringKey(ModelFitPresentation.of(verdict).help), bundle: .module)
        )
        .padding(.top, 10)
        .padding(.horizontal, 2)
    }

    private var title: Text {
        switch verdict {
        case .resident: return Text("Runs great on your Mac", bundle: .module)
        case .streams: return Text("Big models, streamed from disk", bundle: .module)
        case .tight: return Text("Tight on memory", bundle: .module)
        case .unknown: return Text("Not checked on this Mac yet", bundle: .module)
        case .refused:
            // The tab never lists a refused row. If a caller ever does, the
            // existing label is better than a blank heading.
            return Text(LocalizedStringKey(ModelFitPresentation.of(verdict).label), bundle: .module)
        }
    }
}
