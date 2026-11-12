import SwiftUI
import TurboSpark

extension RepoVariant {
    /// Quantization, size, kernel status and file name, for a picker row.
    ///
    /// Shared by the probe sheet and the Hugging Face result pane so the two
    /// pickers cannot drift apart. The quantization name, size and file are
    /// data; only the two status words translate.
    var pickerLabel: Text {
        var label = quantLabel.map { Text(verbatim: $0) } ?? Text("unrecognized", bundle: .module)
        if let bytes {
            label = label + Text(verbatim: " - \(MetricFormat.storage(bytes))")
        }
        if !executable {
            label = label + Text(verbatim: " (") + Text("no kernels", bundle: .module) + Text(verbatim: ")")
        }
        return label + Text(verbatim: " - \(file)")
    }
}
