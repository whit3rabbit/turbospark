import Foundation

public enum DOMSnapshotFixtures {
    public static func url(named name: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: "html")
    }
}
