import Foundation

/// What the helper learned about the process on the other end of its socket.
public struct ComputerUsePeer: Equatable, Sendable {
    public let uid: uid_t
    public let bundlePath: String?
    public let signatureValid: Bool

    public init(uid: uid_t, bundlePath: String?, signatureValid: Bool) {
        self.uid = uid
        self.bundlePath = bundlePath
        self.signatureValid = signatureValid
    }
}

/// Whether the helper serves a peer.
///
/// The helper holds Accessibility and Screen Recording, so anything that can
/// talk to it borrows both. It serves only the validly signed app bundle it
/// ships inside, running as the same user. The token
/// proves the peer is the process that launched this helper instance.
///
/// Known limit: TurboSpark is ad-hoc signed, so "validly signed" cannot pin a
/// developer identity. A same-user process able to rewrite and re-sign the
/// whole app bundle passes. A Developer ID signature would close that.
public enum ComputerUsePeerPolicy {
    public static func accepts(
        peer: ComputerUsePeer,
        ownUID: uid_t,
        expectedBundlePath: String?,
        presentedToken: String,
        expectedToken: String
    ) -> Bool {
        guard peer.uid == ownUID,
              peer.signatureValid,
              let expected = expectedBundlePath.map(canonical),
              let actual = peer.bundlePath.map(canonical),
              expected == actual,
              !expectedToken.isEmpty
        else { return false }
        return constantTimeEqual(presentedToken, expectedToken)
    }

    /// The app bundle containing a helper installed at
    /// `<App>.app/Contents/Helpers/<Helper>.app`, or nil when the helper is
    /// not installed that way (a bare `swift build` product). Code signing
    /// reports a bundled peer's path as its bundle, so this is what a peer's
    /// path is compared with.
    public static func containingAppBundle(helperBundleURL: URL) -> URL? {
        let helpers = helperBundleURL.deletingLastPathComponent()
        let contents = helpers.deletingLastPathComponent()
        let app = contents.deletingLastPathComponent()
        guard helperBundleURL.pathExtension == "app",
              helpers.lastPathComponent == "Helpers",
              contents.lastPathComponent == "Contents",
              app.pathExtension == "app"
        else { return nil }
        return app
    }

    static func canonical(_ path: String) -> String {
        URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL.path
    }

    static func constantTimeEqual(_ a: String, _ b: String) -> Bool {
        let x = Array(a.utf8)
        let y = Array(b.utf8)
        guard x.count == y.count else { return false }
        var diff: UInt8 = 0
        for i in 0..<x.count { diff |= x[i] ^ y[i] }
        return diff == 0
    }
}
