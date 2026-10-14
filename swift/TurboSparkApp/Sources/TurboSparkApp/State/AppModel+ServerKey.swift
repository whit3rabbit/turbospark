import Foundation
import Security

extension AppModel {
    /// Turns Keychain storage of the server API key on or off. Turning it on
    /// saves the key in the field now, which is the one moment macOS may ask
    /// for access, and it stays off if that fails. Turning it off deletes the
    /// item so nothing is left for a later build to be asked to vouch for.
    /// The key in the field is kept either way, so the running session is
    /// unaffected.
    func setServerKeyInKeychain(_ enabled: Bool) {
        guard enabled != serverKeyInKeychain else { return }
        if enabled {
            // An empty field stores nothing yet. The key is saved when typed,
            // by the normal persist path, now that the opt-in is on.
            if let key = Self.serverAPIKey(from: serverAPIKeyInput) {
                let status = ServerKeychain.saveKeyStatus(key)
                guard status == errSecSuccess else {
                    showToast(
                        Self.keychainFailureMessage(
                            "Could not save the server API key to Keychain.", status: status),
                        style: .error)
                    return
                }
            }
            serverKeyLoadFailed = false
        } else {
            ServerKeychain.deleteKey()
            serverKeyLoadFailed = false
        }
        ServerKeychain.setKeychainEnabled(enabled)
        serverKeyInKeychain = enabled
    }
}
