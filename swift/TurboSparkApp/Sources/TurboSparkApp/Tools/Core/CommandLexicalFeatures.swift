import Foundation

/// The 20 hand-written features the permission-gate models were trained with,
/// in the published contract's order.
///
/// **THE KEYWORD LISTS ARE COPIED VERBATIM FROM THE TRAINER AND ABBREVIATING
/// ANY OF THEM SILENTLY MOVES THE SCORE.** They are in their own file so they
/// can be diffed against `train_permission_gate.py` without reading the
/// scoring code around them. `CommandGateTests` pins the result through the
/// oracle fixture rather than asserting these lists directly, so a divergence
/// shows up as a parity failure rather than as a plausible new number.
///
/// Two of these are weaker than they look and are kept anyway, because the
/// weights were fitted against them:
/// - `hasDiscoveryCommand` matches the bare substring "id", which fires on any
///   command containing those two letters.
/// - `hasArchiveCommand` matches "tar " which fires inside "startup ".
enum CommandLexicalFeatures {

    /// Feature count, checked against the weights file header at load.
    static let count = 20

    /// Counts saturate here, matching the trainer's `COUNT_CAP`.
    private static let countCap: Float = 8

    /// `command_length` is `log1p(len) / log1p(4096)`, matching `LENGTH_SCALE`.
    private static let lengthScale = Float(log1p(4096.0))

    private static let metaCharacters: Set<Character> = [
        "|", ";", "&", "$", "`", "(", ")", ">", "<",
    ]

    private static let sudoPattern = try? NSRegularExpression(pattern: "(^|\\s)sudo(\\s|$)")

    static func compute(_ command: String) -> [Float] {
        let lower = command.lowercased()

        func contains(_ needles: [String]) -> Float {
            needles.contains(where: { lower.contains($0) }) ? 1 : 0
        }
        func occurrences(of character: Character) -> Float {
            Float(command.reduce(into: 0) { total, c in if c == character { total += 1 } })
        }
        func bounded(_ character: Character) -> Float {
            min(occurrences(of: character), countCap) / countCap
        }

        let metaCount = command.reduce(into: 0) { total, c in
            if metaCharacters.contains(c) { total += 1 }
        }

        let fetchesRemotely = lower.contains("curl") || lower.contains("wget")
        let pipesIntoShell = lower.contains("| sh") || lower.contains("| bash")
        let range = NSRange(location: 0, length: (lower as NSString).length)
        let hasSudo = sudoPattern?.firstMatch(in: lower, range: range) != nil

        return [
            contains(["rm -rf", "rmdir /s"]),
            contains(["-delete", "shred -u", "del /f"]),
            ((fetchesRemotely && pipesIntoShell)
                || (lower.contains("invoke-webrequest") && lower.contains("iex"))
                || (lower.contains("certutil") && lower.contains(".exe"))) ? 1 : 0,
            contains([
                "/etc", "/root", "/var/lib", "~/.ssh", "/home/", "c:\\windows",
                "system32", "/bin/", "/usr/bin",
            ]),
            contains(["chmod -r 777", "chmod 777", "chown root", "icacls"]),
            hasSudo ? 1 : 0,
            contains([
                "curl ", "wget ", "invoke-webrequest", "bitsadmin", "certutil",
                "ftp ", "scp ",
            ]),
            contains([
                "| sh", "| bash", " os.system", "powershell", "cmd /c", "bash -lc",
                "eval ", "iex ", "sh -c",
            ]),
            contains([
                "base64", "unhexlify", "fromhex", "-enc", "rot13", "^key",
                "printf '%s'",
            ]),
            (command.contains("$(") || command.contains("`")) ? 1 : 0,
            contains(["-encodedcommand", "frombase64string", "powershell -enc"]),
            contains([" kill ", " pkill ", "taskkill", "sc stop", "systemctl stop"]),
            contains(["tar ", "zip ", "gzip ", "7z "]),
            contains([
                "whoami", "id", "uname", "df -h", "ps aux", "systemctl status",
                "ls ", "pwd", "cat ",
            ]),
            bounded("|"),
            bounded(";"),
            bounded("&"),
            bounded("\n"),
            Float(metaCount) / Float(max(1, command.count)),
            Float(log1p(Double(command.count))) / lengthScale,
        ]
    }
}
