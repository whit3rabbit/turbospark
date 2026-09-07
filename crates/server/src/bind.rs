//! Interface binding and Tailscale host resolution for `turbospark-server`.

/// Interface the server listens on. Resolution fails rather than widening:
/// there is no path from `Tailnet` to a wildcard or LAN address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BindMode {
    Loopback,
    Tailnet,
}

impl BindMode {
    pub fn host(self) -> Result<String, String> {
        match self {
            BindMode::Loopback => Ok("127.0.0.1".to_string()),
            BindMode::Tailnet => tailnet_host(&tailscale_ipv4_output()?),
        }
    }
}

/// Accepts exactly one Tailscale IPv4 address. Empty, ambiguous, IPv6-only,
/// malformed, and off-range output all fail; none of them fall back.
pub fn tailnet_host(output: &str) -> Result<String, String> {
    let fields: Vec<&str> = output.split_whitespace().collect();
    match fields.as_slice() {
        [] => Err(
            "tailscale reported no IPv4 address; ensure Tailscale is running and connected"
                .to_string(),
        ),
        [only] if is_tailscale_ipv4(only) => Ok((*only).to_string()),
        // Truncated by CHARACTER count, not by byte offset: a raw
        // `&only[..64]` panics if byte 64 falls inside a multibyte UTF-8
        // sequence, which arbitrary subprocess output is not guaranteed to
        // avoid.
        [only] => Err(format!(
            "tailscale reported \"{}\", which is not a Tailnet IPv4 address",
            only.chars().take(64).collect::<String>()
        )),
        many => Err(format!(
            "tailscale reported {} IPv4 addresses; refusing to guess which to bind",
            many.len()
        )),
    }
}

/// True for a dotted-quad IPv4 inside 100.64.0.0/10, the range Tailscale
/// allocates from. Restricting to that range keeps a wildcard, loopback, or
/// LAN address from ever being bound.
pub fn is_tailscale_ipv4(text: &str) -> bool {
    let parts: Vec<&str> = text.split('.').collect();
    if parts.len() != 4 {
        return false;
    }
    let mut octets = [0u8; 4];
    for (slot, part) in octets.iter_mut().zip(parts) {
        // Reject leading zeros: "100.064.0.1" would otherwise pass here and
        // then be read as octal by some resolvers.
        if part.len() > 1 && part.starts_with('0') {
            return false;
        }
        match part.parse::<u8>() {
            Ok(octet) => *slot = octet,
            Err(_) => return false,
        }
    }
    octets[0] == 100 && (64..=127).contains(&octets[1])
}

/// Raw stdout of `tailscale ip -4`. Spawned directly with no shell, so
/// nothing is interpolated into a command line.
pub fn tailscale_ipv4_output() -> Result<String, String> {
    let out = std::process::Command::new("tailscale")
        .args(["ip", "-4"])
        .stderr(std::process::Stdio::null())
        .output()
        .map_err(|e| {
            format!("could not run tailscale ({e}); install its CLI and keep it on PATH")
        })?;
    if !out.status.success() {
        return Err(format!(
            "tailscale ip -4 exited with {}; ensure Tailscale is running and connected",
            out.status
        ));
    }
    String::from_utf8(out.stdout)
        .map_err(|_| "tailscale ip -4 returned non-UTF-8 output".to_string())
}
