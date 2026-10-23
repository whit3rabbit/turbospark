//! Hand-rolled argument parsing, as `crates/bench/src/main.rs` does: the
//! workspace carries no CLI parser and this binary does not need one.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::knobs;
use crate::workload::{Arm, OpenParams, PromptSource, Residency, Slots};

pub struct Args {
    rest: Vec<String>,
}

impl Args {
    pub fn new(rest: Vec<String>) -> Self {
        Self { rest }
    }

    /// The value after `flag`, removed from the list. Last occurrence wins.
    pub fn take(&mut self, flag: &str) -> Result<Option<String>, String> {
        let mut found = None;
        while let Some(i) = self.rest.iter().position(|a| a == flag) {
            if i + 1 >= self.rest.len() {
                return Err(format!("{flag} needs a value"));
            }
            found = Some(self.rest.remove(i + 1));
            self.rest.remove(i);
        }
        Ok(found)
    }

    /// Every value after every `flag`, in order.
    pub fn take_all(&mut self, flag: &str) -> Result<Vec<String>, String> {
        let mut out = Vec::new();
        while let Some(i) = self.rest.iter().position(|a| a == flag) {
            if i + 1 >= self.rest.len() {
                return Err(format!("{flag} needs a value"));
            }
            out.push(self.rest.remove(i + 1));
            self.rest.remove(i);
        }
        Ok(out)
    }

    pub fn flag(&mut self, flag: &str) -> bool {
        let before = self.rest.len();
        self.rest.retain(|a| a != flag);
        self.rest.len() != before
    }

    pub fn required(&mut self, flag: &str) -> Result<String, String> {
        self.take(flag)?
            .ok_or_else(|| format!("{flag} is required"))
    }

    /// Refuse anything left over: a mistyped flag must not be ignored.
    pub fn finish(self) -> Result<(), String> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(format!("unrecognized arguments: {}", self.rest.join(" ")))
        }
    }

    pub fn parsed<T: std::str::FromStr>(&mut self, flag: &str, default: T) -> Result<T, String> {
        match self.take(flag)? {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("{flag}: cannot parse {v:?}")),
        }
    }
}

/// `protocol:ID`, `messages:PATH`, `text:PATH` or `corpus:N`.
pub fn parse_prompt(s: &str) -> Result<PromptSource, String> {
    match s.split_once(':') {
        Some(("protocol", id)) => Ok(PromptSource::Protocol(id.to_string())),
        Some(("messages", p)) => Ok(PromptSource::MessagesFile(PathBuf::from(p))),
        Some(("text", p)) => Ok(PromptSource::TextFile(PathBuf::from(p))),
        Some(("corpus", n)) => Ok(PromptSource::Corpus(
            n.parse().map_err(|_| format!("corpus length {n:?}"))?,
        )),
        _ => Err(format!(
            "prompt {s:?}: use protocol:<case>, messages:<path> or text:<path>"
        )),
    }
}

pub fn parse_slots(s: &str) -> Result<Slots, String> {
    if s == "auto" {
        return Ok(Slots::Auto);
    }
    let n: u32 = s.parse().map_err(|_| format!("slots {s:?}"))?;
    if !crate::knobs::ALLOWED_SLOTS.contains(&n) {
        return Err(format!(
            "slots {n}: allowed {:?}",
            crate::knobs::ALLOWED_SLOTS
        ));
    }
    Ok(Slots::Fixed(n as usize))
}

pub fn parse_residency(s: &str) -> Result<Residency, String> {
    match s {
        "auto" => Ok(Residency::Auto),
        "streamed" => Ok(Residency::Streamed),
        "mapped" => Ok(Residency::Mapped),
        _ => Err(format!("residency {s:?}: auto, streamed or mapped")),
    }
}

/// `KEY=VALUE`, where KEY is a full `TURBOSPARK_*` name or a knob's short
/// name from the registry.
pub fn parse_env(s: &str) -> Result<(String, String), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("{s:?}: expected KEY=VALUE"))?;
    let key = if k.starts_with("TURBOSPARK_") {
        k.to_string()
    } else if let Some(knob) = knobs::by_name(k) {
        knob.env.to_string()
    } else {
        return Err(format!(
            "{k:?} is neither a TURBOSPARK_* variable nor a known knob name"
        ));
    };
    Ok((key, v.to_string()))
}

/// `NAME[@SLOTS/RESIDENCY][~DRAFTER:BLOCK[a]][:KEY=VALUE,KEY=VALUE]`. Open parameters default
/// to the command's; the common env goes to every arm.
pub fn parse_arm(
    s: &str,
    open: OpenParams,
    common: &BTreeMap<String, String>,
) -> Result<Arm, String> {
    // `~DRAFTER:BLOCK[a]` runs the arm through the speculative loop. It sits
    // before any `:KEY=V` list and has its own ':', so split it off first.
    let (main, speculation) = match s.split_once('~') {
        None => (s.to_string(), None),
        Some((before, after)) => {
            let mut it = after.splitn(3, ':');
            let drafter = it.next().unwrap_or("");
            let block = it
                .next()
                .ok_or_else(|| format!("arm {s:?}: speculation is ~DRAFTER:BLOCK"))?;
            let sp = parse_speculation(&format!("{drafter}:{block}"))?;
            let main = match it.next() {
                Some(env) => format!("{before}:{env}"),
                None => before.to_string(),
            };
            (main, Some(sp))
        }
    };
    let s = main.as_str();
    let (head, rest) = s.split_once(':').unwrap_or((s, ""));
    // `NAME@SLOTS/RESIDENCY` overrides the shared open parameters for this
    // arm, so one interleaved session can compare slot counts and residency.
    let (name, open) = match head.split_once('@') {
        None => (head, open),
        Some((name, o)) => {
            let (slots, residency) = o
                .split_once('/')
                .ok_or_else(|| format!("arm {head:?}: open parameters are SLOTS/RESIDENCY"))?;
            (
                name,
                OpenParams {
                    slots: parse_slots(slots)?,
                    residency: parse_residency(residency)?,
                },
            )
        }
    };
    crate::paths::validate_name(name)?;
    let mut env = common.clone();
    for kv in rest.split(',').filter(|kv| !kv.is_empty()) {
        let (k, v) = parse_env(kv)?;
        env.insert(k, v);
    }
    Ok(Arm {
        name: name.to_string(),
        env,
        open,
        speculation,
    })
}

/// `mtp:2`, `dflash:7`, `auto:4`, with a trailing `a` for adaptive depth
/// (`mtp:4a`).
pub fn parse_speculation(s: &str) -> Result<crate::workload::Speculation, String> {
    use crate::workload::{Drafter, Speculation};
    let (d, b) = s
        .split_once(':')
        .ok_or_else(|| format!("speculation {s:?}: DRAFTER:BLOCK, e.g. mtp:2"))?;
    let drafter = match d {
        "auto" => Drafter::Auto,
        "mtp" => Drafter::Mtp,
        "dflash" => Drafter::Dflash,
        _ => return Err(format!("drafter {d:?}: auto, mtp or dflash")),
    };
    let (b, adaptive) = match b.strip_suffix('a') {
        Some(b) => (b, true),
        None => (b, false),
    };
    let block: usize = b.parse().map_err(|_| format!("block {b:?}"))?;
    if block == 0 {
        return Err("block must be at least 1".into());
    }
    Ok(Speculation {
        drafter,
        block,
        adaptive,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_and_finish() {
        let mut a = Args::new(
            ["--model", "m", "--env", "A=1", "--env", "B=2", "--x"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        assert_eq!(a.take("--model").unwrap().as_deref(), Some("m"));
        assert_eq!(a.take_all("--env").unwrap(), vec!["A=1", "B=2"]);
        assert!(a.finish().is_err(), "leftover --x must be refused");
    }

    #[test]
    fn arms_accept_short_knob_names() {
        let arm = parse_arm(
            "off:prefill_steel=0",
            OpenParams::default(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(arm.env["TURBOSPARK_PREFILL_STEEL"], "0");
        assert!(parse_arm("x:PATH=/bin", OpenParams::default(), &BTreeMap::new()).is_err());
        assert!(parse_arm("bad name", OpenParams::default(), &BTreeMap::new()).is_err());
    }

    #[test]
    fn arms_can_override_open_parameters() {
        let arm = parse_arm("s32@32/streamed", OpenParams::default(), &BTreeMap::new()).unwrap();
        assert_eq!(arm.name, "s32");
        assert_eq!(arm.open.slots, Slots::Fixed(32));
        assert_eq!(arm.open.residency, Residency::Streamed);
        let arm = parse_arm(
            "m@auto/mapped:prefill_steel=0",
            OpenParams::default(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(arm.open.residency, Residency::Mapped);
        assert_eq!(arm.env["TURBOSPARK_PREFILL_STEEL"], "0");
        assert!(parse_arm("x@32", OpenParams::default(), &BTreeMap::new()).is_err());
    }

    #[test]
    fn arms_can_speculate() {
        let a = parse_arm("mtp2~mtp:2", OpenParams::default(), &BTreeMap::new()).unwrap();
        assert_eq!(a.name, "mtp2");
        assert_eq!(a.speculation.as_ref().unwrap().block, 2);
        let a = parse_arm(
            "d7@16/streamed~dflash:7a:prefill_steel=0",
            OpenParams::default(),
            &BTreeMap::new(),
        )
        .unwrap();
        let sp = a.speculation.unwrap();
        assert!(sp.adaptive && sp.block == 7);
        assert_eq!(a.open.slots, Slots::Fixed(16));
        assert_eq!(a.env["TURBOSPARK_PREFILL_STEEL"], "0");
        assert!(parse_arm("x~mtp:0", OpenParams::default(), &BTreeMap::new()).is_err());
        assert!(parse_arm("x~eagle:2", OpenParams::default(), &BTreeMap::new()).is_err());
    }

    #[test]
    fn slots_and_prompts() {
        assert_eq!(parse_slots("auto").unwrap(), Slots::Auto);
        assert_eq!(parse_slots("32").unwrap(), Slots::Fixed(32));
        assert!(parse_slots("33").is_err());
        assert_eq!(
            parse_prompt("protocol:long-synthesis").unwrap(),
            PromptSource::Protocol("long-synthesis".into())
        );
        assert!(parse_prompt("long-synthesis").is_err());
    }
}
