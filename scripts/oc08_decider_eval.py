#!/usr/bin/env python3
"""OC-08 auto-send decider eval.

Runs the REAL decider prompt (system rubric and user-turn shape read from
Audio/VoiceCallSubmitDecider.swift, never copied) through
`turbospark-check --messages-file`, so the model's chat template is applied.
Raw --prompt is never used. The deterministic pre-filter word lists are also
read from the Swift source and applied first, as the app does, so the numbers
describe what the app would do end to end.

Pass the install DIRECTORY of an installed model. A bare alias could start a
download; this script refuses anything that is not an existing directory.

Usage: oc08_decider_eval.py MODEL_DIR [--out results.json]
       oc08_decider_eval.py --prefilter-only    (no model, no load: reports what
                                                  the deterministic pre-filter stops)
"""
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWIFT = ROOT / "swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceCallSubmitDecider.swift"
EVAL = ROOT / "docs/designs/opencode2-parity/OC-08-decider-eval-set.json"
CHECK = ROOT / "target/release/turbospark-check"
LABELS = ["REQUEST", "NOISE", "PARTIAL", "DICTATION", "THIRD_PARTY", "UNSURE"]


def swift_source():
    return SWIFT.read_text()


def rubric(src):
    block = re.search(r'static let system = """\n(.*?)\n\s*"""', src, re.S).group(1)
    lines = [l[8:] if l.startswith(" " * 8) else l.lstrip() for l in block.split("\n")]
    # A trailing backslash joins the next line (Swift multi-line literal).
    return re.sub(r"\\\n", "", "\n".join(lines))


def word_set(src, name):
    body = re.search(name + r": Set<String> = \[(.*?)\n    \]", src, re.S).group(1)
    return set(re.findall(r'"([^"]+)"', body))


def phrases(src):
    body = re.search(r"riskyPhrases = \[(.*?)\]", src, re.S).group(1)
    return re.findall(r'"([^"]+)"', body)


def prefilter(text, src):
    """Mirror of VoiceCallSubmitPreFilter.downgrade: True means the model is
    not asked and the utterance waits in the composer."""
    t = text.strip()
    if not t:
        return True
    if t[0] in "/!#@" or "@" in t or "`" in t or "$(" in t:
        return True
    if len(t) > 400:
        return True
    low = t.lower()
    words = [w for w in re.split(r"[^\w']+", low) if w]
    if len(words) < 3:
        return True
    if any(w in word_set(src, "riskyWords") for w in words):
        return True
    return any(p in low for p in phrases(src))


def user_turn(text):
    return "Utterance (data): %s\nLabel:" % json.dumps(text[:400], ensure_ascii=False) + ""


def ask(model, system, text):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"role": "system", "content": system}, {"role": "user", "content": user_turn(text)}], f)
        path = f.name
    out = subprocess.run(
        [str(CHECK), "--model", model, "--messages-file", path, "--max-new", "6",
         "--temperature", "0", "--top-k", "1", "--reasoning", "off"],
        capture_output=True, text=True, timeout=600)
    return out.stdout, out.stderr, out.returncode


def parse_reply(stdout):
    # The reply is the last non-empty content line that is not a banner or
    # a stats footer; the caller keeps the raw text for audit.
    for line in reversed([l.strip() for l in stdout.splitlines() if l.strip()]):
        cleaned = re.sub(r"[^A-Za-z_]", "", line).upper()
        if cleaned in LABELS:
            return cleaned
    return "?"


def decision(reply):
    """Mirror of VoiceCallSubmitRubric.decision: only the exact label REQUEST allows."""
    return "allow" if reply == "REQUEST" else "not_allow"


def prefilter_only():
    src = swift_source()
    items = json.loads(EVAL.read_text())["items"]
    by = defaultdict(Counter)
    for item in items:
        by[item["group"]]["stopped" if prefilter(item["text"], src) else "reaches model"] += 1
        if item["group"] in ("risky", "injection") and not prefilter(item["text"], src):
            print("reaches the model although it must never send:", item["text"])
    for group, c in by.items():
        print("%-12s stopped=%d reaches_model=%d" % (group, c["stopped"], c["reaches model"]))


def main():
    if "--prefilter-only" in sys.argv:
        return prefilter_only()
    if len(sys.argv) < 2 or not Path(sys.argv[1]).is_dir():
        sys.exit("pass an installed model DIRECTORY (an alias could download)")
    model = sys.argv[1]
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    src = swift_source()
    system = rubric(src)
    items = json.loads(EVAL.read_text())["items"]
    rows = []
    for item in items:
        pre = prefilter(item["text"], src)
        raw, err, code = ("", "", 0) if pre else ask(model, system, item["text"])
        reply = "PREFILTER" if pre else parse_reply(raw)
        sent = (not pre) and decision(reply) == "allow"
        rows.append({**item, "prefilter": pre, "reply": reply, "sent": sent, "rc": code,
                     "raw": raw[-400:] if not pre else ""})
        print("%-11s %-12s pre=%-5s -> %-11s sent=%s  %s" % (
            item["group"], item["label"], pre, reply, sent, item["text"][:50]), flush=True)

    asked = [r for r in rows if not r["prefilter"]]
    print("\nModel-only confusion (items that reached the model: %d)" % len(asked))
    conf = defaultdict(Counter)
    for r in asked:
        conf[r["label"]][r["reply"]] += 1
    cols = LABELS + ["?"]
    print("%-12s" % "truth\\reply" + "".join("%-12s" % c for c in cols))
    for label in LABELS:
        if conf[label]:
            print("%-12s" % label + "".join("%-12d" % conf[label][c] for c in cols))
    sent = [r for r in rows if r["sent"]]
    good = [r for r in sent if r["should_send"]]
    positives = [r for r in rows if r["should_send"]]
    print("\nEnd to end (pre-filter then model), %d items" % len(rows))
    print("ALLOW sent: %d, correct: %d, false allows: %d" % (len(sent), len(good), len(sent) - len(good)))
    if sent:
        print("ALLOW precision: %d/%d = %.3f" % (len(good), len(sent), len(good) / len(sent)))
    print("Recall of safe requests: %d/%d" % (len(good), len(positives)))
    for r in sent:
        if not r["should_send"]:
            print("FALSE ALLOW:", r["group"], "|", r["text"])
    if out_path:
        Path(out_path).write_text(json.dumps({"model": model, "system": system, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
