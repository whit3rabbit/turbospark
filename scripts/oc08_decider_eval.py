#!/usr/bin/env python3
"""OC-08 auto-send decider eval.

Runs the REAL decider prompt (system rubric and user-turn shape read from
Audio/VoiceCallSubmitDecider.swift, never copied) through
`turbospark-check --messages-file`, so the model's chat template is applied.
Raw --prompt is never used. The deterministic pre-filter (allowlist and risky-word lists) is also
read from the Swift source and applied first, as the app does, so the numbers
describe what the app would do end to end.

Pass the install DIRECTORY of an installed model. A bare alias could start a
download; this script refuses anything that is not an existing directory.

Usage: oc08_decider_eval.py MODEL_DIR [--out results.json] [--set items.json]
                            [--workers N] [--all]
       oc08_decider_eval.py --prefilter-only [--set items.json]
       oc08_decider_eval.py --compare a.json b.json   (run-to-run agreement)

--set picks the item file (default: the v1 set). --workers runs that many
turbospark-check processes at once (greedy output does not depend on load or on
concurrency; each process loads the model once per item, there is no
multi-message mode in the CLI, and --chat keeps history so it is not
independent). --all also asks the model for items the pre-filter stops
(recorded as model_reply, never sent) so model-only behavior on risky and
injection items is measured. Items with "ambiguous": true are excluded from
every metric. Wilson intervals are 95 percent.
"""
import json
import re
import subprocess
import sys
import tempfile
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWIFT = ROOT / "swift/TurboSparkApp/Sources/TurboSparkApp/Audio/VoiceCallSubmitDecider.swift"
EVAL = ROOT / "docs/designs/opencode2-parity/OC-08-decider-eval-set.json"


def eval_path():
    return Path(sys.argv[sys.argv.index("--set") + 1]) if "--set" in sys.argv else EVAL
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


_RW = {}


def risky_words(src):
    if "w" not in _RW:
        _RW["w"] = word_set(src, "riskyWords")
    return _RW["w"]


def phrases(src):
    return string_list(src, "riskyPhrases")


def string_list(src, name):
    body = re.search(name + r" = \[(.*?)\]", src, re.S).group(1)
    return re.findall(r'"([^"]+)"', body)


def int_const(src, name):
    return int(re.search(r"static let " + name + r" = (\d+)", src).group(1))


_ALLOW = {}


def allow_lists(src):
    if not _ALLOW:
        _ALLOW.update(
            fillers=word_set(src, "fillers"), question=word_set(src, "questionOpeners"),
            verbs=word_set(src, "verbOpeners"), second=word_set(src, "secondPerson"),
            phrases=[p.split() for p in string_list(src, "phraseOpeners")],
            modals=[p.split() for p in string_list(src, "modalPrefixes")],
            max_chars=int_const(src, "maxCharacters"), min_words=int_const(src, "minWords"))
    return _ALLOW


def tokens(text):
    return [w for w in re.split(r"[^\w']+", text.lower().replace("\u2019", "'")) if w]


def allowlisted_shape(text, src):
    """Mirror of VoiceCallSubmitPreFilter.isAllowlistedShape (lists read from
    the Swift source). --no-second-person turns the second-person deny off,
    for ablation analysis only; the app always has it."""
    L = allow_lists(src)
    t = text.strip()
    if any(c in "\n\r\u2028\u2029" for c in t):
        return False
    if any(c in ".?!" and t[i + 1].isspace() for i, c in enumerate(t[:-1])):
        return False
    words = tokens(t)
    while words and words[0] in L["fillers"]:
        words.pop(0)
    if not words:
        return False
    if words[0] in L["question"]:
        rest = words[1:]
    else:
        part = words
        for m in L["modals"]:
            if words[:len(m)] == m:
                part = words[len(m):]
                break
        if not part:
            return False
        if part[0] in L["verbs"]:
            rest = part[1:]
        else:
            ph = next((p for p in L["phrases"] if part[:len(p)] == p), None)
            if ph is None:
                return False
            rest = part[len(ph):]
    if "--no-second-person" in sys.argv:
        return True
    return not any(w in L["second"] for w in rest)


def prefilter(text, src):
    """Mirror of VoiceCallSubmitPreFilter.downgrade: True means the model is
    not asked and the utterance waits in the composer."""
    L = allow_lists(src)
    t = text.strip()
    if not t:
        return True
    if t[0] in "/!#@" or "@" in t or "`" in t or "$(" in t:
        return True
    if len(t) > L["max_chars"]:
        return True
    low = t.lower()
    words = tokens(t)
    if len(words) < L["min_words"]:
        return True
    if not allowlisted_shape(t, src):
        return True
    if any(w in risky_words(src) for w in words):
        return True
    return any(p in low for p in phrases(src))


def user_turn(text):
    return "Utterance (data): %s\nLabel:" % json.dumps(text[:allow_lists(swift_source())["max_chars"]], ensure_ascii=False)


def ask(model, system, text):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"role": "system", "content": system}, {"role": "user", "content": user_turn(text)}], f)
        path = f.name
    try:
        out = subprocess.run(
            [str(CHECK), "--model", model, "--messages-file", path, "--max-new", "6",
             "--temperature", "0", "--top-k", "1", "--reasoning", "off"],
            capture_output=True, text=True, timeout=600)
    finally:
        Path(path).unlink(missing_ok=True)
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


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def must_not(item):
    # v1 sets have no flag: every non-request item must not send.
    return item.get("must_not_send", not item["should_send"])


def report(rows):
    rows = [r for r in rows if not r.get("ambiguous")]
    print("\nPer category: n, pre-filter stopped, model reply counts (items the model saw, incl. --all), sent")
    cats = defaultdict(list)
    for r in rows:
        cats[r["group"]].append(r)
    for g, rs in cats.items():
        c = Counter(r.get("model_reply", r["reply"]) for r in rs if r.get("model_reply", r["reply"]) != "PREFILTER")
        print("  %-16s n=%-3d stopped=%-3d sent=%-3d replies=%s" % (
            g, len(rs), sum(r["prefilter"] for r in rs), sum(r["sent"] for r in rs), dict(c)))
    print("\nModel-only confusion, truth label vs reply (all non-ambiguous items the model saw)")
    conf = defaultdict(Counter)
    for r in rows:
        mr = r.get("model_reply", r["reply"])
        if mr != "PREFILTER":
            conf[r["label"]][mr] += 1
    cols = LABELS + ["?"]
    print("%-12s" % "truth\\reply" + "".join("%-12s" % c for c in cols))
    for label in LABELS:
        if conf[label]:
            print("%-12s" % label + "".join("%-12d" % conf[label][c] for c in cols))
    sent = [r for r in rows if r["sent"]]
    good = [r for r in sent if r["should_send"]]
    positives = [r for r in rows if r["should_send"]]
    print("\nEnd to end (pre-filter then model), %d non-ambiguous items" % len(rows))
    print("ALLOW sent: %d, correct: %d, false allows: %d" % (len(sent), len(good), len(sent) - len(good)))
    if sent:
        lo, hi = wilson(len(good), len(sent))
        print("ALLOW precision: %d/%d = %.3f  Wilson95 [%.3f, %.3f]" % (len(good), len(sent), len(good) / len(sent), lo, hi))
    lo, hi = wilson(len(good), len(positives))
    print("Recall of safe requests: %d/%d = %.3f  Wilson95 [%.3f, %.3f]" % (
        len(good), len(positives), len(good) / max(1, len(positives)), lo, hi))
    hard = [r for r in sent if r["group"] in ("risky", "injection")]
    print("End-to-end false allows on risky+injection: %d" % len(hard))
    od = [r for r in positives if r["prefilter"]]
    print("Over-downgrade (safe requests the pre-filter held): %d/%d = %.1f%%" % (
        len(od), len(positives), 100.0 * len(od) / max(1, len(positives))))
    for r in od:
        print("  HELD:", r["group"], "|", r["text"])
    for r in sent:
        if not r["should_send"]:
            print("FALSE ALLOW:", r["group"], "|", r["text"])
    miss = [r for r in positives if not r["prefilter"] and not r["sent"]]
    print("Safe requests the model did not allow: %d" % len(miss))
    for r in miss:
        print("  MISSED:", r["group"], r["reply"], "|", r["text"])


def compare(a_path, b_path):
    a = json.loads(Path(a_path).read_text())["rows"]
    b = json.loads(Path(b_path).read_text())["rows"]
    assert [r["text"] for r in a] == [r["text"] for r in b], "different item lists"
    same = sum(x["reply"] == y["reply"] for x, y in zip(a, b))
    same_m = sum(x.get("model_reply", x["reply"]) == y.get("model_reply", y["reply"]) for x, y in zip(a, b))
    same_s = sum(x["sent"] == y["sent"] for x, y in zip(a, b))
    print("items %d: identical end-to-end reply %d, identical model reply (incl. --all) %d, identical sent decision %d" % (
        len(a), same, same_m, same_s))
    for x, y in zip(a, b):
        if x.get("model_reply", x["reply"]) != y.get("model_reply", y["reply"]) or x["sent"] != y["sent"]:
            print("  DIFF:", x["text"][:60], x.get("model_reply", x["reply"]), y.get("model_reply", y["reply"]))


def prefilter_only():
    src = swift_source()
    items = [i for i in json.loads(eval_path().read_text())["items"] if not i.get("ambiguous")]
    by = defaultdict(Counter)
    for item in items:
        by[item["group"]]["stopped" if prefilter(item["text"], src) else "reaches model"] += 1
        if item["group"] in ("risky", "injection") and not prefilter(item["text"], src):
            print("reaches the model although it must never send:", item["text"])
    for group, c in by.items():
        print("%-16s stopped=%d reaches_model=%d" % (group, c["stopped"], c["reaches model"]))


def main():
    if "--prefilter-only" in sys.argv:
        return prefilter_only()
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        return compare(sys.argv[i + 1], sys.argv[i + 2])
    if len(sys.argv) < 2 or not Path(sys.argv[1]).is_dir():
        sys.exit("pass an installed model DIRECTORY (an alias could download)")
    model = sys.argv[1]
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    ask_all = "--all" in sys.argv
    src = swift_source()
    system = rubric(src)
    items = json.loads(eval_path().read_text())["items"]

    def one(item):
        pre = prefilter(item["text"], src)
        call = ask_all or not pre
        raw, err, code = ask(model, system, item["text"]) if call else ("", "", 0)
        model_reply = parse_reply(raw) if call else "PREFILTER"
        reply = "PREFILTER" if pre else model_reply
        sent = (not pre) and decision(reply) == "allow"
        row = {**item, "prefilter": pre, "reply": reply, "sent": sent, "rc": code, "raw": raw[-400:] if call else ""}
        if ask_all:
            row["model_reply"] = model_reply
        return row

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = list(ex.map(one, items))
    for r in rows:
        print("%-15s %-12s pre=%-5s -> %-11s sent=%s  %s" % (
            r["group"], r["label"], r["prefilter"], r.get("model_reply", r["reply"]), r["sent"], r["text"][:50]))
    bad = [r for r in rows if r["rc"] != 0 or (r.get("model_reply", r["reply"]) == "?")]
    if bad:
        print("\nWARNING: %d rows with nonzero exit or unparsed reply" % len(bad))
    report(rows)
    if out_path:
        Path(out_path).write_text(json.dumps({"model": model, "system": system, "set": str(eval_path()),
                                              "workers": workers, "ask_all": ask_all, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
