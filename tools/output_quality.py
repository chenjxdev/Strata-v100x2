"""tools/output_quality.py - the OUTPUT degeneration yardstick (1Cat-vLLM's SM70 quality logs, ported).

`quant_quality.py` scores WEIGHTS (teacher-forced top-1/KL against Q8_0).  This one scores what the user actually
sees: one reply's text, checked for the failure shapes 1Cat's SM70 logs caught before they caught us - the
"numeric gibberish" of a stale graph-replay buffer (digit_ratio 0.59 vs 0.0036 clean on the same prompt) and the
"repeated fragment collapse" of a broken spec-decode state (`555...`, `px3px3...`, `UTF-UTF...`, acceptance pinned
at 1.000).  Both were ENGINE bugs that no perplexity run flagged; a cheap text metric did.

    python tools/output_quality.py OUT [OUT ...] [--json RUN.json] [thresholds...]
    python tools/output_quality.py --compare BASE.json CAND.json

OUT: a .json/.jsonl reply file or raw text.  JSON is unwrapped automatically - OpenAI shape
(`choices[0].message.content` + `reasoning_content`), a bare `content`/`text`/`output` field, or one object per
line; anything else (or a shape not understood) is scored as the file's raw text.

THE METRICS (clean values in brackets are 1Cat's measured clean rows, prose/HTML long outputs):
  digit_ratio     digits / non-whitespace chars                       [clean <= 0.014; gibberish >= 0.12]
  max_digit_run   longest run of consecutive ASCII digits             [clean 4-5]
  num_repeat_runs maximal runs of ONE digit repeated >= 4 times      [clean 0]
  weird_runs      maximal runs of one 2-12 char unit repeated >= 4 times in a row, the unit containing NO
                  whitespace and not all digits (px3px3px3px3, UTF-UTF-UTF-UTF; repeated CODE LINES and
                  rgba(255,255,255, are legitimate and excluded - a unit with whitespace, or only 3 repeats,
                  is not degeneration).  5555 is counted by num_repeat_runs instead.   [clean 0-2]
  max_char_run    longest run of one character (....., aaaa)         [document, do not threshold by default]
  sha256          of the extracted text, bytes as they are - two runs of one config that differ say so here first

VERDICTS are WARN then FAIL per metric (flags below; the defaults are 1Cat's clean-vs-bad split, a STARTING point
for prose and agent replies - code replies carry more digits, raise --digit-warn there, or judge by --compare
against your own "known good" baseline run, which is the stronger use).  Exit 0 all PASS, 1 any WARN, 2 any FAIL,
so a bench harness can gate on it.

SAMPLING RULES THE METRIC CANNOT ENFORCE (1Cat's log, keep them in the harness that produces the replies):
natural EOS only - `ignore_eos`/`max_tokens` pinning past a natural stop pollutes the failure signal; and fix
temperature/seed, or the sha256 column is noise.

    # establish the baseline on today's known-good production config, then gate a candidate against it
    python tools/output_quality.py replies-baseline/*.json --json base.json
    python tools/output_quality.py replies-candidate/*.json --json cand.json
    python tools/output_quality.py --compare base.json cand.json
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path


def unwrap_json(obj):
    """One reply's text out of a JSON object, whatever shelf it sits on.  None: not a reply shape."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        ch = obj.get("choices")
        if isinstance(ch, list) and ch and isinstance(ch[0], dict):
            msg = ch[0].get("message") or {}
            parts = [msg.get("reasoning_content") or "", msg.get("content") or ""]
            return "".join(p for p in parts if isinstance(p, str)) or None
        for k in ("content", "text", "output", "response"):
            v = obj.get(k)
            if isinstance(v, str) and v:
                return v
    if isinstance(obj, list):  # a stream of chunks, already concatenated by the harness?
        parts = [unwrap_json(o) for o in obj]
        parts = [p for p in parts if p]
        return "".join(parts) or None
    return None


def read_reply(path: Path):
    """The reply's text: unwrapped JSON, or the raw file.  (None, reason) when a file has nothing scoreable."""
    raw = path.read_bytes()
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, "not UTF-8 text"
    stripped = txt.lstrip()
    if stripped[:1] in ("{", "["):
        try:
            if stripped.endswith("\n") and "\n" in stripped.strip():
                lines = [json.loads(ln) for ln in stripped.splitlines() if ln.strip()]
                got = unwrap_json(lines)
            else:
                got = unwrap_json(json.loads(stripped))
            if got:
                return got, None
        except (json.JSONDecodeError, ValueError):
            pass
    return txt, None


def score(text: str):
    """The metrics.  NFC first so a full-width digit is not counted as an exotic letter forever."""
    text = unicodedata.normalize("NFC", text)
    non_ws = re.sub(r"\s+", "", text)
    digits = sum(c.isdigit() for c in non_ws)
    m = {
        "n_chars": len(text),
        "n_nonspace": len(non_ws),
        "digit_ratio": round(digits / len(non_ws), 5) if non_ws else 0.0,
        "max_digit_run": 0,
        "num_repeat_runs": 0,
        "weird_runs": 0,
        "max_char_run": 0,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    for run in re.finditer(r"\d+", text):
        m["max_digit_run"] = max(m["max_digit_run"], run.end() - run.start())
    for run in re.finditer(r"(\d)\1{3,}", text):   # one digit 4+ times: 5555, 0000
        m["num_repeat_runs"] += 1
    run_len = 1
    for i in range(1, len(text)):
        run_len = run_len + 1 if text[i] == text[i - 1] else 1
        if run_len > m["max_char_run"]:
            m["max_char_run"] = run_len
    # one 2-12 char unit repeated >= 4 times in a row, the unit with no whitespace, not all digits, and
    # carrying at least one alphanumeric (px3px3px3px3, UTF-UTF-UTF-UTF).  Repeated code lines carry a newline,
    # CSS triples stop at 3, markdown rules are all punctuation: all three stay out.  5554 is num_repeat_runs.
    # Longest unit wins at each position, the whole run is counted once, then the scan resumes after it.
    i, n = 0, len(text)
    weird_chars = 0
    while i < n:
        hit_end, hit_len = 0, 0
        for u in range(2, 13):
            unit = text[i : i + u]
            if len(unit) < u or unit.isdigit() or any(c.isspace() for c in unit) or not any(c.isalnum() for c in unit):
                continue
            reps, j = 1, i
            while text[j + u : j + 2 * u] == unit:
                reps += 1
                j += u
            if reps >= 4 and reps * u > hit_len:
                hit_len, hit_end = reps * u, j + u
        if hit_len:
            m["weird_runs"] += 1
            weird_chars += hit_len
            i = hit_end
        else:
            i += 1
    m["weird_share"] = round(weird_chars / len(non_ws), 4) if non_ws else 0.0
    return m


def verdict(m, th):
    """PASS / WARN / FAIL from the thresholds.  Empty and near-empty replies FAIL on n_nonspace: a reply that
    said nothing is the loudest failure mode this engine has ever shipped.  digit_ratio is judged only from
    200 non-space chars: a short math answer is half digits by design, and 1Cat's gibberish was a LONG-output
    collapse."""
    if m["n_nonspace"] < 20:
        return "FAIL"
    checks = (("digit_ratio", th["digit_warn"], th["digit_fail"]),
              ("max_digit_run", th["maxrun_warn"], th["maxrun_fail"]),
              ("num_repeat_runs", th["numrun_warn"], th["numrun_fail"]),
              ("weird_runs", th["weird_warn"], th["weird_fail"]),
              ("weird_share", th["share_warn"], th["share_fail"]))
    v = "PASS"
    for key, warn, fail in checks:
        if key == "digit_ratio" and m["n_nonspace"] < 200:
            continue
        if key == "weird_share" and m["n_nonspace"] < 100:
            continue
        if m[key] >= fail:
            return "FAIL"
        if m[key] >= warn:
            v = "WARN"
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", type=Path, help="reply files (.json/.jsonl unwrapped; else raw text)")
    ap.add_argument("--json", type=Path, help="write the per-file metrics here (feeds --compare)")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "CAND"))
    ap.add_argument("--digit-warn", type=float, default=0.03)
    ap.add_argument("--digit-fail", type=float, default=0.10)
    ap.add_argument("--maxrun-warn", type=int, default=8)
    ap.add_argument("--maxrun-fail", type=int, default=16)
    ap.add_argument("--numrun-warn", type=int, default=1)
    ap.add_argument("--numrun-fail", type=int, default=3)
    ap.add_argument("--weird-warn", type=int, default=1)
    ap.add_argument("--weird-fail", type=int, default=6)
    ap.add_argument("--share-warn", type=float, default=0.08)
    ap.add_argument("--share-fail", type=float, default=0.20)
    a = ap.parse_args()
    th = {k: v for k, v in vars(a).items() if k.endswith(("_warn", "_fail"))}

    if a.compare:
        base = json.loads(a.compare[0].read_text("utf-8"))
        cand = json.loads(a.compare[1].read_text("utf-8"))
        keys = [k for k in cand if k in base]
        if not keys:
            print("output_quality: the two runs share no reply ids", file=sys.stderr)
            return 2
        worst, n_same, n_diff = 0, 0, 0
        for k in keys:
            b, c = base[k]["metrics"], cand[k]["metrics"]
            same = b["sha256"] == c["sha256"]
            n_same, n_diff = n_same + same, n_diff + (not same)
            v = cand[k]["verdict"] if not same else ("PASS" if base[k]["verdict"] == "PASS" else base[k]["verdict"])
            worst = max(worst, 0 if v == "PASS" else 1 if v == "WARN" else 2)
            mark = "same" if same else f"diff (base {base[k]['verdict']} -> cand {cand[k]['verdict']})"
            drift = "" if same else "".join(
                f" {key} {b[key]}->{c[key]}" for key in ("digit_ratio", "max_digit_run", "num_repeat_runs", "weird_runs")
                if b[key] != c[key])
            print(f"{k}: {mark}{drift}")
        print(f"output_quality: {len(keys)} shared replies, {n_same} bit-equal, {n_diff} differ")
        return worst

    if not a.files:
        ap.error("give reply files, or --compare BASE CAND")
    out, worst = {}, 0
    for p in a.files:
        text, why = read_reply(p)
        if text is None:
            print(f"{p.name}: UNSCOREABLE ({why})", file=sys.stderr)
            out[p.name] = {"verdict": "FAIL", "metrics": None}
            worst = 2
            continue
        m = score(text)
        v = verdict(m, th)
        worst = max(worst, 0 if v == "PASS" else 1 if v == "WARN" else 2)
        out[p.name] = {"verdict": v, "metrics": m}
        print(f"{p.name}: {v}  n={m['n_chars']}  digit_ratio={m['digit_ratio']}  max_digit_run={m['max_digit_run']}  "
              f"num_repeat_runs={m['num_repeat_runs']}  weird_runs={m['weird_runs']}  max_char_run={m['max_char_run']}  "
              f"sha={m['sha256'][:12]}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=1, sort_keys=True), "utf-8")
        print(f"output_quality: {len(out)} replies -> {a.json}", file=sys.stderr)
    return worst


if __name__ == "__main__":
    sys.exit(main())
