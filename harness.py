#!/usr/bin/env python3
"""Kasauti CI gate.

Runs the full suite (baseline STT vs candidate STT, same audio) and exits 1 if the
candidate regresses. Drop it into any CI pipeline — see .github/workflows/kasauti.yml.

  python harness.py --url https://<your-deployment>.vercel.app --snr 10 --threshold 0.05
  python harness.py --local            # call Sarvam directly (needs SARVAM_API_KEY)
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def remote_runner(url, args):
    def run(case):
        q = urllib.parse.urlencode({"id": case["id"], "tts": args.tts, "base": args.base,
                                    "cand": args.cand, "snr": args.snr or "clean", "audio": 0})
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/run?{q}", timeout=120) as r:
            return json.loads(r.read().decode())
    with urllib.request.urlopen(f"{url.rstrip('/')}/api/run?suite=1", timeout=30) as r:
        suite = json.loads(r.read().decode())["suite"]
    return suite, run


def local_runner(args):
    sys.path.insert(0, str(Path(__file__).parent / "api"))
    import run as core  # noqa: E402
    snr = float(args.snr) if args.snr else None
    return core.SUITE, lambda c: core.run_case(c, args.tts, args.base, args.cand, snr, want_audio=False)


def gate(results, threshold):
    by_cat = defaultdict(lambda: {"b_wer": [], "c_wer": [], "b_pass": 0, "c_pass": 0, "n": 0})
    for r in results:
        if r.get("error") or r["baseline"].get("error") or r["candidate"].get("error"):
            continue
        s = by_cat[r["case"]["cat"]]
        s["n"] += 1
        s["b_wer"].append(r["baseline"]["wer"])
        s["c_wer"].append(r["candidate"]["wer"])
        s["b_pass"] += bool(r["baseline"]["judge"].get("pass"))
        s["c_pass"] += bool(r["candidate"]["judge"].get("pass"))
    rows, failed = [], False
    for cat, s in sorted(by_cat.items()):
        bw = sum(s["b_wer"]) / s["n"]
        cw = sum(s["c_wer"]) / s["n"]
        regress = (cw - bw) > threshold or s["c_pass"] < s["b_pass"]
        failed |= regress
        rows.append((cat, s["n"], bw, cw, s["b_pass"], s["c_pass"], "REGRESSION" if regress else "ok"))
    return rows, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="deployed Kasauti URL")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--tts", default="bulbul:v3")
    ap.add_argument("--base", default="saaras:v3")
    ap.add_argument("--cand", default="saaras:v4")
    ap.add_argument("--snr", default="10", help="noise SNR in dB, or '' for clean audio")
    ap.add_argument("--threshold", type=float, default=0.05, help="max allowed WER increase per category")
    ap.add_argument("--out", default="kasauti_report.json")
    args = ap.parse_args()

    suite, run = local_runner(args) if args.local else remote_runner(args.url, args)
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(run, suite))
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))

    rows, failed = gate(results, args.threshold)
    print(f"\nKasauti  |  baseline={args.base}  candidate={args.cand}  tts={args.tts}  snr={args.snr or 'clean'}dB\n")
    print(f"{'category':<10}{'n':>3}{'WER base':>10}{'WER cand':>10}{'judge base':>12}{'judge cand':>12}  verdict")
    for cat, n, bw, cw, bp, cp, v in rows:
        print(f"{cat:<10}{n:>3}{bw:>10.3f}{cw:>10.3f}{bp:>9}/{n:<2}{cp:>9}/{n:<2}  {v}")
    errs = [r for r in results if r.get("error")]
    if errs:
        print(f"\n{len(errs)} case(s) errored, see {args.out}")
    print("\nGATE:", "FAIL — candidate regressed, blocking release" if failed else "PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
