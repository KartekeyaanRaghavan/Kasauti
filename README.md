# Kasauti (कसौटी) — a regression gate for Indic speech models

Everyone is building the next TTS / STT model. Nobody can tell whether Tuesday's deploy broke Tamil numbers.

**Kasauti** replays the same hard, real-world Indic audio through your *old* and *new* speech stack and blocks the release if anything got worse — in the browser or as a CI step.

## How it works

```
text ─► Bulbul TTS ─► + noise @ SNR (fixed seed) ─┬─► Saaras baseline ─► WER · CER · LLM judge ─┐
                                                   └─► Saaras candidate ─► WER · CER · LLM judge ─┴─► per-category gate ─► PASS / FAIL
```

- **No recording needed.** Test audio is generated with Sarvam **Bulbul**, then degraded with noise at a chosen SNR (office / street / crowded bus).
- **Same-audio A/B.** Both STT models hear byte-identical audio, so a delta is the model, not the input.
- **Hard cases that matter in India:** Hinglish / Kanglish code-mix, amounts and OTPs, place and drug names, native scripts in 6 languages.
- **Meaning, not just WER.** A Sarvam LLM judge scores whether the transcript preserved meaning (`₹2,450` vs `2450 rupees` is fine; a wrong digit is not).
- **CI-ready.** `harness.py` exits 1 on regression; `.github/workflows/kasauti.yml` runs it on every PR.

Sarvam models used: **Bulbul** (TTS), **Saaras v3 / v4** (STT), **Sarvam LLM** (judge, `sarvam-m` with `sarvam-105b` fallback).

## Run

```bash
# web dashboard: deploy to Vercel, set SARVAM_API_KEY as an env var
# CLI / CI (stdlib only, no pip install):
export SARVAM_API_KEY=...            # never commit it
python harness.py --local --base saaras:v3 --cand saaras:v4 --snr 10 --threshold 0.05
python harness.py --url https://<deployment> --snr 5
```

Add your own real recordings or sentences: extend `SUITE` in `api/run.py`, or use “Try your own sentence” in the UI.

Built at Sarvam Campus '26, IIT Hyderabad.
