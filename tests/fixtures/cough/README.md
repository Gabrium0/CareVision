# Coswara cough benchmark fixtures

These 24 short WAV files form an offline cough-versus-non-cough integration
benchmark. They were deterministically selected from quality-rated Coswara
recordings, converted to mono 16 kHz PCM16, trimmed at the leading/trailing
silence boundary, and capped at four seconds without gain adjustment.

`ATTRIBUTION.json` records the source rows, immutable source revisions,
transform, citation, and CC BY 4.0 license. It intentionally excludes health,
demographic, transcript, diagnosis, and raw participant metadata.

Run the benchmark after installing `requirements-audio-events.txt`:

```powershell
pytest -s tests/cough_audio_integration_test.py
```
