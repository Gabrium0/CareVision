"""Caregiver alerting: turn Severity.ALERT detections into real notifications.

Deterministic and independent of the voice agent / LLM — the safety loop must
never depend on a model being available.
"""
