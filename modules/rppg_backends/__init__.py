"""Pluggable rPPG backends for the heart_rate module.

Each backend consumes the same per-frame face data and returns a common
reading dict, so the heart_rate module can run one or several at once and
show their numbers side by side.
"""
