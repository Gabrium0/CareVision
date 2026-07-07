"""Pluggable notification channels for caregiver alerts.

Each channel self-disables if its credentials (from environment variables,
never yaml) are missing, mirroring the graceful-fallback contract used by the
detection backends. `console` is always available so alerts are never silently
lost.
"""
from __future__ import annotations

import os
import smtplib
from abc import ABC, abstractmethod
from email.message import EmailMessage


class Channel(ABC):
    name = "channel"
    available = True

    @abstractmethod
    def send(self, subject: str, body: str) -> bool:
        ...


class ConsoleChannel(Channel):
    name = "console"

    def send(self, subject: str, body: str) -> bool:
        print(f"\n*** CAREGIVER ALERT: {subject} ***\n{body}\n", flush=True)
        return True


class EmailChannel(Channel):
    """SMTP email. Reads SMTP_HOST/PORT/USER/PASSWORD and ALERT_EMAIL_TO."""
    name = "email"

    def __init__(self):
        self.host = os.environ.get("SMTP_HOST")
        self.port = int(os.environ.get("SMTP_PORT", "587"))
        self.user = os.environ.get("SMTP_USER")
        self.password = os.environ.get("SMTP_PASSWORD")
        self.to = os.environ.get("ALERT_EMAIL_TO")
        self.available = all([self.host, self.user, self.password, self.to])

    def send(self, subject: str, body: str) -> bool:
        if not self.available:
            return False
        try:
            msg = EmailMessage()
            msg["Subject"] = f"[Care Monitor] {subject}"
            msg["From"] = self.user
            msg["To"] = self.to
            msg.set_content(body)
            with smtplib.SMTP(self.host, self.port, timeout=15) as s:
                s.starttls()
                s.login(self.user, self.password)
                s.send_message(msg)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[alerts/email] send failed: {e}")
            return False


class WebhookChannel(Channel):
    """POST to ALERT_WEBHOOK_URL (Slack/Discord/custom). Uses requests."""
    name = "webhook"

    def __init__(self):
        self.url = os.environ.get("ALERT_WEBHOOK_URL")
        self.available = bool(self.url)

    def send(self, subject: str, body: str) -> bool:
        if not self.available:
            return False
        try:
            import requests
            r = requests.post(self.url, json={"text": f"*{subject}*\n{body}"}, timeout=15)
            return r.status_code < 400
        except Exception as e:  # noqa: BLE001
            print(f"[alerts/webhook] send failed: {e}")
            return False


class SmsChannel(Channel):
    """Twilio SMS. Reads TWILIO_SID/TWILIO_TOKEN/TWILIO_FROM and ALERT_SMS_TO."""
    name = "sms"

    def __init__(self):
        self.sid = os.environ.get("TWILIO_SID")
        self.token = os.environ.get("TWILIO_TOKEN")
        self.from_ = os.environ.get("TWILIO_FROM")
        self.to = os.environ.get("ALERT_SMS_TO")
        self._client = None
        self.available = all([self.sid, self.token, self.from_, self.to])
        if self.available:
            try:
                from twilio.rest import Client
                self._client = Client(self.sid, self.token)
            except Exception as e:  # noqa: BLE001
                print(f"[alerts/sms] twilio unavailable: {e}")
                self.available = False

    def send(self, subject: str, body: str) -> bool:
        if not self.available or self._client is None:
            return False
        try:
            self._client.messages.create(
                body=f"{subject}: {body}"[:1500], from_=self.from_, to=self.to)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[alerts/sms] send failed: {e}")
            return False


_REGISTRY = {c.name: c for c in
             [ConsoleChannel, EmailChannel, WebhookChannel, SmsChannel]}


def build_channels(names: list[str]) -> list[Channel]:
    """Instantiate the named channels; drop any whose creds are missing."""
    out = []
    for n in names:
        cls = _REGISTRY.get(n)
        if cls is None:
            print(f"[alerts] unknown channel '{n}', skipping")
            continue
        inst = cls()
        if inst.available:
            out.append(inst)
        else:
            print(f"[alerts] channel '{n}' disabled (missing credentials)")
    if not out:
        out.append(ConsoleChannel())     # never leave alerts with nowhere to go
    return out
