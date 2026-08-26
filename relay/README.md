# iPad camera relay

## What this is (and is not)

An iPad in Safari can only reach `navigator.mediaDevices` — and therefore
`getUserMedia` — from a secure (HTTPS) context. The laptop running the CV
pipeline has no TLS, and the WiFi network isolates AP clients from each
other so the iPad can't reach the laptop's LAN IP directly anyway.

This service exists for exactly one reason: **to be an HTTPS origin** so the
iPad's browser will expose the camera API at all, plus a WebSocket signaling
channel that rides along for free on the same connection.

**It never sees video.** Once the iPad and laptop finish WebRTC signaling
(exchanging SDP offer/answer and ICE candidates through `/ws`), the actual
frame data flows peer-to-peer over a WebRTC data channel — directly between
the two devices on the shared Windows Mobile Hotspot, not through this
relay. `server.py` enforces this structurally, not just by convention:

- `web.WebSocketResponse(max_msg_size=64*1024)` — a hard ceiling that makes
  relaying a video frame physically impossible over this socket.
- Any binary WebSocket frame closes the connection immediately
  (`close(code=1003)`) — media can never accidentally route through here.
- The relay never parses, stores, or logs a message body. It logs only room
  id, role, and connect/disconnect. No database, no files written.

What the relay *does* see is the SDP the two sides exchange to set up that
peer-to-peer link. SDP contains local ICE candidates (private addresses like
`192.168.137.x` on the hotspot) and DTLS fingerprints — not media, and not
any secret. Anyone who could read this process's WebSocket traffic learns
what LAN addresses your two devices have and each side's DTLS certificate
fingerprint; they cannot decrypt or intercept the actual video, which never
transits this process. This service does not run TLS termination itself —
Render terminates HTTPS/WSS in front of it — so that traffic is encrypted
in transit regardless.

## Auth design

`GET /r/{room}` serves `static/ipad.html` with the shared `RELAY_SECRET`
templated directly into the page (a `const RELAY_SECRET = "...";` line,
substituted server-side for a placeholder in the HTML). This was the
simpler of the two contract-approved options, and it's safe here for a
specific reason: **knowing `RELAY_SECRET` alone is not enough to join a
room.** A valid `sig` also requires the 6-digit pairing code, which the
laptop prints at startup and a human reads and types into the iPad
out-of-band. The relay process itself never receives or stores that code —
it only ever sees the resulting `sig`, and it authenticates a room by
checking that **both** the `host` and `ipad` sockets presented the
identical `sig` for the identical, still-unexpired `exp`
(`hmac.compare_digest`, close code `4401` on mismatch or expiry). That's
sufficient proof both sides know the same secret + code without the relay
ever learning the code itself.

One detail this pushes onto the client: the laptop and the iPad compute
`exp` independently, with no round trip, so they need to land on the same
value. `ipad.html` quantizes `exp` to the next boundary of a fixed
600-second wall-clock window (`Math.ceil(now / 600) * 600`) rather than
"now plus a TTL" — any two clients that connect within the same 10-minute
window arrive at the identical `exp` without coordinating. The laptop-side
signaling code (outside this directory) must use the same quantization for
pairing to succeed.

A second connection for a role that's already occupied in a room (e.g. a
stray reload adding a second `ipad` socket) is **rejected**, not swapped in
— the existing connection is left alone and the newcomer is closed with
code `4409`. This avoids silently kicking a working session out from under
whichever side is mid-stream.

## Deploying

1. Push this repo (or just this `relay/` directory as its own repo/root —
   `render.yaml` sets `rootDir: relay`) to a Git remote Render can see.
2. In the Render dashboard, create a new Blueprint from `relay/render.yaml`,
   or a plain Web Service pointed at this directory with build command
   `pip install -r requirements.txt` and start command `python server.py`.
3. Set `RELAY_SECRET` manually in the service's Environment tab — it is
   declared `sync: false` in `render.yaml` specifically so it is never
   committed to the repo. Use a long random value
   (e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
   and put the same value in the laptop's `ipad_secret` config.
4. Confirm `GET https://<service>.onrender.com/healthz` returns `ok`.

## Cold start

Render's free tier spins the service down after a period of no traffic;
the next request pays a cold-start penalty (roughly 50 seconds). `/healthz`
is the intended warm-up target for a ping. In normal operation this mostly
doesn't matter: **start the laptop before the iPad, always** — the laptop
holds its WebSocket connection to `/ws` open for the whole session, which
keeps the service warm, so by the time a human opens the iPad page and
types the pairing code, the relay is already awake and the room is already
half-formed (the `host` socket is sitting there waiting for `ipad` to show
up). Opening the iPad page first, against a cold service, means the first
load may hang for the better part of a minute.

## Using it

Bookmark `https://<service>.onrender.com/r/<room>` on the iPad — pick any
`room` string that matches what the laptop is configured to use. Loading it
shows a pairing-code screen; type the 6-digit code the laptop printed on
startup and tap Connect.

## Fields to keep in sync with the rest of the repo

`core/ipad_camera.py` defines the binary frame format this page's JS must
match byte-for-byte:

```
struct.Struct("<4sIdHHHH")   # 24 bytes total
  offset 0:  4s  magic "IPF1"
  offset 4:  I   seq (uint32, wraps)
  offset 8:  d   mediaTime (float64 seconds)
  offset 16: H   width (uint16)
  offset 18: H   height (uint16)
  offset 20: H   flags (uint16, currently unused)
  offset 22: H   reserved (uint16, currently unused)
```

If that struct's layout ever changes in `core/ipad_camera.py`, the
`DataView` offsets in `static/ipad.html`'s `sendFrame()` need to change to
match, in the same commit.
