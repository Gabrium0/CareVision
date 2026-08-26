import sys

# The Windows console defaults to cp1252, which cannot encode the Unicode
# block characters qrcode.print_ascii() emits (raises UnicodeEncodeError and
# kills the pairing window before the fallback text below can print).
# Force UTF-8 on stdout/stderr up front; errors='replace' so this can never
# itself crash the one script that must always leave a usable URL on screen.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

url = sys.argv[1] if len(sys.argv) > 1 else ""
print("Open this on the iPad:\n  " + url + "\n")
try:
    import qrcode
except ImportError:
    print("(install 'qrcode' for an on-screen QR:  python -m pip install qrcode)")
    sys.exit(0)
try:
    qr = qrcode.QRCode(border=2)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)
except Exception as exc:  # noqa: BLE001 - QR rendering is best-effort only;
    # the plain-text URL printed above is the one thing that must survive.
    print(f"(QR rendering failed: {exc}; use the URL printed above instead)")
print("\nScan with the iPad camera to open, then enter the 6-digit code shown in the app window.")
