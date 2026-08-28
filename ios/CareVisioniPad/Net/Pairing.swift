import Foundation
import CryptoKit

/// Pairing HMAC, computed identically to the laptop (`core/ipad_link.py`) and the
/// browser page (`relay/static/ipad.html`). The native app authenticates directly
/// to the laptop's LAN WebSocket — no relay mediates — so the laptop recomputes
/// this signature against the 6-digit code it printed at startup.
///
/// Contract (must not drift from `core/ipad_link.py`):
///   * window     = 600 s (`PAIR_WINDOW_SECONDS`)
///   * exp        = ceil((now + 1) / 600) * 600     — quantised, no round trip
///   * message    = "room|code|exp"
///   * sig        = hex(HMAC_SHA256(secret, message))[0..<32]   (first 32 hex chars)
enum Pairing {
    static let windowSeconds: Double = 600

    /// Next pairing-window boundary, matching `pairing_expiry` on the laptop.
    static func expiry(now: Date = Date()) -> Int {
        let t = now.timeIntervalSince1970
        return Int((ceil((t + 1) / windowSeconds) * windowSeconds).rounded(.towardZero))
    }

    /// 32-hex-char pairing signature, matching `pairing_signature` on the laptop.
    static func signature(secret: String, room: String, code: String, exp: Int) -> String {
        let message = "\(room)|\(code)|\(exp)"
        let key = SymmetricKey(data: Data(secret.utf8))
        let mac = HMAC<SHA256>.authenticationCode(for: Data(message.utf8), using: key)
        let hex = mac.map { String(format: "%02x", $0) }.joined()
        return String(hex.prefix(32))
    }

    /// The opening `hello` frame the app sends first on the WebSocket.
    static func helloJSON(room: String, secret: String, code: String,
                          exp: Int? = nil) -> String {
        let e = exp ?? expiry()
        let sig = signature(secret: secret, room: room, code: code, exp: e)
        let obj: [String: Any] = ["type": "hello", "role": "ipad", "room": room,
                                  "exp": e, "sig": sig]
        let data = try! JSONSerialization.data(withJSONObject: obj)
        return String(data: data, encoding: .utf8)!
    }
}
