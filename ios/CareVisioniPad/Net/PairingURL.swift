import Foundation

/// The five things needed to pair, parsed from a scanned QR.
struct PairingInfo {
    let host: String
    let port: Int
    let room: String
    let code: String
    let secret: String
}

/// The `carevision://pair?h=&p=&r=&c=&s=` URL the laptop prints as a QR
/// (`build_lan_pairing_url` in `main.py`). Kept in lockstep with that builder.
enum PairingURL {
    static func parse(_ string: String) -> PairingInfo? {
        guard let comps = URLComponents(string: string),
              comps.scheme == "carevision", comps.host == "pair" else { return nil }
        var q: [String: String] = [:]
        for item in comps.queryItems ?? [] { q[item.name] = item.value ?? "" }
        guard let host = q["h"], !host.isEmpty,
              let portString = q["p"], let port = Int(portString), port > 0,
              let room = q["r"], !room.isEmpty,
              let code = q["c"], code.count == 6, Int(code) != nil,
              let secret = q["s"], !secret.isEmpty else { return nil }
        return PairingInfo(host: host, port: port, room: room, code: code, secret: secret)
    }

    static func build(host: String, port: Int, room: String, code: String,
                      secret: String) -> String {
        var c = URLComponents()
        c.scheme = "carevision"
        c.host = "pair"
        c.queryItems = [URLQueryItem(name: "h", value: host),
                        URLQueryItem(name: "p", value: String(port)),
                        URLQueryItem(name: "r", value: room),
                        URLQueryItem(name: "c", value: code),
                        URLQueryItem(name: "s", value: secret)]
        return c.string ?? ""
    }
}
