import UIKit

/// Lightweight views over the laptop's `telemetry` and `state` control payloads.
/// Parsed leniently from `[String: Any]` because the payloads carry many optional
/// fields and the exact set evolves on the backend; the UI renders what is present.

enum Severity: String {
    case info, notice, warning, alert, ok

    init(any: Any?) {
        self = Severity(rawValue: (any as? String)?.lowercased() ?? "") ?? .info
    }

    var color: UIColor {
        switch self {
        case .alert:   return UIColor.systemRed
        case .warning: return UIColor.systemOrange
        case .notice:  return UIColor.systemYellow
        case .ok:      return UIColor.systemGreen
        case .info:    return UIColor.secondaryLabel
        }
    }
}

struct VitalTile {
    let id: String
    let label: String
    let value: String
    let unit: String
    let present: Bool
    let severity: Severity
    let tier: String?

    init?(_ d: [String: Any]) {
        guard let id = d["id"] as? String else { return nil }
        self.id = id
        self.label = d["label"] as? String ?? id
        self.value = VitalTile.stringValue(d["value"])
        self.unit = d["unit"] as? String ?? ""
        self.present = (d["present"] as? Bool) ?? true
        self.severity = Severity(any: d["severity"])
        self.tier = (d["reliability"] as? [String: Any])?["tier"] as? String
    }

    static func stringValue(_ any: Any?) -> String {
        if let s = any as? String { return s }
        if let n = any as? NSNumber { return NSNumber(value: n.doubleValue).stringValue }
        return "—"
    }
}

struct DemoTile {
    let id: String
    let label: String
    let icon: String
    let value: String
    let present: Bool
    let severity: Severity

    init?(_ d: [String: Any]) {
        guard let id = d["id"] as? String else { return nil }
        self.id = id
        self.label = d["label"] as? String ?? id
        self.icon = d["icon"] as? String ?? ""
        self.value = VitalTile.stringValue(d["value"])
        self.present = (d["present"] as? Bool) ?? true
        self.severity = Severity(any: d["severity"])
    }
}

struct SignalRow {
    let label: String
    let message: String
    let severity: Severity

    init?(_ d: [String: Any]) {
        // Skip provider-internal rows the browser also hides.
        if (d["internal"] as? Bool) == true { return nil }
        self.label = d["label"] as? String ?? d["module"] as? String ?? ""
        self.message = d["message"] as? String ?? ""
        self.severity = Severity(any: d["severity"])
    }
}

struct ModuleRow {
    let name: String
    let label: String
    let running: Bool
    let enabled: Bool
    let toggleable: Bool

    init?(_ d: [String: Any]) {
        guard let name = (d["module"] as? String) ?? (d["name"] as? String) else { return nil }
        self.name = name
        self.label = d["label"] as? String ?? name
        self.running = (d["running"] as? Bool) ?? false
        // `enabled` is a per-scope map in telemetry, a bool in the legacy `state`.
        if let scoped = d["enabled"] as? [String: Any] {
            self.enabled = (scoped["primary"] as? Bool) ?? false
        } else {
            self.enabled = (d["enabled"] as? Bool) ?? false
        }
        if let toggle = d["toggleable"] as? [String: Any] {
            self.toggleable = (toggle["primary"] as? Bool) ?? true
        } else {
            self.toggleable = (d["toggleable"] as? Bool) ?? true
        }
    }
}

struct Telemetry {
    var greeting: String?
    var personPresent: Bool?
    var vitals: [VitalTile] = []
    var demoTiles: [DemoTile] = []
    var signals: [SignalRow] = []
    var modules: [ModuleRow] = []
    var running: Int?
    var registered: Int?

    init(_ d: [String: Any]) {
        greeting = d["greeting"] as? String
        personPresent = d["person_present"] as? Bool
        vitals = (d["vitals"] as? [[String: Any]] ?? []).compactMap(VitalTile.init)
        demoTiles = (d["demo_tiles"] as? [[String: Any]] ?? []).compactMap(DemoTile.init)
        signals = (d["signals"] as? [[String: Any]] ?? []).compactMap(SignalRow.init)
        modules = (d["modules"] as? [[String: Any]] ?? []).compactMap(ModuleRow.init)
        if let counts = d["module_counts"] as? [String: Any] {
            running = (counts["running"] as? NSNumber)?.intValue
            registered = (counts["registered"] as? NSNumber)?.intValue
        }
    }
}

struct AgentState {
    var text: String?
    var speaking: Bool
    var listening: Bool

    init(_ d: [String: Any]?) {
        text = d?["text"] as? String
        speaking = (d?["speaking"] as? Bool) ?? false
        listening = (d?["listening"] as? Bool) ?? false
    }
}
