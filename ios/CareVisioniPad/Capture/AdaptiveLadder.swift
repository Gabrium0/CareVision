import Foundation
import CoreGraphics

/// One capture tier: long edge (px), JPEG quality, and target fps.
struct CaptureProfile: Equatable {
    let tier: Int
    let longEdge: CGFloat
    let quality: CGFloat
    let fps: Double
}

/// ACK-latency-driven capture controller, mirroring the browser policy in
/// `relay/static/ipad_capture_policy.js`: as the laptop's per-frame ACK latency
/// rises (or the sender window backs up) it steps resolution/quality/fps **down**,
/// and steps back **up** once things recover — with hysteresis so it doesn't
/// oscillate. A thermal floor lets `CameraController` force a minimum degradation
/// under `ProcessInfo` thermal pressure. Pure and unit-testable.
final class AdaptiveLadder {
    private let profiles: [CaptureProfile] = [
        CaptureProfile(tier: 0, longEdge: 960, quality: 0.90, fps: 20),
        CaptureProfile(tier: 1, longEdge: 800, quality: 0.85, fps: 20),
        CaptureProfile(tier: 2, longEdge: 800, quality: 0.80, fps: 15),
        CaptureProfile(tier: 3, longEdge: 640, quality: 0.80, fps: 15),
        CaptureProfile(tier: 4, longEdge: 640, quality: 0.76, fps: 12),
        CaptureProfile(tier: 5, longEdge: 640, quality: 0.72, fps: 10),
    ]
    // Thresholds in milliseconds; cooldowns in seconds. Degrade fast, recover slow.
    private let degradeMs = 180.0
    private let improveMs = 90.0
    private let degradeCooldown = 2.0
    private let improveCooldown = 6.0

    private var index = 0
    private var lastChange: TimeInterval = 0

    /// Minimum tier the thermal state pins us to (0 = no floor). Raising it
    /// degrades immediately; lowering it lets the normal recovery take over.
    var thermalFloor = 0 {
        didSet { if index < thermalFloor { index = thermalFloor } }
    }

    var current: CaptureProfile { profiles[index] }

    /// Advance the controller and return the profile to use now.
    @discardableResult
    func update(ackP90ms: Double, backpressured: Bool, now: TimeInterval) -> CaptureProfile {
        let maxTier = profiles.count - 1
        if (backpressured || ackP90ms > degradeMs), index < maxTier,
           now - lastChange > degradeCooldown {
            index += 1
            lastChange = now
        } else if ackP90ms > 0, ackP90ms < improveMs, !backpressured,
                  index > max(thermalFloor, 0), now - lastChange > improveCooldown {
            index -= 1
            lastChange = now
        }
        if index < thermalFloor { index = thermalFloor }
        return current
    }
}
