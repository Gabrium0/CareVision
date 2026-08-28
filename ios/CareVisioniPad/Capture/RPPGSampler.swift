import CoreVideo
import CoreGraphics
import Foundation

/// One rPPG colour sample to stream to the backend: the combined forehead/cheek
/// skin-ROI mean (RGB, 0–255) and pixel count at the frame's PTS.
struct RPPGSample {
    let t: Double            // frame mediaTime (PTS), shared timebase with frames
    let r: Double
    let g: Double
    let b: Double
    let n: Int
}

/// Lightweight face-geometry signals for on-screen feedback (backend stays
/// authoritative; these are never sent as detections).
struct QuickSignals {
    var facePresent: Bool = false
    var ear: Double = 0      // eye aspect ratio (blink when low)
    var mar: Double = 0      // mouth aspect ratio (yawn/talk when high)
    var yawDegrees: Double = 0
}

/// Samples forehead + cheek skin colour from the RAW BGRA pixel buffer (no JPEG
/// chroma loss) and applies the same mouth-motion talk-guard the backend uses —
/// so it hands the backend a ready `(t, R,G,B)` series that drops straight into
/// the rPPG buffers. Cheap: a few hundred pixels per ROI.
final class RPPGSampler {
    // Patch radius as a fraction of face-box width — matches the backend's
    // roi_patch(radius_frac=0.10) so the sampled skin area is comparable.
    private let radiusFrac: CGFloat = 0.10
    private let talkDelta = 0.05
    private var lastMAR: Double?

    /// Sample one frame. Returns the rPPG sample (nil if no usable skin pixels)
    /// and the quick signals. Must be called with the buffer valid (e.g. inside
    /// FaceVision's completion on the vision queue).
    func sample(pixelBuffer: CVPixelBuffer, face: FaceObservation,
                mediaTime: Double) -> (RPPGSample?, QuickSignals) {
        var signals = QuickSignals(facePresent: true)
        signals.ear = eyeAspectRatio(face)
        signals.mar = mouthAspectRatio(face)
        signals.yawDegrees = face.yaw * 180 / .pi

        // Talk-guard: rapid mouth movement drags the cheeks, so drop them for this
        // sample (forehead only), exactly like classical.py / spo2.py.
        var rois = [face.forehead, face.leftCheek, face.rightCheek]
        if let last = lastMAR, abs(signals.mar - last) > talkDelta {
            rois = [face.forehead]
        }
        lastMAR = signals.mar

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return (nil, signals)
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let radius = Int((radiusFrac * face.boxPixels.width).rounded())
        guard radius >= 1 else { return (nil, signals) }
        let ptr = base.assumingMemoryBound(to: UInt8.self)

        var sumR = 0.0, sumG = 0.0, sumB = 0.0, count = 0
        for roi in rois {
            let cx = Int(roi.x.rounded()), cy = Int(roi.y.rounded())
            let x0 = max(0, cx - radius), x1 = min(width - 1, cx + radius)
            let y0 = max(0, cy - radius), y1 = min(height - 1, cy + radius)
            if x0 > x1 || y0 > y1 { continue }
            var y = y0
            while y <= y1 {
                let row = y * bytesPerRow
                var x = x0
                while x <= x1 {
                    let p = row + x * 4          // BGRA
                    sumB += Double(ptr[p]); sumG += Double(ptr[p + 1]); sumR += Double(ptr[p + 2])
                    count += 1
                    x += 1
                }
                y += 1
            }
        }
        guard count > 0 else { return (nil, signals) }
        let sample = RPPGSample(t: mediaTime, r: sumR / Double(count),
                                g: sumG / Double(count), b: sumB / Double(count),
                                n: count)
        return (sample, signals)
    }

    func reset() { lastMAR = nil }

    // MARK: - geometry

    private func mouthAspectRatio(_ face: FaceObservation) -> Double {
        aspect(face.innerLips)
    }

    private func eyeAspectRatio(_ face: FaceObservation) -> Double {
        let l = aspect(face.leftEye), r = aspect(face.rightEye)
        let vals = [l, r].filter { $0 > 0 }
        return vals.isEmpty ? 0 : vals.reduce(0, +) / Double(vals.count)
    }

    /// Vertical extent / horizontal extent of a landmark region (0 when absent).
    private func aspect(_ points: [CGPoint]) -> Double {
        guard points.count >= 3 else { return 0 }
        let xs = points.map { $0.x }, ys = points.map { $0.y }
        let w = (xs.max()! - xs.min()!), h = (ys.max()! - ys.min()!)
        return w > 0 ? Double(h / w) : 0
    }
}
