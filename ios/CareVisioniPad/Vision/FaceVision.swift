import Vision
import CoreVideo
import UIKit

/// One face observation from Apple Vision: the bounding box plus the landmark
/// regions we need for rPPG ROIs and the quick signals. Chosen over MediaPipe
/// FaceMesh because `VNDetectFaceLandmarksRequest` is hardware-accelerated and
/// low-latency on the A10 (iPad 6th gen) — a 478-point mesh every frame would
/// tax it and risk the very lag we want to avoid. The tradeoff is Vision's
/// ~76-point 2D topology (no iris/dense mesh), so ROIs are *derived* from its
/// regions rather than index-matched to the backend's MediaPipe indices.
struct FaceObservation {
    /// Bounding box in image pixel coordinates, origin TOP-left.
    let boxPixels: CGRect
    /// Landmark regions, points already in image pixels, origin TOP-left.
    let forehead: CGPoint          // derived (Vision has no forehead landmarks)
    let leftCheek: CGPoint
    let rightCheek: CGPoint
    let leftEye: [CGPoint]
    let rightEye: [CGPoint]
    let innerLips: [CGPoint]
    let yaw: Double                 // radians, 0 = facing camera
    let roll: Double
}

/// Runs the landmark request on a dedicated queue, one request in flight, so the
/// capture path is never blocked (the core lag-avoidance measure). Caller
/// throttles the cadence.
final class FaceVision {
    private let queue = DispatchQueue(label: "carevision.vision", qos: .userInitiated)
    private let stateLock = NSLock()
    private var inFlight = false

    var isBusy: Bool { stateLock.lock(); defer { stateLock.unlock() }; return inFlight }

    /// Detect on `pixelBuffer` (retained for the duration). `completion` runs on
    /// the vision queue with the buffer still valid, so the caller can sample ROI
    /// pixels from the same frame the landmarks came from. Drops the request if
    /// one is already in flight, so a slow A10 pass never queues a backlog.
    func detect(pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation,
                completion: @escaping (CVPixelBuffer, FaceObservation?) -> Void) {
        stateLock.lock()
        if inFlight { stateLock.unlock(); return }
        inFlight = true
        stateLock.unlock()
        queue.async { [weak self] in
            guard let self = self else { return }
            defer { self.stateLock.lock(); self.inFlight = false; self.stateLock.unlock() }
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            let imageSize = CGSize(width: width, height: height)
            let request = VNDetectFaceLandmarksRequest()
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                orientation: orientation, options: [:])
            try? handler.perform([request])
            let obs = (request.results as? [VNFaceObservation])?
                .max(by: { $0.boundingBox.width < $1.boundingBox.width })
            completion(pixelBuffer, obs.flatMap { Self.build($0, imageSize: imageSize) })
        }
    }

    // MARK: - conversion

    /// Convert a Vision observation (normalized, origin bottom-left) into image
    /// pixel coordinates with origin top-left, and derive forehead/cheek anchors.
    private static func build(_ obs: VNFaceObservation, imageSize: CGSize) -> FaceObservation? {
        let w = imageSize.width, h = imageSize.height
        // boundingBox: normalized, origin bottom-left → pixels, origin top-left.
        let bb = obs.boundingBox
        let box = CGRect(x: bb.origin.x * w,
                         y: (1 - bb.origin.y - bb.height) * h,
                         width: bb.width * w, height: bb.height * h)

        func pts(_ region: VNFaceLandmarkRegion2D?) -> [CGPoint] {
            guard let region = region else { return [] }
            return region.pointsInImage(imageSize: imageSize).map {
                CGPoint(x: $0.x, y: h - $0.y)          // flip y to top-left origin
            }
        }
        let landmarks = obs.landmarks
        let leftEye = pts(landmarks?.leftEye)
        let rightEye = pts(landmarks?.rightEye)
        let innerLips = pts(landmarks?.innerLips)
        let leftBrow = pts(landmarks?.leftEyebrow)
        let rightBrow = pts(landmarks?.rightEyebrow)
        let nose = pts(landmarks?.nose)

        // Forehead: no Vision landmarks exist above the eyebrows, so anchor it a
        // little below the top of the face box on the median line — the same skin
        // region the backend samples at MediaPipe index 10.
        let foreheadX = box.midX
        let browY = (leftBrow + rightBrow).map { $0.y }.min() ?? box.minY
        let foreheadY = max(box.minY + 0.08 * box.height,
                            browY - 0.18 * box.height)
        // Cheeks: below the eyes, between nose and face edge. Approximated from the
        // box + nose tip (backend indices 50/280).
        let noseTipY = nose.map { $0.y }.max() ?? box.midY
        let cheekY = min(box.maxY - 0.15 * box.height, noseTipY)
        let leftCheek = CGPoint(x: box.minX + 0.22 * box.width, y: cheekY)
        let rightCheek = CGPoint(x: box.minX + 0.78 * box.width, y: cheekY)

        return FaceObservation(
            boxPixels: box,
            forehead: CGPoint(x: foreheadX, y: foreheadY),
            leftCheek: leftCheek, rightCheek: rightCheek,
            leftEye: leftEye, rightEye: rightEye, innerLips: innerLips,
            yaw: obs.yaw?.doubleValue ?? 0, roll: obs.roll?.doubleValue ?? 0)
    }
}
