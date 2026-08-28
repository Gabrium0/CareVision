import AVFoundation
import CoreImage
import UIKit

/// AVFoundation capture that feeds the link with IPF1 JPEG frames.
///
/// The native wins over the old Safari page live here: frames come from the raw
/// camera pipeline with **precise, evenly-spaced `CMSampleBuffer` PTS timestamps**
/// (reported as clock source `avfoundation-pts`, replacing the browser's
/// `performance.now()` jitter that inflated rPPG error), and the app encodes at
/// full quality with no secure-context or tab-backgrounding constraints.
///
/// Flow control is cooperative: before spending CPU on a JPEG the controller asks
/// `link.readyForFrame`, so a stalled link sheds work instead of buffering —
/// mirroring the browser policy's window.
final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private weak var link: LinkClient?
    private let session = AVCaptureSession()
    private let sampleQueue = DispatchQueue(label: "carevision.capture")
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    /// Long edge of the encoded frame and JPEG quality — a small fixed profile for
    /// v1 (the backend's adaptive ladder can be layered on later).
    private let targetLongEdge: CGFloat = 960
    private let jpegQuality: CGFloat = 0.85
    private let targetFPS: Double = 20

    private var lastSentUptime: TimeInterval = 0
    private var framesSinceStat = 0
    private var lastStatAt: TimeInterval = 0
    private var lastReportedSize = CGSize.zero
    private weak var videoOutput: AVCaptureVideoDataOutput?
    private var currentOrientation: AVCaptureVideoOrientation = .landscapeRight

    private(set) lazy var previewLayer: AVCaptureVideoPreviewLayer = {
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        return layer
    }()

    init(link: LinkClient) {
        self.link = link
        super.init()
    }

    /// Ask for camera access, then configure and start the session. `completion`
    /// reports whether capture actually started (false = permission denied / no
    /// camera), so the UI can show a clear message instead of a black preview.
    func start(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndRun(completion)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted { self?.configureAndRun(completion) } else { completion(false) }
                }
            }
        default:
            completion(false)
        }
    }

    func stop() {
        sampleQueue.async { [weak self] in
            guard let self = self else { return }
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    private func configureAndRun(_ completion: @escaping (Bool) -> Void) {
        sampleQueue.async { [weak self] in
            guard let self = self else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .hd1280x720

            guard let device = self.frontCamera(),
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                DispatchQueue.main.async { completion(false) }
                return
            }
            self.session.addInput(input)

            let output = AVCaptureVideoDataOutput()
            output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: self.sampleQueue)
            guard self.session.canAddOutput(output) else {
                self.session.commitConfiguration()
                DispatchQueue.main.async { completion(false) }
                return
            }
            self.session.addOutput(output)
            self.videoOutput = output
            // The iPad is mounted in landscape; stream upright landscape frames so
            // the backend's pose/face geometry is not rotated 90°.
            self.applyOrientation(self.currentOrientation)
            self.session.commitConfiguration()
            self.session.startRunning()

            // Tell the laptop which timebase stamps these frames — the analogue of
            // the browser's clock_source report, but a clean hardware PTS.
            self.link?.sendControl(["type": "clock_source",
                                    "clock_source": "avfoundation-pts"])
            DispatchQueue.main.async { completion(true) }
        }
    }

    /// Sync capture + preview to the current landscape interface orientation.
    /// Call from the view controller on layout and rotation.
    func updateOrientation(_ interface: UIInterfaceOrientation) {
        let video: AVCaptureVideoOrientation
        switch interface {
        case .landscapeLeft:  video = .landscapeLeft
        case .landscapeRight: video = .landscapeRight
        case .portrait:       video = .portrait
        case .portraitUpsideDown: video = .portraitUpsideDown
        default: video = .landscapeRight
        }
        currentOrientation = video
        sampleQueue.async { [weak self] in self?.applyOrientation(video) }
        if let conn = previewLayer.connection, conn.isVideoOrientationSupported {
            conn.videoOrientation = video
        }
    }

    private func applyOrientation(_ orientation: AVCaptureVideoOrientation) {
        if let conn = videoOutput?.connection(with: .video), conn.isVideoOrientationSupported {
            conn.videoOrientation = orientation
        }
    }

    private func frontCamera() -> AVCaptureDevice? {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .front)
        return discovery.devices.first
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
            ?? AVCaptureDevice.default(for: .video)
    }

    // MARK: - frame path

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let link = link, link.readyForFrame else { return }

        // Pace to the target fps so we never out-run the analysis pipeline.
        let now = ProcessInfo.processInfo.systemUptime
        if now - lastSentUptime < (1.0 / targetFPS) { return }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let mediaTime = CMTimeGetSeconds(pts)
        guard mediaTime.isFinite else { return }

        guard let (jpeg, size) = encodeJPEG(pixelBuffer) else { return }
        lastSentUptime = now
        link.sendFrame(jpeg: jpeg, width: UInt16(size.width), height: UInt16(size.height),
                       mediaTime: mediaTime)
        reportStats(size: size, bytes: jpeg.count, now: now)
    }

    private func encodeJPEG(_ pixelBuffer: CVPixelBuffer) -> (Data, CGSize)? {
        var image = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = image.extent
        let longEdge = max(extent.width, extent.height)
        if longEdge > targetLongEdge {
            let scale = targetLongEdge / longEdge
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        let outSize = CGSize(width: image.extent.width.rounded(),
                             height: image.extent.height.rounded())
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let options: [CIImageRepresentationOption: Any] = [
            kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: jpegQuality]
        guard let data = ciContext.jpegRepresentation(of: image, colorSpace: colorSpace,
                                                       options: options) else { return nil }
        return (data, outSize)
    }

    private func reportStats(size: CGSize, bytes: Int, now: TimeInterval) {
        framesSinceStat += 1
        lastReportedSize = size
        if lastStatAt == 0 { lastStatAt = now; return }
        let dt = now - lastStatAt
        guard dt >= 1.0 else { return }
        let fps = Double(framesSinceStat) / dt
        link?.sendControl([
            "type": "sender_stats",
            "sent_fps": round(fps * 100) / 100,
            "target_fps": targetFPS,
            "width": Int(size.width),
            "height": Int(size.height),
            "jpeg_quality": Double(jpegQuality),
            "jpeg_bytes": bytes])
        framesSinceStat = 0
        lastStatAt = now
    }
}
