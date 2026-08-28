import AVFoundation
import CoreImage
import UIKit
import Vision

/// How well the person is framed, for on-screen coaching.
enum Framing { case noFace, tooFar, offCenter, good }

protocol CameraControllerDelegate: AnyObject {
    func camera(_ c: CameraController, framing: Framing)
    func camera(_ c: CameraController, signals: QuickSignals)
}

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
final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate,
                              AVCaptureMetadataOutputObjectsDelegate {
    weak var delegate: CameraControllerDelegate?
    private weak var link: LinkClient?
    private let session = AVCaptureSession()
    private let sampleQueue = DispatchQueue(label: "carevision.capture")
    private let metadataQueue = DispatchQueue(label: "carevision.metadata")
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // On-device perception. Vision landmarks are throttled well below the capture
    // rate and run one-at-a-time on their own queue so the A10 never stalls the
    // capture path — the core anti-lag measure.
    private let faceVision = FaceVision()
    private let rppgSampler = RPPGSampler()
    private let visionInterval: TimeInterval = 1.0 / 12.0
    private var lastVisionAt: TimeInterval = 0
    private var lastFramingSent: Framing = .good

    // Capture profile, driven by the ACK-latency ladder + thermal state.
    private let ladder = AdaptiveLadder()
    private var targetLongEdge: CGFloat = 960
    private var jpegQuality: CGFloat = 0.90
    private var targetFPS: Double = 20
    private var lastSentTier = -1
    private var visionThermalPause = false
    private var configured = false

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

    /// Resume the already-configured session (e.g. returning from background).
    /// A no-op before the first successful `start`.
    func resume() {
        sampleQueue.async { [weak self] in
            guard let self = self, self.configured, !self.session.isRunning else { return }
            self.session.startRunning()
        }
    }

    deinit { NotificationCenter.default.removeObserver(self) }

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

            // Near-free hardware face detection for always-on presence/framing
            // coaching (separate from the throttled Vision landmark pass).
            let metadata = AVCaptureMetadataOutput()
            if self.session.canAddOutput(metadata) {
                self.session.addOutput(metadata)
                metadata.setMetadataObjectsDelegate(self, queue: self.metadataQueue)
                if metadata.availableMetadataObjectTypes.contains(.face) {
                    metadata.metadataObjectTypes = [.face]
                }
            }
            self.session.commitConfiguration()
            self.session.startRunning()
            self.configured = true

            // Tell the laptop which timebase stamps these frames — the analogue of
            // the browser's clock_source report, but a clean hardware PTS.
            self.link?.sendControl(["type": "clock_source",
                                    "clock_source": "avfoundation-pts"])
            DispatchQueue.main.async {
                self.startThermalMonitoring()
                completion(true)
            }
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
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let mediaTime = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        guard mediaTime.isFinite else { return }

        runVisionIfDue(pixelBuffer: pixelBuffer, mediaTime: mediaTime)

        // Frame send: paced and gated by the link's windowed-ACK flow control.
        guard let link = link, link.readyForFrame else { return }
        let now = ProcessInfo.processInfo.systemUptime
        if now - lastSentUptime < (1.0 / targetFPS) { return }
        guard let (jpeg, size) = encodeJPEG(pixelBuffer) else { return }
        lastSentUptime = now
        link.sendFrame(jpeg: jpeg, width: UInt16(size.width), height: UInt16(size.height),
                       mediaTime: mediaTime)
        reportStats(size: size, bytes: jpeg.count, now: now)
    }

    /// Dispatch a throttled, one-in-flight Vision landmark pass; on a hit, sample
    /// rPPG ROI colour from the same buffer and stream it, and publish quick
    /// signals. Runs off the capture queue so it never stalls frame delivery.
    private func runVisionIfDue(pixelBuffer: CVPixelBuffer, mediaTime: Double) {
        let now = ProcessInfo.processInfo.systemUptime
        guard !visionThermalPause else { return }   // shed Vision under thermal pressure
        guard now - lastVisionAt >= visionInterval, !faceVision.isBusy else { return }
        lastVisionAt = now
        faceVision.detect(pixelBuffer: pixelBuffer,
                          orientation: visionOrientation) { [weak self] buffer, obs in
            guard let self = self else { return }
            guard let obs = obs else { return }
            let (sample, signals) = self.rppgSampler.sample(
                pixelBuffer: buffer, face: obs, mediaTime: mediaTime)
            if let s = sample {
                self.link?.sendControl(["type": "rppg_samples",
                                        "s": [[s.t, s.r, s.g, s.b, Double(s.n)]]])
            }
            DispatchQueue.main.async { self.delegate?.camera(self, signals: signals) }
        }
    }

    // MARK: - presence / framing (metadata face detection)

    func metadataOutput(_ output: AVCaptureMetadataOutput,
                        didOutput objects: [AVMetadataObject],
                        from connection: AVCaptureConnection) {
        let faces = objects.compactMap { $0 as? AVMetadataFaceObject }
        let framing: Framing
        if let face = faces.max(by: { $0.bounds.width < $1.bounds.width }) {
            let b = face.bounds                     // normalized in the output space
            if b.width < 0.16 {
                framing = .tooFar
            } else if b.midX < 0.25 || b.midX > 0.75 {
                framing = .offCenter
            } else {
                framing = .good
            }
        } else {
            framing = .noFace
        }
        guard framing != lastFramingSent else { return }
        lastFramingSent = framing
        DispatchQueue.main.async { self.delegate?.camera(self, framing: framing) }
    }

    /// Vision orientation for the delivered buffer. The output connection is set
    /// to a landscape `videoOrientation`, so buffers arrive upright and `.up`
    /// matches; front-camera mirroring is symmetric for our colour/aspect uses.
    /// (If landmark detection ever fails on-device, this is the knob to tune.)
    private var visionOrientation: CGImagePropertyOrientation { .up }

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

        // Adapt the capture profile to the link's ACK latency / backpressure.
        let flow: (ackP90Ms: Double, backpressured: Bool) =
            link?.drainFlowStats() ?? (ackP90Ms: 0, backpressured: false)
        let profile = ladder.update(ackP90ms: flow.ackP90Ms,
                                    backpressured: flow.backpressured, now: now)
        applyProfile(profile)

        let stats: [String: Any] = [
            "type": "sender_stats",
            "sent_fps": (fps * 100).rounded() / 100,
            "target_fps": targetFPS,
            "width": Int(size.width),
            "height": Int(size.height),
            "jpeg_quality": Double(jpegQuality),
            "jpeg_bytes": bytes,
            "tier": profile.tier,
            "quality_tier": profile.tier,
            "ack_p90_ms": (flow.ackP90Ms * 100).rounded() / 100,
        ]
        link?.sendControl(stats)
        framesSinceStat = 0
        lastStatAt = now
    }

    /// Adopt a capture tier; on a change, announce it as a `capture_profile` so the
    /// backend resets rPPG buffers before the first frame of the new profile.
    private func applyProfile(_ p: CaptureProfile) {
        targetLongEdge = p.longEdge
        jpegQuality = p.quality
        targetFPS = p.fps
        guard p.tier != lastSentTier else { return }
        lastSentTier = p.tier
        let h = Int((p.longEdge * 0.75).rounded())
        link?.sendControl([
            "type": "capture_profile",
            "width": Int(p.longEdge), "height": h,
            "tier": p.tier, "quality_tier": p.tier,
            "jpeg_quality": Double(p.quality)])
    }

    // MARK: - thermal

    private func startThermalMonitoring() {
        NotificationCenter.default.addObserver(
            self, selector: #selector(thermalChanged),
            name: ProcessInfo.thermalStateDidChangeNotification, object: nil)
        thermalChanged()
    }

    @objc private func thermalChanged() {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:  ladder.thermalFloor = 0; visionThermalPause = false
        case .fair:     ladder.thermalFloor = 1; visionThermalPause = false
        case .serious:  ladder.thermalFloor = 3; visionThermalPause = true
        case .critical: ladder.thermalFloor = 4; visionThermalPause = true
        @unknown default: ladder.thermalFloor = 1; visionThermalPause = false
        }
    }
}
