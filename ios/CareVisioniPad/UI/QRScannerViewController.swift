import UIKit
import AVFoundation

/// Scans the `carevision://pair` QR the laptop prints and returns a `PairingInfo`.
/// Uses `AVCaptureMetadataOutput` with `.qr` — the same output family the live
/// screen uses for face presence.
final class QRScannerViewController: UIViewController, AVCaptureMetadataOutputObjectsDelegate {
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "carevision.qr")
    private let onScan: (PairingInfo) -> Void
    private var handled = false
    private let hint = UILabel()

    init(onScan: @escaping (PairingInfo) -> Void) {
        self.onScan = onScan
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) not used") }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        title = "Scan to connect"
        navigationItem.leftBarButtonItem = UIBarButtonItem(
            barButtonSystemItem: .cancel, target: self, action: #selector(cancel))

        hint.text = "Point the camera at the QR shown in the laptop window"
        hint.textColor = .white
        hint.textAlignment = .center
        hint.numberOfLines = 0
        hint.font = .systemFont(ofSize: 16, weight: .medium)
        hint.translatesAutoresizingMaskIntoConstraints = false

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: configure()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
                DispatchQueue.main.async { ok ? self?.configure() : self?.showDenied() }
            }
        default: showDenied()
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        (view.layer.sublayers?.first { $0 is AVCaptureVideoPreviewLayer })?.frame = view.bounds
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sessionQueue.async { [weak self] in
            if self?.session.isRunning == true { self?.session.stopRunning() }
        }
    }

    private func configure() {
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else { showDenied(); return }
        session.addInput(input)
        let output = AVCaptureMetadataOutput()
        guard session.canAddOutput(output) else { showDenied(); return }
        session.addOutput(output)
        output.setMetadataObjectsDelegate(self, queue: .main)
        output.metadataObjectTypes = output.availableMetadataObjectTypes.contains(.qr) ? [.qr] : []

        let preview = AVCaptureVideoPreviewLayer(session: session)
        preview.videoGravity = .resizeAspectFill
        preview.frame = view.bounds
        view.layer.addSublayer(preview)

        view.addSubview(hint)
        NSLayoutConstraint.activate([
            hint.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 40),
            hint.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -40),
            hint.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -40)])

        sessionQueue.async { [weak self] in self?.session.startRunning() }
    }

    func metadataOutput(_ output: AVCaptureMetadataOutput,
                        didOutput objects: [AVMetadataObject],
                        from connection: AVCaptureConnection) {
        guard !handled,
              let qr = objects.compactMap({ $0 as? AVMetadataMachineReadableCodeObject }).first,
              qr.type == .qr, let string = qr.stringValue,
              let info = PairingURL.parse(string) else { return }
        handled = true
        sessionQueue.async { [weak self] in self?.session.stopRunning() }
        dismiss(animated: true) { self.onScan(info) }
    }

    private func showDenied() {
        hint.text = "Camera access is needed to scan. Enable it in Settings, or enter the details by hand."
        if hint.superview == nil {
            view.addSubview(hint)
            NSLayoutConstraint.activate([
                hint.centerYAnchor.constraint(equalTo: view.centerYAnchor),
                hint.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 40),
                hint.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -40)])
        }
    }

    @objc private func cancel() { dismiss(animated: true) }
}
