import UIKit
import AVFoundation

/// The paired control surface: camera preview behind a live overlay of vitals,
/// demo tiles, the signal feed, the agent voice pill, and a module toggle drawer
/// — all driven by the laptop's `telemetry` and `state` control messages, the
/// same payloads the old browser page rendered.
final class LiveViewController: UIViewController, LinkClientDelegate {
    private let link: LinkClient
    private lazy var camera = CameraController(link: link)
    private let voice = VoiceController()

    private let statusPill = PaddedLabel()
    private let headline = UILabel()
    private let vitalsStack = UIStackView()
    private let demoStack = UIStackView()
    private let signalsStack = UIStackView()
    private let agentPill = PaddedLabel()
    private var latestModules: [ModuleRow] = []

    init(config: LinkClient.Config) {
        self.link = LinkClient(config: config)
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) not used") }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        buildOverlay()
        link.delegate = self

        voice.onTranscript = { [weak self] text in
            self?.link.sendControl(["type": "asr_text", "text": text])
        }
        voice.onSpeakingChange = { [weak self] speaking in
            self?.link.sendControl(["type": "speaking", "speaking": speaking])
        }

        link.start()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        camera.previewLayer.frame = view.bounds
        camera.updateOrientation(currentInterfaceOrientation)
    }

    override func viewWillTransition(to size: CGSize,
                                     with coordinator: UIViewControllerTransitionCoordinator) {
        super.viewWillTransition(to: size, with: coordinator)
        coordinator.animate(alongsideTransition: { _ in
            self.camera.previewLayer.frame = CGRect(origin: .zero, size: size)
            self.camera.updateOrientation(self.currentInterfaceOrientation)
        })
    }

    override var supportedInterfaceOrientations: UIInterfaceOrientationMask { .landscape }

    /// The current interface orientation, resilient on iPadOS 13.3 (no scene) —
    /// prefers the window scene, falls back to the app's status-bar orientation.
    private var currentInterfaceOrientation: UIInterfaceOrientation {
        if #available(iOS 13.0, *), let scene = view.window?.windowScene {
            return scene.interfaceOrientation
        }
        return UIApplication.shared.statusBarOrientation
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        if isMovingFromParent {
            link.stop(); camera.stop(); voice.stop()
        }
    }

    // MARK: - layout

    private func buildOverlay() {
        camera.previewLayer.frame = view.bounds
        view.layer.addSublayer(camera.previewLayer)

        statusPill.font = .systemFont(ofSize: 13, weight: .semibold)
        statusPill.textColor = .white
        statusPill.backgroundColor = UIColor.black.withAlphaComponent(0.55)
        statusPill.layer.cornerRadius = 12
        statusPill.clipsToBounds = true
        statusPill.text = "Connecting…"

        headline.font = .systemFont(ofSize: 22, weight: .bold)
        headline.textColor = .white
        headline.numberOfLines = 2
        headline.shadowColor = .black
        headline.shadowOffset = CGSize(width: 0, height: 1)

        let modulesButton = UIButton(type: .system)
        modulesButton.setTitle("Modules", for: .normal)
        modulesButton.setTitleColor(.white, for: .normal)
        modulesButton.titleLabel?.font = .systemFont(ofSize: 14, weight: .semibold)
        modulesButton.backgroundColor = UIColor.black.withAlphaComponent(0.55)
        modulesButton.contentEdgeInsets = UIEdgeInsets(top: 6, left: 12, bottom: 6, right: 12)
        modulesButton.layer.cornerRadius = 12
        modulesButton.addTarget(self, action: #selector(openModules), for: .touchUpInside)

        let topRow = UIStackView(arrangedSubviews: [statusPill, UIView(), modulesButton])
        topRow.alignment = .center

        vitalsStack.axis = .horizontal
        vitalsStack.distribution = .fillEqually
        vitalsStack.spacing = 10

        demoStack.axis = .horizontal
        demoStack.distribution = .fillEqually
        demoStack.spacing = 8

        signalsStack.axis = .vertical
        signalsStack.spacing = 4

        agentPill.font = .systemFont(ofSize: 15, weight: .medium)
        agentPill.textColor = .white
        agentPill.backgroundColor = UIColor.black.withAlphaComponent(0.55)
        agentPill.layer.cornerRadius = 16
        agentPill.clipsToBounds = true
        agentPill.numberOfLines = 0
        agentPill.text = "voice idle"

        let column = UIStackView(arrangedSubviews: [
            topRow, headline, vitalsStack, demoStack, signalsStack, UIView(), agentPill])
        column.axis = .vertical
        column.spacing = 12
        column.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(column)
        NSLayoutConstraint.activate([
            column.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            column.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            column.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            column.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -16)])
    }

    @objc private func openModules() {
        let vc = ModulesViewController(modules: latestModules) { [weak self] name, enable in
            self?.link.sendControl(["type": "module",
                                    "action": enable ? "enable" : "disable",
                                    "target": name, "scope": "primary"])
        }
        let nav = UINavigationController(rootViewController: vc)
        present(nav, animated: true)
    }

    // MARK: - LinkClientDelegate

    func link(_ link: LinkClient, didChangeState state: LinkState) {
        switch state {
        case .idle:         statusPill.text = "Idle"
        case .connecting:   statusPill.text = "Connecting…"
        case .disconnected: statusPill.text = "Reconnecting…"
        case .failed(let m): statusPill.text = "Error: \(m)"
        case .paired:
            statusPill.text = "Paired"
            reportClientVersion()
            camera.start { ok in
                if !ok { self.statusPill.text = "Camera unavailable" }
            }
            voice.startListening()
        }
    }

    func link(_ link: LinkClient, didReceiveControl type: String, payload: [String: Any]) {
        switch type {
        case "telemetry": applyTelemetry(Telemetry(payload))
        case "state":     applyState(payload)
        case "speak":     if let text = payload["text"] as? String { voice.speak(text) }
        case "module_result": break   // the drawer re-reads from the next telemetry
        default: break
        }
    }

    private func reportClientVersion() {
        link.sendControl([
            "type": "client_version",
            "build": "native-ios-1.0",
            "ua": "CareVisioniPad/1.0 (iPadOS 13)",
            "caps": ["native": true, "avfoundation": true, "ondevice_asr": true,
                     "ondevice_tts": true]])
    }

    // MARK: - rendering

    private func applyTelemetry(_ t: Telemetry) {
        headline.text = t.greeting ?? (t.personPresent == true ? "" : "No one in view")
        // Promote the most severe warning/alert over the greeting, like the page.
        if let worst = t.signals.first(where: { $0.severity == .alert })
            ?? t.signals.first(where: { $0.severity == .warning }) {
            headline.text = worst.message.isEmpty ? worst.label : worst.message
            headline.textColor = worst.severity.color
        } else {
            headline.textColor = .white
        }

        rebuild(vitalsStack, count: t.vitals.count, make: { VitalTileView() }) { view, i in
            (view as? VitalTileView)?.apply(t.vitals[i])
        }
        rebuild(demoStack, count: t.demoTiles.count, make: { DemoTileView() }) { view, i in
            (view as? DemoTileView)?.apply(t.demoTiles[i])
        }
        renderSignals(Array(t.signals.prefix(4)))
        latestModules = t.modules
    }

    private func applyState(_ payload: [String: Any]) {
        let agent = AgentState(payload["agent"] as? [String: Any])
        if agent.speaking {
            agentPill.text = "Ada is speaking"
        } else if agent.listening {
            agentPill.text = "listening…"
        } else {
            agentPill.text = "voice idle"
        }
        // Fallback path when the backend has not (yet) been wired to send an
        // explicit `speak`: voice a fresh agent line as it appears.
        if agent.speaking, let text = agent.text, !text.isEmpty {
            voice.speak(text)
        }
        if latestModules.isEmpty, let mods = payload["modules"] as? [[String: Any]] {
            latestModules = mods.compactMap(ModuleRow.init)
        }
    }

    private func renderSignals(_ signals: [SignalRow]) {
        signalsStack.arrangedSubviews.forEach { $0.removeFromSuperview() }
        for signal in signals where !signal.message.isEmpty {
            let label = PaddedLabel()
            label.font = .systemFont(ofSize: 13, weight: .medium)
            label.textColor = .white
            label.backgroundColor = signal.severity.color.withAlphaComponent(0.7)
            label.layer.cornerRadius = 10
            label.clipsToBounds = true
            label.numberOfLines = 2
            label.text = signal.message
            signalsStack.addArrangedSubview(label)
        }
    }

    /// Reconcile a horizontal stack to `count` reusable tile views, then apply.
    private func rebuild(_ stack: UIStackView, count: Int, make: () -> UIView,
                         apply: (UIView, Int) -> Void) {
        while stack.arrangedSubviews.count > count {
            stack.arrangedSubviews.last.map { $0.removeFromSuperview() }
        }
        while stack.arrangedSubviews.count < count {
            stack.addArrangedSubview(make())
        }
        for (i, view) in stack.arrangedSubviews.enumerated() { apply(view, i) }
        stack.isHidden = count == 0
    }
}

/// A UILabel with interior padding, used for the pills and signal chips.
final class PaddedLabel: UILabel {
    var insets = UIEdgeInsets(top: 6, left: 12, bottom: 6, right: 12)
    override func drawText(in rect: CGRect) { super.drawText(in: rect.inset(by: insets)) }
    override var intrinsicContentSize: CGSize {
        let s = super.intrinsicContentSize
        return CGSize(width: s.width + insets.left + insets.right,
                      height: s.height + insets.top + insets.bottom)
    }
}
