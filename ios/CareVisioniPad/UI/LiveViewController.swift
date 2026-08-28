import UIKit
import AVFoundation
import AudioToolbox

/// The paired control surface: camera preview behind a live overlay of vitals,
/// demo tiles, the signal feed, the agent voice pill, and a module toggle drawer
/// — all driven by the laptop's `telemetry` and `state` control messages, the
/// same payloads the old browser page rendered.
final class LiveViewController: UIViewController, LinkClientDelegate, CameraControllerDelegate {
    private let link: LinkClient
    private lazy var camera = CameraController(link: link)
    private let voice = VoiceController()
    private let framingBanner = PaddedLabel()
    private let signalsChip = PaddedLabel()
    private let alertBanner = PaddedLabel()
    private var alertActive = false

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

        // Tap the status pill to force a reconnect when the link is down.
        statusPill.isUserInteractionEnabled = true
        statusPill.addGestureRecognizer(
            UITapGestureRecognizer(target: self, action: #selector(reconnectTapped)))

        // Keep the companion screen awake, and pause/resume camera + mic around
        // backgrounding (iOS forbids camera use in the background anyway).
        UIApplication.shared.isIdleTimerDisabled = true
        NotificationCenter.default.addObserver(
            self, selector: #selector(appBackground),
            name: UIApplication.didEnterBackgroundNotification, object: nil)
        NotificationCenter.default.addObserver(
            self, selector: #selector(appForeground),
            name: UIApplication.willEnterForegroundNotification, object: nil)

        link.start()
    }

    @objc private func reconnectTapped() {
        if case .paired = link.state { return }
        link.reconnectNow()
    }

    @objc private func appBackground() {
        camera.stop(); voice.stop()
    }

    @objc private func appForeground() {
        camera.resume(); voice.startListening()
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
            UIApplication.shared.isIdleTimerDisabled = false
            NotificationCenter.default.removeObserver(self)
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

        let modulesButton = pillButton("Modules", #selector(openModules))
        let settingsButton = pillButton("⚙︎", #selector(openSettings))

        let topRow = UIStackView(arrangedSubviews: [statusPill, UIView(),
                                                    settingsButton, modulesButton])
        topRow.spacing = 8
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

        signalsChip.font = .monospacedDigitSystemFont(ofSize: 13, weight: .medium)
        signalsChip.textColor = .white
        signalsChip.backgroundColor = UIColor.black.withAlphaComponent(0.55)
        signalsChip.layer.cornerRadius = 12
        signalsChip.clipsToBounds = true
        signalsChip.text = "on-device: —"

        let bottomRow = UIStackView(arrangedSubviews: [agentPill, UIView(), signalsChip])
        bottomRow.alignment = .center

        let column = UIStackView(arrangedSubviews: [
            topRow, headline, vitalsStack, demoStack, signalsStack, UIView(), bottomRow])
        column.axis = .vertical
        column.spacing = 12
        column.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(column)
        NSLayoutConstraint.activate([
            column.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            column.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            column.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            column.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -16)])

        // Framing coaching banner, centered, shown only when framing is off.
        framingBanner.font = .systemFont(ofSize: 20, weight: .semibold)
        framingBanner.textColor = .white
        framingBanner.backgroundColor = UIColor.systemOrange.withAlphaComponent(0.9)
        framingBanner.layer.cornerRadius = 16
        framingBanner.clipsToBounds = true
        framingBanner.numberOfLines = 0
        framingBanner.textAlignment = .center
        framingBanner.isHidden = true
        framingBanner.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(framingBanner)
        NSLayoutConstraint.activate([
            framingBanner.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            framingBanner.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            framingBanner.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 40),
            framingBanner.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -40)])

        // Alert banner — dominant, full-width, red; distinct from the orange
        // framing banner. Shown only while a caregiver-alert signal is active.
        alertBanner.font = .systemFont(ofSize: 22, weight: .bold)
        alertBanner.textColor = .white
        alertBanner.backgroundColor = UIColor.systemRed
        alertBanner.numberOfLines = 0
        alertBanner.textAlignment = .center
        alertBanner.isHidden = true
        alertBanner.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(alertBanner)
        NSLayoutConstraint.activate([
            alertBanner.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            alertBanner.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            alertBanner.trailingAnchor.constraint(equalTo: view.trailingAnchor)])
    }

    private func pillButton(_ title: String, _ action: Selector) -> UIButton {
        let b = UIButton(type: .system)
        b.setTitle(title, for: .normal)
        b.setTitleColor(.white, for: .normal)
        b.titleLabel?.font = .systemFont(ofSize: 14, weight: .semibold)
        b.backgroundColor = UIColor.black.withAlphaComponent(0.55)
        b.contentEdgeInsets = UIEdgeInsets(top: 6, left: 12, bottom: 6, right: 12)
        b.layer.cornerRadius = 12
        b.addTarget(self, action: action, for: .touchUpInside)
        return b
    }

    @objc private func openSettings() {
        present(UINavigationController(rootViewController: SettingsViewController()),
                animated: true)
    }

    // MARK: - CameraControllerDelegate

    func camera(_ c: CameraController, framing: Framing) {
        switch framing {
        case .good:
            framingBanner.isHidden = true
        case .noFace:
            framingBanner.text = "No one in view"; framingBanner.isHidden = false
        case .tooFar:
            framingBanner.text = "Move a little closer"; framingBanner.isHidden = false
        case .offCenter:
            framingBanner.text = "Center yourself in view"; framingBanner.isHidden = false
        }
    }

    func camera(_ c: CameraController, signals: QuickSignals) {
        guard signals.facePresent else { signalsChip.text = "on-device: no face"; return }
        var tags: [String] = []
        if signals.ear > 0 && signals.ear < 0.18 { tags.append("blink") }
        if signals.mar > 0.55 { tags.append("yawn") }
        let head = String(format: "yaw %.0f°", signals.yawDegrees)
        signalsChip.text = "on-device: " + (tags.isEmpty ? head : tags.joined(separator: " · ") + " · " + head)
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
            camera.delegate = self
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
                     "ondevice_tts": true, "ondevice_rppg": true]])
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
        updateAlert(t.signals.first { $0.severity == .alert })
        latestModules = t.modules
    }

    /// Surface a caregiver alert prominently: a red banner that persists while the
    /// condition holds, plus a distinct chime on the rising edge. (No haptics —
    /// iPads have no Taptic Engine.)
    private func updateAlert(_ signal: SignalRow?) {
        if let s = signal {
            alertBanner.text = "⚠︎  " + (s.message.isEmpty ? s.label : s.message)
            alertBanner.isHidden = false
            if !alertActive {
                alertActive = true
                AudioServicesPlaySystemSound(1005)
            }
        } else {
            alertBanner.isHidden = true
            alertActive = false
        }
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
