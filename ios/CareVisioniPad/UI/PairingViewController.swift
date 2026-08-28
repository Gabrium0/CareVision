import UIKit

/// First screen: enter the laptop's hotspot address, room, and the 6-digit
/// pairing code the laptop printed, then Connect. Values are remembered between
/// launches so a returning demo only re-types the code.
final class PairingViewController: UIViewController {
    private let hostField = PairingViewController.makeField("192.168.137.1")
    private let portField = PairingViewController.makeField("8788")
    private let roomField = PairingViewController.makeField("room")
    private let codeField = PairingViewController.makeField("6-digit code")
    private let secretField = PairingViewController.makeField("shared secret")
    private let statusLabel = UILabel()
    private let connectButton = UIButton(type: .system)

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        title = "CareVision"

        let defaults = UserDefaults.standard
        hostField.text = defaults.string(forKey: "host") ?? "192.168.137.1"
        portField.text = defaults.string(forKey: "port") ?? "8788"
        roomField.text = defaults.string(forKey: "room") ?? ""
        secretField.text = KeychainStore.get("secret")
            ?? defaults.string(forKey: "secret") ?? ""   // migrate any old value
        portField.keyboardType = .numberPad
        codeField.keyboardType = .numberPad
        secretField.isSecureTextEntry = true

        let heading = UILabel()
        heading.text = "Pair with the laptop"
        heading.font = .systemFont(ofSize: 26, weight: .bold)

        let subtitle = UILabel()
        subtitle.text = "The iPad must be on the laptop's Windows hotspot."
        subtitle.font = .systemFont(ofSize: 14)
        subtitle.textColor = .secondaryLabel
        subtitle.numberOfLines = 0

        let scanButton = UIButton(type: .system)
        scanButton.setTitle("📷  Scan QR to connect", for: .normal)
        scanButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .semibold)
        scanButton.backgroundColor = .systemBlue
        scanButton.setTitleColor(.white, for: .normal)
        scanButton.layer.cornerRadius = 12
        scanButton.addTarget(self, action: #selector(scan), for: .touchUpInside)
        scanButton.heightAnchor.constraint(equalToConstant: 52).isActive = true

        let orLabel = UILabel()
        orLabel.text = "or enter the details by hand"
        orLabel.font = .systemFont(ofSize: 13)
        orLabel.textColor = .tertiaryLabel
        orLabel.textAlignment = .center

        connectButton.setTitle("Connect", for: .normal)
        connectButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .semibold)
        connectButton.backgroundColor = .secondarySystemBackground
        connectButton.setTitleColor(.label, for: .normal)
        connectButton.layer.cornerRadius = 12
        connectButton.addTarget(self, action: #selector(connect), for: .touchUpInside)
        connectButton.heightAnchor.constraint(equalToConstant: 52).isActive = true

        statusLabel.font = .systemFont(ofSize: 14)
        statusLabel.textColor = .systemRed
        statusLabel.numberOfLines = 0

        let stack = UIStackView(arrangedSubviews: [
            heading, subtitle, scanButton, orLabel,
            labeled("Laptop address", hostField),
            labeled("Port", portField),
            labeled("Room", roomField),
            labeled("Shared secret", secretField),
            labeled("Pairing code", codeField),
            connectButton, statusLabel])
        stack.axis = .vertical
        stack.spacing = 14
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 32),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -32),
            stack.widthAnchor.constraint(lessThanOrEqualToConstant: 520)])
    }

    private func labeled(_ title: String, _ field: UITextField) -> UIView {
        let label = UILabel()
        label.text = title
        label.font = .systemFont(ofSize: 12, weight: .medium)
        label.textColor = .secondaryLabel
        let stack = UIStackView(arrangedSubviews: [label, field])
        stack.axis = .vertical
        stack.spacing = 4
        return stack
    }

    @objc private func connect() {
        statusLabel.text = ""
        let host = (hostField.text ?? "").trimmingCharacters(in: .whitespaces)
        let room = (roomField.text ?? "").trimmingCharacters(in: .whitespaces)
        let secret = secretField.text ?? ""
        let code = (codeField.text ?? "").trimmingCharacters(in: .whitespaces)
        guard let port = Int(portField.text ?? ""), !host.isEmpty, !room.isEmpty,
              !secret.isEmpty, code.count == 6, Int(code) != nil else {
            statusLabel.text = "Fill in every field; the code is 6 digits."
            return
        }
        launch(LinkClient.Config(host: host, port: port, room: room,
                                 secret: secret, code: code))
    }

    @objc private func scan() {
        let scanner = QRScannerViewController { [weak self] info in
            self?.launch(LinkClient.Config(host: info.host, port: info.port,
                                           room: info.room, secret: info.secret,
                                           code: info.code))
        }
        let nav = UINavigationController(rootViewController: scanner)
        nav.modalPresentationStyle = .fullScreen
        present(nav, animated: true)
    }

    private func launch(_ config: LinkClient.Config) {
        let defaults = UserDefaults.standard
        defaults.set(config.host, forKey: "host")
        defaults.set(String(config.port), forKey: "port")
        defaults.set(config.room, forKey: "room")
        defaults.removeObject(forKey: "secret")        // the secret lives in Keychain only
        KeychainStore.set(config.secret, for: "secret")
        // Reflect the values in the fields so a scanned pairing is transparent.
        hostField.text = config.host
        portField.text = String(config.port)
        roomField.text = config.room
        secretField.text = config.secret
        codeField.text = config.code
        navigationController?.pushViewController(LiveViewController(config: config), animated: true)
    }

    private static func makeField(_ placeholder: String) -> UITextField {
        let field = UITextField()
        field.placeholder = placeholder
        field.borderStyle = .roundedRect
        field.font = .systemFont(ofSize: 17)
        field.autocapitalizationType = .none
        field.autocorrectionType = .no
        return field
    }
}
