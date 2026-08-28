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
        secretField.text = defaults.string(forKey: "secret") ?? ""
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

        connectButton.setTitle("Connect", for: .normal)
        connectButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .semibold)
        connectButton.backgroundColor = .systemBlue
        connectButton.setTitleColor(.white, for: .normal)
        connectButton.layer.cornerRadius = 12
        connectButton.addTarget(self, action: #selector(connect), for: .touchUpInside)
        connectButton.heightAnchor.constraint(equalToConstant: 52).isActive = true

        statusLabel.font = .systemFont(ofSize: 14)
        statusLabel.textColor = .systemRed
        statusLabel.numberOfLines = 0

        let stack = UIStackView(arrangedSubviews: [
            heading, subtitle,
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
        let defaults = UserDefaults.standard
        defaults.set(host, forKey: "host")
        defaults.set(String(port), forKey: "port")
        defaults.set(room, forKey: "room")
        defaults.set(secret, forKey: "secret")

        let config = LinkClient.Config(host: host, port: port, room: room,
                                       secret: secret, code: code)
        let live = LiveViewController(config: config)
        navigationController?.pushViewController(live, animated: true)
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
