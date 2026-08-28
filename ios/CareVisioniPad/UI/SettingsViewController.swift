import UIKit
import AVFoundation

/// Persisted on-device audio settings, applied by `VoiceController.speak`.
enum SettingsStore {
    private static let d = UserDefaults.standard

    static var muted: Bool {
        get { d.bool(forKey: "tts.muted") }
        set { d.set(newValue, forKey: "tts.muted") }
    }
    static var rate: Float {
        get { d.object(forKey: "tts.rate") == nil ? AVSpeechUtteranceDefaultSpeechRate
                                                  : d.float(forKey: "tts.rate") }
        set { d.set(newValue, forKey: "tts.rate") }
    }
    static var volume: Float {
        get { d.object(forKey: "tts.volume") == nil ? 1.0 : d.float(forKey: "tts.volume") }
        set { d.set(newValue, forKey: "tts.volume") }
    }
    static var voiceId: String? {
        get { d.string(forKey: "tts.voice") }
        set { d.set(newValue, forKey: "tts.voice") }
    }
    static var voice: AVSpeechSynthesisVoice? {
        voiceId.flatMap { AVSpeechSynthesisVoice(identifier: $0) }
    }
}

/// A cell with a title and a slider.
private final class SliderCell: UITableViewCell {
    let slider = UISlider()
    let title = UILabel()
    var onChange: ((Float) -> Void)?

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        selectionStyle = .none
        title.font = .systemFont(ofSize: 15)
        title.setContentHuggingPriority(.required, for: .horizontal)
        slider.addTarget(self, action: #selector(changed), for: .valueChanged)
        let stack = UIStackView(arrangedSubviews: [title, slider])
        stack.spacing = 16
        stack.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: contentView.layoutMarginsGuide.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: contentView.layoutMarginsGuide.trailingAnchor),
            stack.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 8),
            stack.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -8)])
    }
    required init?(coder: NSCoder) { fatalError() }
    @objc private func changed() { onChange?(slider.value) }
}

/// Audio settings: mute, speech rate, volume, and TTS voice.
final class SettingsViewController: UITableViewController {
    private let voices = AVSpeechSynthesisVoice.speechVoices()
        .filter { $0.language.hasPrefix("en") }
        .sorted { $0.name < $1.name }

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Voice & sound"
        navigationItem.leftBarButtonItem = UIBarButtonItem(
            barButtonSystemItem: .done, target: self, action: #selector(done))
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
    }

    @objc private func done() { dismiss(animated: true) }

    override func numberOfSections(in tableView: UITableView) -> Int { 4 }

    override func tableView(_ t: UITableView, titleForHeaderInSection s: Int) -> String? {
        ["", "Speech rate", "Volume", "Voice"][s]
    }

    override func tableView(_ t: UITableView, numberOfRowsInSection s: Int) -> Int {
        s == 3 ? max(1, voices.count) : 1
    }

    override func tableView(_ t: UITableView, cellForRowAt ip: IndexPath) -> UITableViewCell {
        switch ip.section {
        case 0:
            let cell = t.dequeueReusableCell(withIdentifier: "cell", for: ip)
            cell.textLabel?.text = "Mute companion voice"
            cell.selectionStyle = .none
            let sw = UISwitch()
            sw.isOn = SettingsStore.muted
            sw.addTarget(self, action: #selector(muteChanged(_:)), for: .valueChanged)
            cell.accessoryView = sw
            return cell
        case 1:
            let cell = SliderCell(style: .default, reuseIdentifier: nil)
            cell.title.text = "🐢 → 🐇"
            cell.slider.minimumValue = AVSpeechUtteranceMinimumSpeechRate
            cell.slider.maximumValue = AVSpeechUtteranceMaximumSpeechRate
            cell.slider.value = SettingsStore.rate
            cell.onChange = { SettingsStore.rate = $0 }
            return cell
        case 2:
            let cell = SliderCell(style: .default, reuseIdentifier: nil)
            cell.title.text = "🔈 → 🔊"
            cell.slider.minimumValue = 0
            cell.slider.maximumValue = 1
            cell.slider.value = SettingsStore.volume
            cell.onChange = { SettingsStore.volume = $0 }
            return cell
        default:
            let cell = t.dequeueReusableCell(withIdentifier: "cell", for: ip)
            guard !voices.isEmpty else {
                cell.textLabel?.text = "System default"; cell.selectionStyle = .none; return cell
            }
            let voice = voices[ip.row]
            cell.textLabel?.text = "\(voice.name) (\(voice.language))"
            cell.accessoryType = voice.identifier == SettingsStore.voiceId ? .checkmark : .none
            return cell
        }
    }

    override func tableView(_ t: UITableView, didSelectRowAt ip: IndexPath) {
        guard ip.section == 3, ip.row < voices.count else { return }
        SettingsStore.voiceId = voices[ip.row].identifier
        t.reloadSections(IndexSet(integer: 3), with: .none)
    }

    @objc private func muteChanged(_ sw: UISwitch) { SettingsStore.muted = sw.isOn }
}
