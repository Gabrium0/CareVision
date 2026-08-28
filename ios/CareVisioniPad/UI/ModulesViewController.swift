import UIKit

/// The module toggle drawer — the native equivalent of the browser `/modules`
/// console. Each row shows a detector and a switch; flipping it sends a `module`
/// enable/disable command to the laptop. Locked (non-toggleable) rows are shown
/// disabled, matching the page.
final class ModulesViewController: UITableViewController {
    private let modules: [ModuleRow]
    private let onToggle: (String, Bool) -> Void

    init(modules: [ModuleRow], onToggle: @escaping (String, Bool) -> Void) {
        self.modules = modules.sorted { $0.label < $1.label }
        self.onToggle = onToggle
        super.init(style: .insetGrouped)
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) not used") }

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Modules"
        navigationItem.leftBarButtonItem = UIBarButtonItem(
            barButtonSystemItem: .done, target: self, action: #selector(done))
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
    }

    @objc private func done() { dismiss(animated: true) }

    override func tableView(_ t: UITableView, numberOfRowsInSection s: Int) -> Int {
        modules.isEmpty ? 1 : modules.count
    }

    override func tableView(_ t: UITableView, cellForRowAt ip: IndexPath) -> UITableViewCell {
        let cell = t.dequeueReusableCell(withIdentifier: "cell", for: ip)
        cell.selectionStyle = .none
        cell.accessoryView = nil
        guard !modules.isEmpty else {
            cell.textLabel?.text = "Waiting for module list…"
            cell.textLabel?.textColor = .secondaryLabel
            return cell
        }
        let module = modules[ip.row]
        cell.textLabel?.text = module.label
        cell.detailTextLabel?.text = module.running ? "running" : "idle"

        let toggle = UISwitch()
        toggle.isOn = module.enabled
        toggle.isEnabled = module.toggleable
        toggle.tag = ip.row
        toggle.addTarget(self, action: #selector(switched(_:)), for: .valueChanged)
        cell.accessoryView = toggle
        return cell
    }

    @objc private func switched(_ sender: UISwitch) {
        guard sender.tag < modules.count else { return }
        onToggle(modules[sender.tag].name, sender.isOn)
    }
}
