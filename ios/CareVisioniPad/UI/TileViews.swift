import UIKit

/// A rounded card showing one vital: label, big value + unit, a severity dot, and
/// a faint reliability tier. Greys out when the signal is not currently present.
final class VitalTileView: UIView {
    private let labelView = UILabel()
    private let valueView = UILabel()
    private let unitView = UILabel()
    private let dot = UIView()
    private let tierView = UILabel()

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = UIColor.secondarySystemBackground
        layer.cornerRadius = 14
        layer.cornerCurve = .continuous

        labelView.font = .systemFont(ofSize: 13, weight: .medium)
        labelView.textColor = .secondaryLabel
        valueView.font = .monospacedDigitSystemFont(ofSize: 30, weight: .bold)
        valueView.textColor = .label
        unitView.font = .systemFont(ofSize: 13, weight: .regular)
        unitView.textColor = .secondaryLabel
        tierView.font = .systemFont(ofSize: 11, weight: .regular)
        tierView.textColor = .tertiaryLabel
        dot.layer.cornerRadius = 5

        let valueRow = UIStackView(arrangedSubviews: [valueView, unitView])
        valueRow.alignment = .lastBaseline
        valueRow.spacing = 4

        let header = UIStackView(arrangedSubviews: [labelView, UIView(), dot])
        header.alignment = .center
        dot.widthAnchor.constraint(equalToConstant: 10).isActive = true
        dot.heightAnchor.constraint(equalToConstant: 10).isActive = true

        let stack = UIStackView(arrangedSubviews: [header, valueRow, tierView])
        stack.axis = .vertical
        stack.spacing = 4
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 12),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -12),
            stack.topAnchor.constraint(equalTo: topAnchor, constant: 10),
            stack.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -10)])
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) not used") }

    func apply(_ tile: VitalTile) {
        labelView.text = tile.label
        valueView.text = tile.present ? tile.value : "—"
        unitView.text = tile.present ? tile.unit : "measuring…"
        tierView.text = tile.tier.map { "reliability: \($0)" } ?? ""
        dot.backgroundColor = tile.present ? tile.severity.color : .quaternaryLabel
        alpha = tile.present ? 1.0 : 0.6
    }
}

/// A compact live-demo counter/label that flashes when its value changes.
final class DemoTileView: UIView {
    private let iconView = UILabel()
    private let valueView = UILabel()
    private let labelView = UILabel()
    private var lastValue: String?

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = UIColor.tertiarySystemBackground
        layer.cornerRadius = 12
        layer.cornerCurve = .continuous

        iconView.font = .systemFont(ofSize: 20)
        valueView.font = .monospacedDigitSystemFont(ofSize: 22, weight: .bold)
        valueView.textColor = .label
        labelView.font = .systemFont(ofSize: 11, weight: .medium)
        labelView.textColor = .secondaryLabel
        labelView.numberOfLines = 1

        let stack = UIStackView(arrangedSubviews: [iconView, valueView, labelView])
        stack.axis = .vertical
        stack.alignment = .center
        stack.spacing = 2
        stack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stack)
        NSLayoutConstraint.activate([
            stack.centerXAnchor.constraint(equalTo: centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: centerYAnchor),
            stack.leadingAnchor.constraint(greaterThanOrEqualTo: leadingAnchor, constant: 6),
            stack.trailingAnchor.constraint(lessThanOrEqualTo: trailingAnchor, constant: -6)])
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) not used") }

    func apply(_ tile: DemoTile) {
        iconView.text = tile.icon
        valueView.text = tile.value
        labelView.text = tile.label
        valueView.textColor = tile.severity == .info ? .label : tile.severity.color
        if let last = lastValue, last != tile.value { flash() }
        lastValue = tile.value
    }

    private func flash() {
        let original = backgroundColor
        backgroundColor = UIColor.systemBlue.withAlphaComponent(0.35)
        UIView.animate(withDuration: 0.5) { self.backgroundColor = original }
    }
}
