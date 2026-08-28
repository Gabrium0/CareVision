import Foundation

/// Connection state surfaced to the UI.
enum LinkState: Equatable {
    case idle
    case connecting
    case paired
    case disconnected
    case failed(String)
}

protocol LinkClientDelegate: AnyObject {
    func link(_ link: LinkClient, didChangeState state: LinkState)
    /// A decoded control message from the laptop (telemetry, state, module_result,
    /// speak). Delivered on the main thread. `frame_ack` is consumed internally.
    func link(_ link: LinkClient, didReceiveControl type: String, payload: [String: Any])
}

/// Direct-LAN WebSocket client to the laptop's `--ipad-transport lan` server.
///
/// Speaks the exact contract analysed on the backend: an opening `hello`
/// (Pairing HMAC), IPF1 binary frames with windowed-ACK flow control, and
/// control JSON both ways. There is no relay and no WebRTC — on the Windows
/// hotspot the app reaches the laptop directly, so this is a plain
/// `URLSessionWebSocketTask` (iOS 13+).
///
/// Flow control mirrors `relay/static/ipad_capture_policy.js`: at most
/// `frameWindow` JPEGs in flight, and if the oldest is not ACKed within
/// `ackTimeout` the link reconnects. Everything runs on a private serial queue;
/// delegate callbacks hop to main.
final class LinkClient: NSObject {
    struct Config {
        var host: String
        var port: Int
        var room: String
        var secret: String
        var code: String
    }

    weak var delegate: LinkClientDelegate?

    // Flow-control constants — keep in lockstep with the browser capture policy.
    private let frameWindow = 3
    private let ackTimeout: TimeInterval = 2.5

    private let config: Config
    private let queue = DispatchQueue(label: "carevision.link")
    private var session: URLSession!
    private var task: URLSessionWebSocketTask?
    private var running = false
    private var reconnectAttempt = 0

    private var nextSeq: UInt32 = 0
    private var inFlight: [UInt32: TimeInterval] = [:]   // seq -> sentAt (monotonic)
    private var ackWatchdog: DispatchSourceTimer?
    private var ackLatencies: [Double] = []              // ms, rolling window
    private var backpressureCount = 0                    // window-full drops since drain

    private(set) var state: LinkState = .idle {
        didSet { emitState() }
    }

    init(config: Config) {
        self.config = config
        super.init()
        let cfg = URLSessionConfiguration.default
        cfg.waitsForConnectivity = false
        cfg.timeoutIntervalForRequest = 20
        self.session = URLSession(configuration: cfg)
    }

    // MARK: - lifecycle

    func start() {
        queue.async { [weak self] in
            guard let self = self, !self.running else { return }
            self.running = true
            self.connect()
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self = self else { return }
            self.running = false
            self.teardown(state: .idle)
        }
    }

    /// Force an immediate reconnect (manual "Reconnect"), resetting backoff.
    func reconnectNow() {
        queue.async { [weak self] in
            guard let self = self, self.running else { return }
            self.reconnectAttempt = 0
            self.teardown(state: .connecting)
            self.connect()
        }
    }

    private func connect() {
        guard running else { return }
        state = .connecting
        guard let url = URL(string: "ws://\(config.host):\(config.port)/ws") else {
            state = .failed("bad url"); return
        }
        let task = session.webSocketTask(with: url)
        self.task = task
        inFlight.removeAll()
        task.resume()
        // Send the hello first thing; the server closes 4401 on a bad one.
        let hello = Pairing.helloJSON(room: config.room, secret: config.secret,
                                      code: config.code)
        task.send(.string(hello)) { [weak self] error in
            guard let self = self else { return }
            self.queue.async {
                if let error = error {
                    self.scheduleReconnect(reason: "hello: \(error.localizedDescription)")
                    return
                }
                self.state = .paired
                self.reconnectAttempt = 0
                self.startAckWatchdog()
            }
        }
        receiveLoop()
    }

    private func teardown(state newState: LinkState) {
        ackWatchdog?.cancel(); ackWatchdog = nil
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        inFlight.removeAll()
        state = newState
    }

    private func scheduleReconnect(reason: String) {
        guard running else { return }
        teardown(state: .disconnected)
        reconnectAttempt += 1
        let delay = min(pow(2.0, Double(min(reconnectAttempt, 5))), 30.0)
        queue.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self = self, self.running else { return }
            self.connect()
        }
    }

    // MARK: - receive

    private func receiveLoop() {
        task?.receive { [weak self] result in
            guard let self = self else { return }
            self.queue.async {
                switch result {
                case .failure(let error):
                    self.scheduleReconnect(reason: error.localizedDescription)
                case .success(let message):
                    switch message {
                    case .string(let text): self.handleControl(text)
                    case .data(let data):
                        if let text = String(data: data, encoding: .utf8) {
                            self.handleControl(text)
                        }
                    @unknown default: break
                    }
                    if self.task != nil { self.receiveLoop() }
                }
            }
        }
    }

    private func handleControl(_ text: String) {
        guard let data = text.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = obj["type"] as? String else { return }
        if type == "frame_ack" {
            if let seq = (obj["seq"] as? NSNumber)?.uint32Value { acknowledge(seq) }
            return
        }
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.delegate?.link(self, didReceiveControl: type, payload: obj)
        }
    }

    // MARK: - flow control

    /// True when the sender window has room. The camera checks this before it
    /// spends CPU encoding a JPEG, so a stalled link sheds work instead of
    /// buffering.
    var readyForFrame: Bool {
        queue.sync { task != nil && state == .paired && inFlight.count < frameWindow }
    }

    /// Send one JPEG as an IPF1 frame. No-op (dropped) if the window is full.
    func sendFrame(jpeg: Data, width: UInt16, height: UInt16, mediaTime: Double) {
        queue.async { [weak self] in
            guard let self = self, let task = self.task, self.state == .paired else { return }
            if self.inFlight.count >= self.frameWindow {
                self.backpressureCount += 1     // sender window full → shed this frame
                return
            }
            let seq = self.nextSeq
            self.nextSeq = self.nextSeq &+ 1
            self.inFlight[seq] = ProcessInfo.processInfo.systemUptime
            let msg = FrameProtocol.frameMessage(seq: seq, mediaTime: mediaTime,
                                                 width: width, height: height, jpeg: jpeg)
            task.send(.data(msg)) { [weak self] error in
                guard let self = self, error != nil else { return }
                self.queue.async { self.inFlight.removeValue(forKey: seq) }
            }
        }
    }

    private func acknowledge(_ seq: UInt32) {
        guard let sentAt = inFlight.removeValue(forKey: seq) else { return }
        let ms = (ProcessInfo.processInfo.systemUptime - sentAt) * 1000
        ackLatencies.append(ms)
        if ackLatencies.count > 60 { ackLatencies.removeFirst(ackLatencies.count - 60) }
    }

    /// Rolling ACK p90 (ms) and whether the window backed up, then reset the
    /// window. Drives the adaptive capture ladder. Safe from any thread.
    func drainFlowStats() -> (ackP90Ms: Double, backpressured: Bool) {
        queue.sync {
            var p90 = 0.0
            if !ackLatencies.isEmpty {
                let sorted = ackLatencies.sorted()
                p90 = sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.9))]
            }
            let bp = backpressureCount > 0
            ackLatencies.removeAll(keepingCapacity: true)
            backpressureCount = 0
            return (p90, bp)
        }
    }

    private func startAckWatchdog() {
        ackWatchdog?.cancel()
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 1, repeating: 1)
        timer.setEventHandler { [weak self] in
            guard let self = self else { return }
            let now = ProcessInfo.processInfo.systemUptime
            if let oldest = self.inFlight.values.min(), now - oldest > self.ackTimeout {
                self.scheduleReconnect(reason: "ack timeout")
            }
        }
        timer.resume()
        ackWatchdog = timer
    }

    // MARK: - control send

    /// Send a JSON control message to the laptop (client_version, sender_stats,
    /// camera_tuning, capture_profile, module, asr_text, speaking).
    func sendControl(_ payload: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let text = String(data: data, encoding: .utf8) else { return }
        queue.async { [weak self] in
            self?.task?.send(.string(text)) { _ in }
        }
    }

    private func emitState() {
        let s = state
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.delegate?.link(self, didChangeState: s)
        }
    }
}
