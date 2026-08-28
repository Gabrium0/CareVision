import Foundation
import AVFoundation
import Speech

/// On-device voice for the companion, replacing the backend Piper/Whister round
/// trip and the WebRTC audio tracks. The laptop still **authors and guards every
/// spoken line** (non-diagnostic, no numbers-with-units, deterministic
/// fallbacks); this class only voices that text and transcribes the person.
///
///   * TTS: `AVSpeechSynthesizer` speaks lines the laptop sends.
///   * ASR: `SFSpeechRecognizer` on-device (iOS 13's `requiresOnDeviceRecognition`)
///     transcribes the mic and emits `asr_text` through `onTranscript`.
///
/// Turn-taking is half-duplex: the mic is muted while the synthesiser is audible
/// plus a short room-echo tail, mirroring `Listener._agent_is_speaking` on the
/// backend so the agent never hears its own voice.
final class VoiceController: NSObject, AVSpeechSynthesizerDelegate {
    /// Called with each finalised transcript to forward as `asr_text`.
    var onTranscript: ((String) -> Void)?
    /// Called at speaking edges (true when audible, false when done) so the UI /
    /// link can report the agent's turn.
    var onSpeakingChange: ((Bool) -> Void)?

    private let synth = AVSpeechSynthesizer()
    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private let audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    private var lastSpokenText: String?
    private var muted = false
    private let echoTail: TimeInterval = 0.6
    private var listening = false
    private var bargeInHits = 0            // consecutive speech buffers during TTS

    override init() {
        super.init()
        synth.delegate = self
    }

    // MARK: - permissions

    /// Request speech + mic permission, then (if granted) start listening.
    func startListening() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            guard status == .authorized else { return }
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                guard granted else { return }
                DispatchQueue.main.async { self?.beginRecognition() }
            }
        }
    }

    func stop() {
        endRecognition()
        synth.stopSpeaking(at: .immediate)
    }

    // MARK: - TTS

    /// Speak a guarded line from the laptop. De-duplicates identical consecutive
    /// lines so a repeated `state.agent.text` is not spoken twice.
    func speak(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != lastSpokenText else { return }
        lastSpokenText = trimmed
        guard !SettingsStore.muted else { return }
        configureSession(forSpeaking: true)
        let utterance = AVSpeechUtterance(string: trimmed)
        utterance.voice = SettingsStore.voice ?? AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = SettingsStore.rate
        utterance.volume = SettingsStore.volume
        synth.speak(utterance)
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didStart u: AVSpeechUtterance) {
        muted = true
        onSpeakingChange?(true)
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didFinish u: AVSpeechUtterance) {
        finishSpeaking()
    }

    func speechSynthesizer(_ s: AVSpeechSynthesizer, didCancel u: AVSpeechUtterance) {
        finishSpeaking()
    }

    private func finishSpeaking() {
        onSpeakingChange?(false)
        // Keep the mic muted for a short echo tail before re-arming.
        DispatchQueue.main.asyncAfter(deadline: .now() + echoTail) { [weak self] in
            self?.muted = false
            self?.configureSession(forSpeaking: false)
        }
    }

    // MARK: - ASR

    private func beginRecognition() {
        guard let recognizer = recognizer, recognizer.isAvailable, !listening else { return }
        configureSession(forSpeaking: false)
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = false
        if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true   // no audio leaves the iPad
        }
        self.request = request

        let node = audioEngine.inputNode
        // Hardware echo cancellation (iOS 13+) so the mic doesn't hear the agent —
        // this is what makes barge-in possible without transcribing our own TTS.
        try? node.setVoiceProcessingEnabled(true)
        let format = node.outputFormat(forBus: 0)
        node.removeTap(onBus: 0)
        node.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            guard let self = self else { return }
            if self.muted {
                // Barge-in: while the agent talks, watch the echo-cancelled mic for
                // real speech; a short sustained burst stops the agent mid-sentence.
                if Self.isSpeech(buffer) {
                    self.bargeInHits += 1
                    if self.bargeInHits >= 3 {
                        self.bargeInHits = 0
                        DispatchQueue.main.async { self.synth.stopSpeaking(at: .immediate) }
                    }
                } else {
                    self.bargeInHits = 0
                }
                return
            }
            self.request?.append(buffer)
        }
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            return
        }
        listening = true
        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self = self else { return }
            if let result = result, result.isFinal {
                let text = result.bestTranscription.formattedString
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !self.muted, !text.isEmpty { self.onTranscript?(text) }
                self.restartRecognition()
            } else if error != nil {
                self.restartRecognition()
            }
        }
    }

    private func restartRecognition() {
        endRecognition()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { [weak self] in
            self?.beginRecognition()
        }
    }

    private func endRecognition() {
        listening = false
        task?.cancel(); task = nil
        request?.endAudio(); request = nil
        if audioEngine.isRunning { audioEngine.stop() }
        audioEngine.inputNode.removeTap(onBus: 0)
    }

    /// Cheap RMS energy VAD on one mic buffer (echo-cancelled), for barge-in.
    private static func isSpeech(_ buffer: AVAudioPCMBuffer) -> Bool {
        guard let ch = buffer.floatChannelData?[0] else { return false }
        let n = Int(buffer.frameLength)
        guard n > 0 else { return false }
        var sum: Float = 0
        for i in 0..<n { let s = ch[i]; sum += s * s }
        return (sum / Float(n)).squareRoot() > 0.02
    }

    private func configureSession(forSpeaking: Bool) {
        let session = AVAudioSession.sharedInstance()
        do {
            // playAndRecord with default-to-speaker so the person hears the
            // companion while the mic keeps capturing; ducking others is polite.
            try session.setCategory(.playAndRecord, mode: .voiceChat,
                                    options: [.defaultToSpeaker, .duckOthers,
                                              .allowBluetooth])
            try session.setActive(true, options: [])
        } catch {
            // A failed session config must not crash the app; voice degrades.
        }
    }
}
