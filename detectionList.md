# Detection List

Candidate signals detectable from RGB video footage (webcam/humanoid camera), organized by category. Feasibility notes indicate how reliable/mature each detection is with standard camera hardware vs. requiring special conditions (close-up, good lighting, IR/thermal, or longitudinal tracking).

## Vital Signs (remote photoplethysmography / motion-based)
- Emotion
- rPPG heart rate estimation
- Heart rate variability (HRV) — stress/fatigue proxy, needs stable rPPG signal
- Respiratory rate — via chest/shoulder movement or rPPG-derived signal
- Blood oxygen proxy (SpO2 estimation) — experimental, needs calibration, low reliability with RGB-only
- Blood pressure trend estimation — experimental, rPPG-based, low reliability

## Facial & Skin Analysis
- Rash / skin eruption detection
- Pallor (unusually pale skin) — possible anemia/illness indicator
- Flushing / redness — possible fever, hypertension, alcohol
- Cyanosis (bluish tint to lips/skin) — possible low oxygen, needs good color calibration
- Jaundice (yellowing of skin/eyes) — possible liver issue, needs good white balance
- Facial asymmetry / drooping — stroke indicator (FAST protocol relevant)
- Bruising / visible injury
- Swelling / edema (face, eyelids)
- Eye redness / conjunctivitis
- Excessive sweating (visible perspiration)
- Dry/cracked lips — possible dehydration indicator
- Grooming/hygiene changes over time — self-neglect indicator (longitudinal)

## Neurological / Motor Function
- Hand or limb tremor
- Gait abnormality (shuffling, limping, asymmetric stride)
- Balance instability / staggering / swaying
- Slowed movement (bradykinesia)
- Reduced facial expressiveness (masked face) — possible Parkinsonian indicator
- Abnormal eye movement / nystagmus
- Pupil size abnormality (given adequate camera resolution/lighting)
- Slurred or difficult speech onset (via mouth movement, if paired with audio)

## Fatigue, Drowsiness & Consciousness
- Eye closure duration / PERCLOS (drowsiness)
- Yawning frequency
- Head nodding / micro-sleep episodes
- Prolonged unresponsiveness / lack of movement
- Loss of consciousness / collapse

## Falls & Safety Events
- Fall detection (sudden posture collapse)
- Prolonged immobility after a fall
- Difficulty getting up / repeated attempts to stand
- Wandering behavior (repetitive pacing, disorientation in space)
- Entering hazardous zones (stove, stairs) — needs scene/zone calibration

## Emotional & Cognitive State
- Basic emotion classification (happy, sad, angry, fear, surprise, disgust, neutral)
- Pain expression / grimacing
- Confusion or disorientation (facial cues + behavior pattern)
- Agitation / restlessness (repetitive motion, pacing)
- Apathy / flat affect over time (longitudinal)
- Social withdrawal (reduced facial engagement, gaze avoidance) — longitudinal

## Behavioral & Routine Patterns (longitudinal, needs history)
- Activity level / inactivity trends
- Sleep pattern quality (motion during sleep, if camera covers bedroom)
- Medication-taking behavior (pill bottle + hand-to-mouth gesture, needs scene context)
- Eating/drinking frequency (needs kitchen/dining view)
- Time spent in each room / routine disruption
- Weight change estimation (body habitus over time) — low precision, needs consistent camera angle/distance

## Demographic / Contextual Estimation
- Age estimation
- Approximate BMI/body composition estimation — low precision from RGB alone
- Clothing appropriateness (e.g., not dressed for cold, same clothes for days) — self-neglect proxy

---

**Notes on reliability:**
- High confidence with standard RGB webcam + good lighting: emotion, gaze/eye closure, gross posture/fall, gait, tremor.
- Medium confidence, sensitive to lighting/camera quality/skin tone calibration: rPPG heart rate/HRV/respiration, pallor/flushing/cyanosis/jaundice, pain expression.
- Low confidence / experimental with RGB-only: SpO2, blood pressure, BMI, dehydration, weight change — consider flagging these as "possible indicator, not diagnostic" rather than hard detections.
- Longitudinal detections (grooming, routine, activity trends) require historical baselines per person, not single-frame detection.
- Consider pairing with audio (cough, speech slurring, falls sound) and IR/thermal camera if skin-temperature or low-light detection becomes a priority — these are outside pure RGB video scope.
