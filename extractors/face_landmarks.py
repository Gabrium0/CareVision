"""MediaPipe FaceMesh landmark indices shared across modules.

Index reference: https://github.com/google-ai-edge/mediapipe (canonical
face mesh, 468 points + 10 iris points with refine_landmarks=True).
"""

# Eye contours for EAR (P1..P6 order: outer, top1, top2, inner, bottom2, bottom1)
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Full eye rings (for sclera masks)
LEFT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_RING = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                  466, 388, 387, 386, 385, 384, 398]

# Iris (refine_landmarks): center + 4 rim points each
LEFT_IRIS = [468, 469, 470, 471, 472]     # 468 = center
RIGHT_IRIS = [473, 474, 475, 476, 477]    # 473 = center

# Mouth
MOUTH_TOP_INNER = 13
MOUTH_BOTTOM_INNER = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
OUTER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
              375, 321, 405, 314, 17, 84, 181, 91, 146]

# Eyebrows (top edge)
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]

# Head geometry
NOSE_TIP = 1
CHIN = 152
FOREHEAD_TOP = 10
LEFT_FACE_EDGE = 234
RIGHT_FACE_EDGE = 454

# ROIs for photoplethysmography / skin color (approx patch centers)
FOREHEAD_ROI = [10, 109, 338, 151]        # between hairline and brows
LEFT_CHEEK = 50
RIGHT_CHEEK = 280

# Symmetric landmark pairs (left, right) for asymmetry / droop scoring
SYMMETRY_PAIRS = [
    (61, 291),    # mouth corners
    (105, 334),   # brow mid
    (159, 386),   # upper eyelid
    (145, 374),   # lower eyelid
    (50, 280),    # cheeks
    (234, 454),   # face edges
]

# Face oval (for whole-face skin mask)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
