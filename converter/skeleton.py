import numpy as np

# VRM 1.0 humanoid — T-pose, Y-up, +Z = หน้า, หน่วยเมตร
REST = [
    ("hips",          None,           (0.000,  0.900,  0.000)),
    ("spine",         "hips",         (0.000,  0.100,  0.000)),
    ("chest",         "spine",        (0.000,  0.120,  0.000)),
    ("upperChest",    "chest",        (0.000,  0.120,  0.000)),
    ("neck",          "upperChest",   (0.000,  0.120,  0.000)),
    ("head",          "neck",         (0.000,  0.080,  0.000)),
    ("leftShoulder",  "upperChest",   (0.040,  0.100,  0.000)),
    ("leftUpperArm",  "leftShoulder", (0.100,  0.000,  0.000)),
    ("leftLowerArm",  "leftUpperArm", (0.260,  0.000,  0.000)),
    ("leftHand",      "leftLowerArm", (0.240,  0.000,  0.000)),
    ("rightShoulder", "upperChest",   (-0.040, 0.100,  0.000)),
    ("rightUpperArm", "rightShoulder",(-0.100, 0.000,  0.000)),
    ("rightLowerArm", "rightUpperArm",(-0.260, 0.000,  0.000)),
    ("rightHand",     "rightLowerArm",(-0.240, 0.000,  0.000)),
    ("leftUpperLeg",  "hips",         (0.090, -0.060,  0.000)),
    ("leftLowerLeg",  "leftUpperLeg", (0.000, -0.400,  0.000)),
    ("leftFoot",      "leftLowerLeg", (0.000, -0.400,  0.000)),
    ("leftToes",      "leftFoot",     (0.000, -0.060,  0.100)),
    ("rightUpperLeg", "hips",         (-0.090,-0.060,  0.000)),
    ("rightLowerLeg", "rightUpperLeg",(0.000, -0.400,  0.000)),
    ("rightFoot",     "rightLowerLeg",(0.000, -0.400,  0.000)),
    ("rightToes",     "rightFoot",    (0.000, -0.060,  0.100)),
    # --- ใบหน้า (ออปชัน) ---
    ("jaw",           "head",         (0.000,  0.020,  0.030)),
    ("leftEye",       "head",         (0.032,  0.060,  0.055)),
    ("rightEye",      "head",         (-0.032, 0.060,  0.055)),
]

FINGER_SEGMENTS = {
    "Thumb":  ["Metacarpal", "Proximal", "Distal"],
    "Index":  ["Proximal", "Intermediate", "Distal"],
    "Middle": ["Proximal", "Intermediate", "Distal"],
    "Ring":   ["Proximal", "Intermediate", "Distal"],
    "Little": ["Proximal", "Intermediate", "Distal"],
}
_FINGER_LEN = {"Thumb": 0.033, "Index": 0.032, "Middle": 0.034, "Ring": 0.030, "Little": 0.026}
_FINGER_SPREAD = {"Thumb": 0.030, "Index": 0.016, "Middle": 0.000, "Ring": -0.016, "Little": -0.032}


def _append_fingers():
    for side, sign in (("left", 1.0), ("right", -1.0)):
        for finger, segs in FINGER_SEGMENTS.items():
            prev = f"{side}Hand"
            for i, seg in enumerate(segs):
                dx = sign * (0.030 if i == 0 else _FINGER_LEN[finger])
                off = (dx, 0.0, _FINGER_SPREAD[finger] if i == 0 else 0.0)
                REST.append((f"{side}{finger}{seg}", prev, off))
                prev = f"{side}{finger}{seg}"


_append_fingers()

BONES = [b for b, _, _ in REST]
PARENT = {b: p for b, p, _ in REST}
OFFSET = {b: np.array(o, dtype=np.float64) for b, _, o in REST}

FACE_BONES = ("jaw", "leftEye", "rightEye")
FINGER_BONES = tuple(b for b in BONES if any(f in b for f in FINGER_SEGMENTS))

_EXPLICIT_DIR = {
    "hips": (0, 1, 0), "head": (0, 1, 0),
    "leftHand": (1, 0, 0), "rightHand": (-1, 0, 0),
    "leftToes": (0, 0, 1), "rightToes": (0, 0, 1),
    "jaw": (0, -0.4, 1), "leftEye": (0, 0, 1), "rightEye": (0, 0, 1),
}


def _norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])


def target_rest_dirs():
    """ทิศของกระดูกแต่ละชิ้นในท่า T-pose ปลายทาง (ใช้คำนวณ rest-delta)"""
    child_off = {}
    for b, p, o in REST:
        if p is not None and p not in child_off:
            child_off[p] = np.array(o, dtype=np.float64)

    dirs = {}
    for b in BONES:
        if b in _EXPLICIT_DIR:
            dirs[b] = _norm(_EXPLICIT_DIR[b])
        elif b in child_off:
            dirs[b] = _norm(child_off[b])
        else:
            dirs[b] = _norm((1.0 if b.startswith("left") else -1.0, 0.0, 0.0))
    return dirs


def hips_height():
    return float(OFFSET["hips"][1])