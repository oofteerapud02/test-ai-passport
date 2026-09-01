import re

import numpy as np
from scipy.spatial.transform import Rotation as R

from .motion import Motion

CENTER = {
    "hips": "hips", "hip": "hips", "pelvis": "hips", "root": "hips", "reference": "hips",
    "spine": "spine", "spine1": "chest", "chest": "chest",
    "spine2": "upperChest", "spine3": "upperChest", "upperchest": "upperChest",
    "neck": "neck", "neck1": "neck", "neck2": "neck",
    "head": "head", "jaw": "jaw",
}
SIDED = {
    "shoulder": "Shoulder", "collar": "Shoulder", "clavicle": "Shoulder",
    "arm": "UpperArm", "upperarm": "UpperArm", "uparm": "UpperArm",
    "forearm": "LowerArm", "lowerarm": "LowerArm", "elbow": "LowerArm",
    "hand": "Hand", "wrist": "Hand",
    "upleg": "UpperLeg", "upperleg": "UpperLeg", "thigh": "UpperLeg", "hipjoint": "UpperLeg",
    "leg": "LowerLeg", "lowerleg": "LowerLeg", "knee": "LowerLeg", "shin": "LowerLeg", "calf": "LowerLeg",
    "foot": "Foot", "ankle": "Foot",
    "toebase": "Toes", "toe": "Toes", "toe0": "Toes", "ball": "Toes",
    "eye": "Eye",
}
_FINGER_RE = re.compile(r"^(?:hand)?(thumb|index|middle|ring|pinky|little)(\d)$")
_FINGER_VRM = {"thumb": "Thumb", "index": "Index", "middle": "Middle",
               "ring": "Ring", "pinky": "Little", "little": "Little"}
_SEG_BY_NUM = {"thumb": {"1": "Metacarpal", "2": "Proximal", "3": "Distal"},
               "_": {"1": "Proximal", "2": "Intermediate", "3": "Distal"}}


def _norm_name(n: str) -> str:
    n = n.split(":")[-1].lower()
    return re.sub(r"[^a-z0-9]", "", n)


def _split_side(n: str):
    for pre, side in (("left", "left"), ("right", "right"), ("l", "left"), ("r", "right")):
        if n.startswith(pre) and len(n) > len(pre):
            return side, n[len(pre):]
    for suf, side in (("left", "left"), ("right", "right"), ("l", "left"), ("r", "right")):
        if n.endswith(suf) and len(n) > len(suf):
            return side, n[: -len(suf)]
    return "", n


def guess_bone(name: str):
    """แปลงชื่อ joint BVH -> ชื่อกระดูก VRM (คืน None ถ้าไม่รู้จัก)"""
    n = _norm_name(name)
    if n in CENTER:
        return CENTER[n]

    side, base = _split_side(n)
    if not side:
        return None

    m = _FINGER_RE.match(base)
    if m:
        finger, num = m.group(1), m.group(2)
        seg = _SEG_BY_NUM["thumb" if finger == "thumb" else "_"].get(num)
        return f"{side}{_FINGER_VRM[finger]}{seg}" if seg else None

    key = SIDED.get(base)
    return f"{side}{key}" if key else None


class _Joint:
    __slots__ = ("name", "parent", "offset", "channels", "cidx")

    def __init__(self):
        self.offset = np.zeros(3)
        self.channels = []
        self.cidx = []


def parse_bvh(path):
    with open(path, "r", errors="ignore") as f:
        tok = f.read().replace("\n", " \n ").split()

    joints, stack, nch, i = [], [], 0, 0
    while i < len(tok):
        t = tok[i]
        if t in ("ROOT", "JOINT"):
            j = _Joint()
            j.name = tok[i + 1]
            j.parent = stack[-1] if stack and stack[-1] >= 0 else -1
            joints.append(j)
            stack.append(len(joints) - 1)
            i += 2
        elif t == "End":
            stack.append(-2)
            i += 2
        elif t == "OFFSET":
            off = np.array([float(tok[i + 1]), float(tok[i + 2]), float(tok[i + 3])])
            if stack and stack[-1] >= 0:
                joints[stack[-1]].offset = off
            i += 4
        elif t == "CHANNELS":
            c = int(tok[i + 1])
            j = joints[stack[-1]]
            j.channels = tok[i + 2: i + 2 + c]
            j.cidx = list(range(nch, nch + c))
            nch += c
            i += 2 + c
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "MOTION":
            break
        else:
            i += 1

    nframes = int(tok[tok.index("Frames:") + 1])
    ftime = float(tok[tok.index("Time:") + 1])
    start = tok.index("Time:") + 2
    data = np.array(tok[start: start + nframes * nch], dtype=np.float64).reshape(nframes, nch)
    return joints, data, ftime


def load_bvh(path) -> Motion:
    joints, data, ftime = parse_bvh(path)
    n = data.shape[0]

    span = float(sum(np.linalg.norm(j.offset) for j in joints))
    unit = 0.01 if span > 20.0 else 1.0     # เดา cm -> m

    world_q = [None] * len(joints)
    world, root_pos = {}, np.zeros((n, 3))
    first_child = {}

    for j in joints:
        if j.parent >= 0:
            first_child.setdefault(joints[j.parent].name, j.offset)

    for k, j in enumerate(joints):
        rot_ch = [(c, idx) for c, idx in zip(j.channels, j.cidx) if c.lower().endswith("rotation")]
        if rot_ch:
            seq = "".join(c[0].upper() for c, _ in rot_ch)
            ang = np.stack([data[:, idx] for _, idx in rot_ch], axis=1)
            local = R.from_euler(seq, ang, degrees=True)
        else:
            local = R.identity(n)
        world_q[k] = local if j.parent < 0 else world_q[j.parent] * local

        bone = guess_bone(j.name)
        if bone and bone not in world:
            world[bone] = world_q[k].as_quat()

        pos_ch = [idx for c, idx in zip(j.channels, j.cidx) if c.lower().endswith("position")]
        if j.parent < 0 and len(pos_ch) == 3:
            root_pos = data[:, pos_ch] * unit

    rest = {}
    for j in joints:
        bone = guess_bone(j.name)
        d = first_child.get(j.name)
        if bone and d is not None and np.linalg.norm(d) > 1e-9 and bone not in rest:
            rest[bone] = d / np.linalg.norm(d)

    return Motion(world_rot=world, hips_pos=root_pos, rest_dir=rest,
                  fps=(1.0 / ftime if ftime > 0 else 30.0), space="src", up="y", source="bvh")