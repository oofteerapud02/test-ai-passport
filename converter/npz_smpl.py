import numpy as np
from scipy.spatial.transform import Rotation as R

from .motion import Motion
from .skeleton import PARENT, target_rest_dirs

SMPL_PARENT_22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SMPL2VRM_22 = {
    0: "hips", 1: "leftUpperLeg", 2: "rightUpperLeg", 3: "spine",
    4: "leftLowerLeg", 5: "rightLowerLeg", 6: "chest",
    7: "leftFoot", 8: "rightFoot", 9: "upperChest",
    10: "leftToes", 11: "rightToes", 12: "neck",
    13: "leftShoulder", 14: "rightShoulder", 15: "head",
    16: "leftUpperArm", 17: "rightUpperArm", 18: "leftLowerArm",
    19: "rightLowerArm", 20: "leftHand", 21: "rightHand",
}
SMPL_REST_DIR = {
    "spine": (0, 1, 0), "chest": (0, 1, 0), "upperChest": (0, 1, 0),
    "neck": (0, 1, 0), "head": (0, 1, 0),
    "leftShoulder": (0.98, 0.20, 0), "rightShoulder": (-0.98, 0.20, 0),
    "leftUpperArm": (0.985, -0.174, 0), "rightUpperArm": (-0.985, -0.174, 0),
    "leftLowerArm": (0.985, -0.174, 0), "rightLowerArm": (-0.985, -0.174, 0),
    "leftHand": (1, 0, 0), "rightHand": (-1, 0, 0),
    "leftUpperLeg": (0.06, -0.998, 0), "rightUpperLeg": (-0.06, -0.998, 0),
    "leftLowerLeg": (0, -1, 0), "rightLowerLeg": (0, -1, 0),
    "leftFoot": (0, -0.35, 0.94), "rightFoot": (0, -0.35, 0.94),
    "leftToes": (0, 0, 1), "rightToes": (0, 0, 1),
}
_MANO_ORDER = ["index", "middle", "pinky", "ring", "thumb"]
_MANO_VRM = {"index": "Index", "middle": "Middle", "pinky": "Little",
             "ring": "Ring", "thumb": "Thumb"}


def _pick(d, keys):
    for k in keys:
        if k in d:
            return np.asarray(d[k])
    return None


def _add_hand(parents, mapping, idx, wrist, side):
    for finger in _MANO_ORDER:
        prev = wrist
        segs = (["Metacarpal", "Proximal", "Distal"] if finger == "thumb"
                else ["Proximal", "Intermediate", "Distal"])
        for seg in segs:
            parents.append(prev)
            mapping[idx] = f"{side}{_MANO_VRM[finger]}{seg}"
            prev = idx
            idx += 1
    return idx


def build_joint_schema(n_joints, with_face=False):
    """auto-detect SMPL(24) / SMPL-H(52) / SMPL-X(55)"""
    parents = list(SMPL_PARENT_22)
    mapping = dict(SMPL2VRM_22)

    if n_joints <= 24:
        parents += [20, 21]
        return parents[:n_joints], mapping

    if n_joints <= 52:                       # SMPL-H
        idx = _add_hand(parents, mapping, 22, 20, "left")
        _add_hand(parents, mapping, idx, 21, "right")
        return parents[:n_joints], mapping

    parents += [15, 15, 15]                  # SMPL-X: jaw, leftEye, rightEye
    if with_face:
        mapping[22] = "jaw"
        mapping[23] = "leftEye"
        mapping[24] = "rightEye"
    idx = _add_hand(parents, mapping, 25, 20, "left")
    _add_hand(parents, mapping, idx, 21, "right")
    return parents[:n_joints], mapping


def load_npz(path, with_face=False) -> Motion:
    d = dict(np.load(path, allow_pickle=True))
    fps_raw = _pick(d, ["mocap_framerate", "mocap_frame_rate", "fps", "frame_rate"])
    fps = float(np.asarray(fps_raw).reshape(-1)[0]) if fps_raw is not None else 30.0

    poses = _pick(d, ["poses", "smpl_poses", "pose", "thetas", "rotations"])
    if poses is None and "pose_body" in d:
        body = np.asarray(d["pose_body"])
        root = _pick(d, ["root_orient"])
        root = np.zeros((body.shape[0], 3)) if root is None else np.asarray(root)
        parts = [root.reshape(body.shape[0], -1), body.reshape(body.shape[0], -1)]
        for key in ("pose_jaw", "pose_eye", "pose_hand"):
            if key in d:
                parts.append(np.asarray(d[key]).reshape(body.shape[0], -1))
        poses = np.concatenate(parts, axis=1)

    if poses is not None:
        poses = np.asarray(poses, dtype=np.float64)
        poses = poses.reshape(poses.shape[0], -1)
        nj = poses.shape[1] // 3
        aa = poses[:, : nj * 3].reshape(-1, nj, 3)
        trans = _pick(d, ["trans", "transl", "translation", "root_translation"])
        trans = (np.zeros((aa.shape[0], 3)) if trans is None
                 else np.asarray(trans, dtype=np.float64).reshape(-1, 3))
        parents, mapping = build_joint_schema(nj, with_face)
        return _smpl_to_world(aa, trans, fps, parents, mapping)

    joints = _pick(d, ["motion", "joints", "positions", "pred_xyz", "keypoints3d", "joints3d"])
    if joints is None:
        raise ValueError(f"ไม่พบคีย์ pose/joint ที่รองรับ (คีย์ในไฟล์: {list(d.keys())[:12]})")

    j = np.asarray(joints, dtype=np.float64).squeeze()
    if j.ndim == 4:
        j = j[0]
    if j.ndim == 3 and j.shape[1] == 3 and j.shape[0] in (22, 24, 52, 55):
        j = j.transpose(2, 0, 1)
    if j.ndim != 3:
        raise ValueError(f"รูปทรงข้อมูลข้อต่อไม่รองรับ: {j.shape}")
    return _positions_to_world(j, fps)


def _smpl_to_world(aa, trans, fps, parents, mapping) -> Motion:
    n, nj, _ = aa.shape
    world = [None] * nj
    out, rest = {}, {}
    for k in range(nj):
        local = R.from_rotvec(aa[:, k])
        p = parents[k] if k < len(parents) else -1
        world[k] = local if p < 0 else world[p] * local
        bone = mapping.get(k)
        if bone:
            out[bone] = world[k].as_quat()
            if bone in SMPL_REST_DIR:
                rest[bone] = np.array(SMPL_REST_DIR[bone], dtype=np.float64)
    return Motion(world_rot=out, hips_pos=trans.copy(), rest_dir=rest,
                  fps=fps, space="src", up="z", source="smpl")


def _swing(a, b):
    v = np.cross(a, b)
    w = 1.0 + np.sum(a * b, axis=1)
    q = np.concatenate([v, w[:, None]], axis=1)
    bad = w < 1e-6
    if np.any(bad):
        q[bad] = np.array([0.0, 0.0, 1.0, 0.0])
    return R.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True))


def _positions_to_world(j, fps) -> Motion:
    """โหมดสำรอง: มีแค่ตำแหน่งข้อต่อ (MDM / HumanML3D 22 จุด)"""
    dirs = target_rest_dirs()
    idx = {b: k for k, b in SMPL2VRM_22.items() if k < j.shape[1]}
    n = j.shape[0]

    up_ref = "upperChest" if "upperChest" in idx else "spine"
    x = j[:, idx["leftUpperLeg"]] - j[:, idx["rightUpperLeg"]]
    y = j[:, idx[up_ref]] - j[:, idx["hips"]]
    z = np.cross(x, y)
    x = np.cross(y, z)
    unit = lambda v: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    body = R.from_matrix(np.stack([unit(x), unit(y), unit(z)], axis=2))

    child_of = {}
    for c, p in PARENT.items():
        if p and p in idx and c in idx:
            child_of.setdefault(p, c)

    out = {"hips": body.as_quat()}
    inv = body.inv()
    for b, c in child_of.items():
        if b == "hips":
            continue
        d = unit(j[:, idx[c]] - j[:, idx[b]])
        swing = _swing(np.tile(dirs[b], (n, 1)), inv.apply(d))
        out[b] = (body * swing).as_quat()

    return Motion(world_rot=out, hips_pos=j[:, idx["hips"]].copy(), rest_dir={},
                  fps=fps, space="vrm", up="y", source="positions")