"""
Retarget: SMPL/AMASS world rotations -> VRM normalized humanoid local rotations

สมการหลัก:
    T_b(t) = W_b(t) · A_b          โดย A_b · t_b = d_b
    local_b = T_parent(t)⁻¹ · T_b(t)

จุดสำคัญ: A_b ต้องกำหนด "twist" รอบแกนกระดูกให้สอดคล้องกันทุกข้อ
ไม่งั้น twist ที่ไม่ตรงกันระหว่างพ่อ-ลูกจะสะสมเป็นการบิด (แขนไขว้ / เข่าหุบ)
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp  # type: ignore[attr-defined]

from config import TARGET_FRAMES
from .motion import Motion
from .skeleton import BONES, PARENT, target_rest_dirs


# ---------------------------------------------------------------- #
# Twist-stable alignment
# ---------------------------------------------------------------- #

_REF_PRIMARY = np.array([0.0, 0.0, 1.0])   # +Z
_REF_FALLBACK = np.array([0.0, 1.0, 0.0])  # +Y


def _basis(direction, ref=_REF_PRIMARY):
    """สร้าง orthonormal frame จากทิศกระดูก -> twist ถูกกำหนดแน่นอน"""
    x = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(x)
    if n < 1e-9:
        return R.identity()
    x = x / n
    r = ref if abs(float(np.dot(x, ref))) < 0.99 else _REF_FALLBACK
    z = np.cross(x, r)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return R.from_matrix(np.column_stack([x, y, z]))


def _align(t_dir, d_dir):
    """A ที่ทำให้ A·t_dir = d_dir พร้อม twist ที่สอดคล้องกันทุกกระดูก"""
    return _basis(d_dir) * _basis(t_dir).inv()


def _rot_between(a, b):
    """[DEPRECATED] minimal-arc rotation — twist ไม่ถูกกำหนด ทำให้ข้อพับบิด
    เก็บไว้เพื่อเทียบผลเท่านั้น อย่าใช้ใน build_tracks
    """
    v = np.cross(a, b)
    w = 1.0 + float(np.dot(a, b))
    if w < 1e-8:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return R.from_quat(np.append(axis / np.linalg.norm(axis), 0.0))
    q = np.append(v, w)
    return R.from_quat(q / np.linalg.norm(q))


# ---------------------------------------------------------------- #
# Debug
# ---------------------------------------------------------------- #

def _debug_dump(local):
    """ตรวจความสมมาตรซ้าย-ขวาของเฟรมแรก (เปิดด้วย RETARGET_DEBUG=1)"""
    def _e(rot):
        e = rot.as_euler("XYZ", degrees=True)
        return e[0] if np.asarray(e).ndim > 1 else e

    print(f"\n{'bone':<20}{'X':>8}{'Y':>8}{'Z':>8}")
    print("-" * 44)
    for b in BONES:
        if b in local:
            e = _e(local[b])
            print(f"{b:<20}{e[0]:8.1f}{e[1]:8.1f}{e[2]:8.1f}")

    print(f"\n{'symmetry check':<20}{'err':>8}")
    print("-" * 30)
    worst = 0.0
    for lb in [b for b in BONES if b in local and b.startswith("left")]:
        rb = "right" + lb[4:]
        if rb not in local:
            continue
        a, c = _e(local[lb]), _e(local[rb])
        # ซ้าย/ขวาควรสมมาตร: X เท่ากัน, Y และ Z กลับเครื่องหมาย
        err = abs(a[0] - c[0]) + abs(a[1] + c[1]) + abs(a[2] + c[2])
        worst = max(worst, err)
        print(f"{lb:<20}{err:8.1f}{'  <-- ASYM' if err > 25 else ''}")
    print(f"\nworst asymmetry = {worst:.1f} deg "
          f"({'OK' if worst <= 25 else 'ยังมีปัญหา twist/rest'})\n")


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #

def build_tracks(motion: Motion, src_up=None, yaw_deg=0.0, rest_correct=True,
                 out_fps=30.0, mode="resample", frames=TARGET_FRAMES):
    up = src_up or motion.up
    C = R.identity()
    if up == "z":                                   # <-- ไม่ผูกกับ motion.space แล้ว
        C = R.from_euler("x", -90, degrees=True)    # Z-up -> Y-up
    Cin = C.inv()
    tdirs = target_rest_dirs()

    if os.getenv("RETARGET_DEBUG"):
        print(f"[dbg] space={getattr(motion, 'space', None)!r} up={up!r} "
              f"fps={motion.fps} bones={len(motion.world_rot)}")

    # --- guard: rest correction ต้องทำครบทุกกระดูก หรือไม่ทำเลย -------------
    if rest_correct:
        missing = [b for b in BONES
                   if b in motion.world_rot and b not in motion.rest_dir]
        if missing:
            print(f"[warn] rest_dir ขาด {len(missing)} กระดูก "
                  f"-> ปิด rest_correct ทั้งชุด: {missing[:6]}")
            rest_correct = False

    # --- 1) แปลงพิกัด + ชดเชย rest pose -> world rotation ในสเปซ VRM --------
    world = {}
    for bone, quat in motion.world_rot.items():
        if bone not in BONES:
            continue
        w = C * R.from_quat(np.asarray(quat, dtype=np.float64)) * Cin
        if rest_correct and bone in motion.rest_dir:
            d = C.apply(np.asarray(motion.rest_dir[bone], dtype=np.float64))
            norm = np.linalg.norm(d)
            if norm > 1e-9:
                w = w * _align(tdirs[bone], d / norm)   # T(t) = W_src(t) · A_b
        world[bone] = w

    if "hips" not in world:
        raise ValueError("ไม่พบกระดูก hips ในไฟล์ต้นทาง (ตรวจชื่อ joint หรือรูปแบบไฟล์)")
    n = len(world["hips"])

    # --- guard: ทุก track ต้องมีจำนวนเฟรมเท่ากัน (Slerp พังเงียบถ้าไม่เท่า) --
    bad = {b: len(r) for b, r in world.items() if len(r) != n}
    if bad:
        raise ValueError(f"จำนวนเฟรมไม่ตรงกับ hips ({n}): {bad}")

    # --- 2) re-localize ตามสาย parent ของ VRM (กระดูกที่ไม่ map จะถูกยุบเข้าลูก)
    local = {}
    for bone in BONES:
        if bone not in world:
            continue
        p = PARENT[bone]
        while p is not None and p not in world:
            p = PARENT[p]
        local[bone] = world[bone] if p is None else world[p].inv() * world[bone]

    # --- 3) หมุนทั้งตัวด้วย yaw (pre-multiply ที่ root เท่านั้น) ------------
    hips_pos = C.apply(np.asarray(motion.hips_pos, dtype=np.float64).reshape(-1, 3))
    if abs(yaw_deg) > 1e-6:
        Y = R.from_euler("y", yaw_deg, degrees=True)
        local["hips"] = Y * local["hips"]
        hips_pos = Y.apply(hips_pos)
    if hips_pos.shape[0] != n:
        hips_pos = (np.tile(hips_pos[:1], (n, 1)) if hips_pos.size
                    else np.zeros((n, 3)))

    if os.getenv("RETARGET_DEBUG"):
        _debug_dump(local)

    # --- 4) resample -> จำนวนเฟรมคงที่ (ค่าเริ่มต้น 180) --------------------
    src_t = np.arange(n) / max(motion.fps, 1e-6)
    out_t = (np.arange(frames) / out_fps).astype(np.float32)
    if n < 2:
        dst_t = np.zeros(frames)
    elif mode == "clip":                    # คงความเร็วเดิม ตัด/ค้างที่เฟรมสุดท้าย
        dst_t = np.minimum(np.arange(frames) / out_fps, src_t[-1])
    else:                                   # resample: ยืด/บีบทั้งคลิปให้พอดี
        dst_t = np.linspace(src_t[0], src_t[-1], frames)

    tracks = {}
    for bone, rot in local.items():
        if n < 2:
            tracks[bone] = np.tile(rot.as_quat().reshape(-1, 4)[0],
                                   (frames, 1)).astype(np.float32)
        else:
            tracks[bone] = Slerp(src_t, rot)(dst_t).as_quat().astype(np.float32)

    if n < 2:
        hips_track = np.tile(hips_pos[0], (frames, 1)).astype(np.float32)
    else:
        hips_track = np.stack([np.interp(dst_t, src_t, hips_pos[:, k])
                               for k in range(3)], axis=1).astype(np.float32)

    return out_t, tracks, hips_track