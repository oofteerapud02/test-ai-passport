import argparse
import glob
import os
import sys

from config import TARGET_FRAMES
from .bvh import load_bvh
from .npz_smpl import load_npz
from .retarget import build_tracks
from .vrma import write_vrma

SUPPORTED = (".bvh", ".npz")


def convert(src, dst=None, src_up=None, yaw=0.0, out_fps=30.0, mode="resample",
            rest_correct=True, with_face=False, frames=TARGET_FRAMES):
    ext = os.path.splitext(src)[1].lower()
    if ext == ".bvh":
        motion = load_bvh(src)
    elif ext == ".npz":
        motion = load_npz(src, with_face=with_face)
    else:
        raise ValueError("รองรับเฉพาะ .bvh และ .npz")

    times, tracks, hips = build_tracks(motion, src_up=src_up, yaw_deg=yaw,
                                       rest_correct=rest_correct, out_fps=out_fps,
                                       mode=mode, frames=frames)
    dst = dst or os.path.splitext(src)[0] + ".vrma"
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    return write_vrma(dst, times, tracks, hips,
                      title=os.path.basename(os.path.splitext(src)[0]))


def _expand(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for ext in SUPPORTED:
                files += sorted(glob.glob(os.path.join(p, "**", "*" + ext), recursive=True))
        elif any(ch in p for ch in "*?["):
            files += sorted(glob.glob(p, recursive=True))
        else:
            files.append(p)
    return [f for f in files if os.path.splitext(f)[1].lower() in SUPPORTED]


def main():
    ap = argparse.ArgumentParser(description="แปลง .npz/.bvh เป็น .vrma (VRM Animation)")
    ap.add_argument("inputs", nargs="+", help="ไฟล์ / โฟลเดอร์ / glob")
    ap.add_argument("-o", "--output", help="ไฟล์ปลายทาง หรือโฟลเดอร์ (กรณีหลายไฟล์)")
    ap.add_argument("--up", choices=["y", "z"], default=None)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--frames", type=int, default=TARGET_FRAMES)
    ap.add_argument("--mode", choices=["resample", "clip"], default="resample")
    ap.add_argument("--face", action="store_true", help="รวม jaw/eyes (SMPL-X)")
    ap.add_argument("--no-rest-correct", action="store_true")
    a = ap.parse_args()

    files = _expand(a.inputs)
    if not files:
        print("ไม่พบไฟล์ .bvh/.npz ที่ตรงเงื่อนไข")
        sys.exit(1)

    outdir = a.output if (a.output and (len(files) > 1 or os.path.isdir(a.output))) else None
    ok = 0
    for f in files:
        dst = (os.path.join(outdir, os.path.splitext(os.path.basename(f))[0] + ".vrma")
               if outdir else (a.output if len(files) == 1 else None))
        try:
            print("OK  ", convert(f, dst, a.up, a.yaw, a.fps, a.mode,
                                  not a.no_rest_correct, a.face, a.frames))
            ok += 1
        except Exception as e:
            print("FAIL", f, "->", e)
    print(f"\nสำเร็จ {ok}/{len(files)} ไฟล์")


if __name__ == "__main__":
    main()