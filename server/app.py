import io
import os
import shutil
import tempfile
import uuid
import zipfile
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from converter.cli import convert

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(BASE, "web")
WORK = os.path.join(tempfile.gettempdir(), "vrma_converter")
os.makedirs(WORK, exist_ok=True)

app = FastAPI(title="VRMA Converter", version="1.1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ---------- ตัวแปลงค่าแบบทนพัง (กัน "undefined" / ค่าว่างจากฝั่งเว็บ) ----------
def _int(v, default):
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return default


def _float(v, default):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _clean(up, yaw, fps, mode, face, frames):
    u = str(up).strip().lower()
    f = _float(fps, 30.0)
    n = _int(frames, config.TARGET_FRAMES)
    return (
        u if u in ("y", "z") else "",
        _float(yaw, 0.0),
        f if f > 0 else 30.0,
        mode if mode in ("resample", "clip") else "resample",
        _bool(face),
        n if 1 <= n <= 100000 else config.TARGET_FRAMES,
    )


def _run(upload: UploadFile, data: bytes, up, yaw, fps, mode, face, frames):
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in (".bvh", ".npz"):
        raise HTTPException(400, f"รองรับเฉพาะ .bvh และ .npz (ได้รับ: {upload.filename})")
    if not data:
        raise HTTPException(400, f"ไฟล์ว่างเปล่า: {upload.filename}")

    uid = uuid.uuid4().hex[:10]
    src = os.path.join(WORK, uid + ext)
    with open(src, "wb") as f:
        f.write(data)
    try:
        dst = convert(src, os.path.join(WORK, uid + ".vrma"),
                      src_up=(up or None), yaw=yaw, out_fps=fps, mode=mode,
                      with_face=face, frames=frames)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"แปลงไม่สำเร็จ ({upload.filename}): {e}")
    finally:
        try:
            os.remove(src)
        except OSError:
            pass
    return dst


@app.post("/api/convert")
async def api_convert(file: UploadFile = File(...),
                      up: str = Form(""),
                      yaw: str = Form("0"),
                      fps: str = Form("30"),
                      mode: str = Form("resample"),
                      face: str = Form("false"),
                      frames: str = Form("")):
    args = _clean(up, yaw, fps, mode, face, frames)
    dst = _run(file, await file.read(), *args)
    name = os.path.splitext(os.path.basename(file.filename or "motion"))[0] + ".vrma"
    return FileResponse(dst, media_type="model/gltf-binary", filename=name)


@app.post("/api/convert-batch")
async def api_convert_batch(files: List[UploadFile] = File(...),
                            up: str = Form(""),
                            yaw: str = Form("0"),
                            fps: str = Form("30"),
                            mode: str = Form("resample"),
                            face: str = Form("false"),
                            frames: str = Form("")):
    if not files:
        raise HTTPException(400, "ไม่มีไฟล์ที่อัปโหลด")

    args = _clean(up, yaw, fps, mode, face, frames)
    mem, errors, ok = io.BytesIO(), [], 0

    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            try:
                dst = _run(f, await f.read(), *args)
                zf.write(dst, os.path.splitext(os.path.basename(f.filename))[0] + ".vrma")
                ok += 1
            except HTTPException as e:
                errors.append(f"{f.filename}: {e.detail}")
        if errors:
            zf.writestr("_errors.txt", "\n".join(errors))
        if ok == 0:
            raise HTTPException(422, "แปลงไม่สำเร็จทุกไฟล์:\n" + "\n".join(errors))

    mem.seek(0)
    return StreamingResponse(
        mem, media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="vrma_batch.zip"'})


@app.get("/api/health")
def health():
    return {"status": "ok", "frames": config.TARGET_FRAMES, "version": "1.1.1"}


@app.post("/api/cleanup")
def cleanup():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    return {"status": "cleaned"}


app.mount("/web", StaticFiles(directory=WEB, html=True), name="web")


@app.get("/")
def root():
    return RedirectResponse("/web/index.html")