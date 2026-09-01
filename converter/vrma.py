import json
import struct

import numpy as np

from .skeleton import BONES, OFFSET, PARENT


class _Buffer:
    def __init__(self):
        self.bin = bytearray()
        self.views = []
        self.accessors = []

    def add(self, arr, type_str, minmax=False):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        while len(self.bin) % 4:
            self.bin.append(0)
        offset = len(self.bin)
        self.bin.extend(arr.tobytes())
        self.views.append({"buffer": 0, "byteOffset": offset, "byteLength": int(arr.nbytes)})
        acc = {"bufferView": len(self.views) - 1, "componentType": 5126,
               "count": int(arr.shape[0]), "type": type_str}
        if minmax:
            flat = arr.reshape(arr.shape[0], -1)
            acc["min"] = [float(v) for v in flat.min(axis=0)]
            acc["max"] = [float(v) for v in flat.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def write_vrma(path, times, tracks, hips_track, title="motion"):
    buf = _Buffer()
    used = [b for b in BONES if b in tracks]
    if "hips" not in used:
        raise ValueError("ไม่มีแทร็กของ hips จึงเขียน .vrma ไม่ได้")

    nodes, nidx = [], {}
    for bone in used:
        nidx[bone] = len(nodes)
        nodes.append({"name": bone,
                      "translation": [float(v) for v in OFFSET[bone]],
                      "rotation": [0.0, 0.0, 0.0, 1.0],
                      "scale": [1.0, 1.0, 1.0]})

    children = set()
    for bone in used:
        p = PARENT[bone]
        while p is not None and p not in nidx:
            p = PARENT[p]
        if p is not None:
            nodes[nidx[p]].setdefault("children", []).append(nidx[bone])
            children.add(nidx[bone])
    roots = [nidx[b] for b in used if nidx[b] not in children]

    t_acc = buf.add(np.asarray(times).reshape(-1), "SCALAR", minmax=True)
    samplers, channels = [], []
    for bone in used:
        out = buf.add(tracks[bone], "VEC4")
        samplers.append({"input": t_acc, "output": out, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": nidx[bone], "path": "rotation"}})

    out = buf.add(hips_track, "VEC3")
    samplers.append({"input": t_acc, "output": out, "interpolation": "LINEAR"})
    channels.append({"sampler": len(samplers) - 1,
                     "target": {"node": nidx["hips"], "path": "translation"}})

    gltf = {
        "asset": {"version": "2.0", "generator": "vrma-converter/1.1"},
        "extensionsUsed": ["VRMC_vrm_animation"],
        "scene": 0,
        "scenes": [{"nodes": roots}],
        "nodes": nodes,
        "buffers": [{"byteLength": len(buf.bin)}],
        "bufferViews": buf.views,
        "accessors": buf.accessors,
        "animations": [{"name": title, "samplers": samplers, "channels": channels}],
        "extensions": {
            "VRMC_vrm_animation": {
                "specVersion": "1.0",
                "humanoid": {"humanBones": {b: {"node": nidx[b]} for b in used}},
            }
        },
    }

    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    bn = bytes(buf.bin)
    bn += b"\x00" * ((4 - len(bn) % 4) % 4)

    glb = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bn))
    glb += struct.pack("<II", len(js), 0x4E4F534A) + js
    glb += struct.pack("<II", len(bn), 0x004E4942) + bn

    with open(path, "wb") as f:
        f.write(glb)
    return path