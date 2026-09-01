from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class Motion:
    """world_rot: {vrm_bone: (N,4) xyzw}  |  hips_pos: (N,3)  |  rest_dir: {bone: (3,)}"""
    world_rot: Dict[str, np.ndarray]
    hips_pos: np.ndarray
    rest_dir: Dict[str, np.ndarray] = field(default_factory=dict)
    fps: float = 30.0
    space: str = "src"   # "src" = ต้องแปลงแกน, "vrm" = อยู่ในสเปซ VRM แล้ว
    up: str = "y"        # แกนขึ้นของต้นทาง: "y" หรือ "z"
    source: str = ""