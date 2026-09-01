import os

HOST = os.environ.get("VRMA_HOST", "127.0.0.1")
PORT = int(os.environ.get("VRMA_PORT", "8787"))     # ← เปลี่ยนพอร์ตที่นี่
OPEN_BROWSER = os.environ.get("VRMA_OPEN", "1") == "1"
TARGET_FRAMES = int(os.environ.get("VRMA_FRAMES", "180"))
