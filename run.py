"""ตัวรันเซิร์ฟเวอร์: python run.py (ปกติเรียกผ่าน run.bat / run.sh)"""
import socket
import sys
import threading
import webbrowser

import config


def find_free_port(host: str, start: int, tries: int = 50) -> int:
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"หาพอร์ตว่างไม่ได้ในช่วง {start}-{start + tries}")


def main() -> None:
    import uvicorn

    port = find_free_port(config.HOST, config.PORT)
    url = f"http://{config.HOST}:{port}/"

    if port != config.PORT:
        print(f"[!] พอร์ต {config.PORT} ถูกใช้อยู่ -> เปลี่ยนไปใช้ {port}")

    print("=" * 62)
    print("  VRMA Converter  (.npz / .bvh  ->  .vrma, 180 frames)")
    print(f"  หน้าแปลงไฟล์ : {url}")
    print(f"  หน้าพรีวิว 3D: {url}web/preview.html")
    print(f"  API endpoint : {url}api/convert")
    print("  กด Ctrl+C เพื่อหยุดเซิร์ฟเวอร์")
    print("=" * 62)

    if config.OPEN_BROWSER:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    try:
        uvicorn.run("server.app:app", host=config.HOST, port=port, log_level="info")
    except KeyboardInterrupt:
        print("\nปิดเซิร์ฟเวอร์แล้ว")
        sys.exit(0)


if __name__ == "__main__":
    main()