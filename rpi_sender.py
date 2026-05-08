"""
rpi_sender.py — Run this on the Raspberry Pi.

Captures frames from the Pi Camera Module (via Picamera2) and streams them
to the laptop dashboard over a TCP socket using pickle + struct framing.

Usage on the Pi:
    python rpi_sender.py
    python rpi_sender.py 10.156.86.179 8080   # override IP/port via CLI

Dependencies (install on Pi):
    pip install picamera2 opencv-python-headless
"""

import sys
import time
import socket
import pickle
import struct

import cv2

# ─────────────────────────────────────────────────────────────────────────────
# TOGGLE: Comment/uncomment ONE of the two import + capture blocks below
# to switch between the Pi Camera Module and a USB webcam.
# ─────────────────────────────────────────────────────────────────────────────

# ── Option A: Pi Camera Module (Picamera2) ───────────────────────────────────
# Use this when the official Pi Camera ribbon cable is connected.
from picamera2 import Picamera2           # ← Comment this line for USB webcam
USE_PICAMERA = True                       # ← Change to False for USB webcam

# ── Option B: USB / V4L2 webcam ──────────────────────────────────────────────
# Use this when RPi is unavailable or you want a plain USB camera instead.
# Uncomment the line below AND set USE_PICAMERA = False above.
# USB_CAMERA_INDEX = 0                    # 0 = first USB cam, 1 = second, etc.


# ─── Network ──────────────────────────────────────────────────────────────────
LAPTOP_IP   = "10.156.86.179"   # ← Laptop LAN IP (server side)
LAPTOP_PORT = 8080              # must match RPI_PORT in app/config.py

# ─── Frame settings ───────────────────────────────────────────────────────────
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
JPEG_QUALITY  = 60    # pre-compress before pickling to save bandwidth (1-100)
                      # set to 0 to send raw BGR (higher quality, more bandwidth)

RECONNECT_DELAY = 3.0   # seconds between reconnect attempts

# ─── Protocol (must match server side in main.py) ─────────────────────────────
PAYLOAD_FMT = "!I"  # network byte order uint32 — always 4 bytes on all platforms


def _encode_frame(frame):
    """Optionally JPEG-compress the frame before pickling to save bandwidth."""
    if JPEG_QUALITY > 0:
        ret, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if ret:
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return pickle.dumps(frame)


def stream(laptop_ip: str, laptop_port: int) -> None:
    # ── Camera init ───────────────────────────────────────────────────────────
    if USE_PICAMERA:
        # ── Pi Camera Module (Picamera2) ──────────────────────────────────────
        print("[Sender] Starting Pi Camera (Picamera2)...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)   # warm-up
        print("[Sender] Pi Camera ready!")

        def read_frame():
            return picam2.capture_array()

    else:
        # ── USB / V4L2 webcam ─────────────────────────────────────────────────
        # To switch to USB cam: set USE_PICAMERA = False at the top.
        print("[Sender] Opening USB webcam...")
        cap = cv2.VideoCapture(USB_CAMERA_INDEX)  # noqa: F821
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not cap.isOpened():
            print("[Sender] ERROR: Cannot open USB webcam.")
            return
        print("[Sender] USB webcam ready!")

        def read_frame():
            ok, frame = cap.read()
            return frame if ok else None

    # ── Connect & stream loop ─────────────────────────────────────────────────
    attempt = 0
    while True:
        attempt += 1
        print(f"[Sender] Connecting to {laptop_ip}:{laptop_port} (attempt {attempt})")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((laptop_ip, laptop_port))
        except (ConnectionRefusedError, OSError) as e:
            print(f"[Sender] Connection failed: {e}. Retrying in {RECONNECT_DELAY}s")
            time.sleep(RECONNECT_DELAY)
            continue

        print("[Sender] Connected. Streaming ...")
        attempt = 0

        try:
            while True:
                frame = read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                payload  = _encode_frame(frame)
                msg_size = struct.pack(PAYLOAD_FMT, len(payload))
                sock.sendall(msg_size + payload)   # atomic send: length + data

        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"[Sender] Connection lost: {e}")
        finally:
            sock.close()

        print(f"[Sender] Reconnecting in {RECONNECT_DELAY}s ...")
        time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    ip   = sys.argv[1] if len(sys.argv) > 1 else LAPTOP_IP
    port = int(sys.argv[2]) if len(sys.argv) > 2 else LAPTOP_PORT
    try:
        stream(ip, port)
    except KeyboardInterrupt:
        print("\n[Sender] Stopped.")
