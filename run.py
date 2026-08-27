"""
Entry point for IBVAP — Intelligent Border Video Analytics Platform.
Run: python run.py
Stop: Ctrl+C  (uvicorn handles graceful shutdown)
"""
import uvicorn
from app.config import HOST, PORT

if __name__ == "__main__":
    print("=" * 62)
    print("  IBVAP — Intelligent Border Video Analytics Platform")
    print("  SIH 2026 · PS:SIH26187")
    print(f"  Dashboard   → http://localhost:{PORT}")
    print(f"  Metrics     → http://localhost:{PORT}/metrics")
    print(f"  API Cameras → http://localhost:{PORT}/api/cameras")
    print("  Stop        → Ctrl+C")
    print("=" * 62)
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=1,
        log_level="info",
        access_log=False,   # suppress per-frame MJPEG GET spam
    )
