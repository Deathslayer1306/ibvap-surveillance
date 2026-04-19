"""
Entry point for the Suspicious Activity Detection server.
Run: python run.py
"""
import uvicorn
from app.config import HOST, PORT

if __name__ == "__main__":
    print("=" * 60)
    print("  DRONE SURVEILLANCE — Suspicious Activity Detection")
    print(f"  Dashboard -> http://localhost:{PORT}")
    print("=" * 60)
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=1,
        log_level="info",
        access_log=True,
    )
