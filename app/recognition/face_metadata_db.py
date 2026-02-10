import sqlite3
from datetime import datetime, UTC
import os

os.makedirs("data/databases", exist_ok=True)

DB_FILE = "data/databases/faces.db"


class FaceMetadataDB:
    def __init__(self):
        self.conn = sqlite3.connect(DB_FILE, check_same_thread=False)

        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            vector_id INTEGER PRIMARY KEY,
            person_id TEXT,
            angle TEXT,
            image_path TEXT,
            created_at TEXT
        )
        """)

        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_person
        ON faces(person_id)
        """)

        self.conn.commit()

    def add_face(self, vector_id, person_id, angle, image_path):
        self.conn.execute(
            """
        INSERT INTO faces (
            vector_id,
            person_id,
            angle,
            image_path,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                vector_id,
                person_id,
                angle,
                image_path,
                datetime.now(UTC).isoformat(),
            ),
        )

        self.conn.commit()

    def get_person_id(self, vector_id):
        cur = self.conn.execute(
            "SELECT person_id FROM faces WHERE vector_id=?",
            (vector_id,),
        )

        row = cur.fetchone()
        if row is None:
            raise RuntimeError("FAISS / SQLite desync detected")

        return row[0]

    def get_angles(self, person_id):
        cur = self.conn.execute(
            "SELECT angle FROM faces WHERE person_id=?",
            (person_id,),
        )

        return {r[0] for r in cur.fetchall()}
