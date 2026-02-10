from flask import Flask, Response, render_template
import cv2
import numpy as np
import os
import atexit
from ultralytics import YOLO
from collections import defaultdict, deque

from app.recognition.insightface_embedder import InsightFaceEmbedder
from app.recognition.face_vector_db import FaceVectorDB
from app.recognition.face_metadata_db import FaceMetadataDB

app = Flask(__name__, template_folder="../templates")

# models
person_model = YOLO("models/yolo11n.onnx", task="detect")
face_model = YOLO("models/yolov8n-face.onnx", task="detect")
pose_model = YOLO("models/yolo11n-pose.onnx", task="pose")
weapon_model = YOLO("models/weapon.onnx", task="detect")
emotion_model = YOLO("models/emotion_better.onnx", task="classify")

# recognition
embedder = InsightFaceEmbedder("models/insightface.onnx")
vector_db = FaceVectorDB()
metadata_db = FaceMetadataDB()

# runtime paths
WATCHLIST_DIR = "data/watchlist"
FACES_DIR = "data/faces"
DB_DIR = "data/databases"

os.makedirs(WATCHLIST_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# globals
PERSON_COUNTER = 0
frame_count = 0
flagged_people = set()

identity_memory = {}
embedding_history = defaultdict(lambda: deque(maxlen=10))

last_weapon_boxes = []
last_emotions = {}


# camera
def create_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("camera failed")
    return cap


cap = create_camera()


def release_camera():
    cap.release()


atexit.register(release_camera)


# helpers
def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = clamp(int(x1), 0, w - 1)
    x2 = clamp(int(x2), 0, w - 1)
    y1 = clamp(int(y1), 0, h - 1)
    y2 = clamp(int(y2), 0, h - 1)
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def get_boxes(result):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    return result.boxes.xyxy.cpu().numpy()


def is_blurry(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 80


def preprocess_emotion(face):
    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face


def stabilize_identity(embedding, threshold=0.65):
    best_id = None
    best_score = 0
    for pid, avg_emb in identity_memory.items():
        score = np.dot(embedding, avg_emb)
        if score > best_score and score > threshold:
            best_score = score
            best_id = pid
    return best_id


def suspicion_score(emotion=None, weapon=False):
    score = 0
    if weapon:
        score += 5
    if emotion in ["angry", "fear"]:
        score += 2
    return score


def save_watchlist(face, frame, person_id):
    folder = f"{WATCHLIST_DIR}/{person_id}"
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(f"{folder}/face.jpg", face)
    cv2.imwrite(f"{folder}/scene.jpg", frame)


# streaming pipeline
def generate_frames():
    global PERSON_COUNTER, frame_count, last_weapon_boxes

    while True:
        try:
            ok, frame = cap.read()
            if not ok:
                continue

            annotated = frame.copy()
            frame_count += 1

            # pose
            if frame_count % 4 == 0:
                pose_img = pose_model(frame.copy(), verbose=False)[0].plot()
                if pose_img.dtype != np.uint8:
                    pose_img = (pose_img * 255).astype(np.uint8)
                if pose_img.shape == annotated.shape:
                    annotated = pose_img

            # weapon detection
            if frame_count % 6 == 0:
                wres = weapon_model(frame.copy(), conf=0.4, verbose=False)[0]
                last_weapon_boxes = get_boxes(wres)

            weapon_detected = len(last_weapon_boxes) > 0

            for box in last_weapon_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    annotated,
                    "WEAPON",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # fast person detection
            small = cv2.resize(frame, (640, 360))
            sx = frame.shape[1] / 640
            sy = frame.shape[0] / 360

            pres = person_model(small.copy(), conf=0.25, verbose=False)[0]

            for box in get_boxes(pres):
                x1 = int(box[0] * sx)
                y1 = int(box[1] * sy)
                x2 = int(box[2] * sx)
                y2 = int(box[3] * sy)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if frame_count % 2 != 0:
                    continue

                person_crop = safe_crop(frame, x1, y1, x2, y2)
                if person_crop is None:
                    continue

                fres = face_model(person_crop.copy(), conf=0.3, verbose=False)[0]

                for fbox in get_boxes(fres):
                    fx1, fy1, fx2, fy2 = map(int, fbox)
                    face = safe_crop(person_crop, fx1, fy1, fx2, fy2)
                    if face is None or is_blurry(face):
                        continue

                    embedding = embedder.get_embedding(face.copy())
                    person_id = stabilize_identity(embedding)

                    if person_id is None:
                        PERSON_COUNTER += 1
                        person_id = f"person_{PERSON_COUNTER}"

                    embedding_history[person_id].append(embedding)
                    identity_memory[person_id] = np.mean(
                        embedding_history[person_id], axis=0
                    )

                    emotion = last_emotions.get(person_id)

                    if frame_count % 3 == 0:
                        blob = preprocess_emotion(face.copy())
                        eres = emotion_model(blob, verbose=False)[0]
                        if eres.probs is not None:
                            emotion = emotion_model.names[int(eres.probs.top1)]
                            last_emotions[person_id] = emotion

                    gx1, gy1 = x1 + fx1, y1 + fy1
                    gx2, gy2 = x1 + fx2, y1 + fy2

                    cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
                    cv2.putText(
                        annotated,
                        f"{person_id} {emotion or ''}",
                        (gx1, gy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

                    score = suspicion_score(emotion, weapon_detected)

                    if score >= 4 and person_id not in flagged_people:
                        flagged_people.add(person_id)
                        save_watchlist(face, frame, person_id)

            if annotated.dtype != np.uint8:
                annotated = np.clip(annotated, 0, 255).astype(np.uint8)

            if annotated.ndim != 3 or annotated.shape[0] < 10:
                continue

            ret, buf = cv2.imencode(".jpg", annotated)
            if not ret:
                continue

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )

        except Exception as e:
            print("stream error:", e)
            continue


# routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
