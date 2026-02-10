import cv2
import numpy as np
import onnxruntime as ort
from typing import List, cast


class InsightFaceEmbedder:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, face: np.ndarray) -> np.ndarray:
        face = cv2.resize(face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0
        face = (face - 0.5) / 0.5

        # NHWC: (1, 112, 112, 3)
        return np.expand_dims(face, axis=0)

    def get_embedding(self, face: np.ndarray) -> np.ndarray:
        blob = self.preprocess(face)

        outputs = cast(
            List[np.ndarray],
            self.session.run(
                [self.output_name],
                {self.input_name: blob},
            ),
        )

        embedding = outputs[0][0]
        embedding /= np.linalg.norm(embedding)

        return embedding.astype("float32")
