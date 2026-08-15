import os
from os import environ
from pathlib import Path
import cv2
import jsonpickle
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, request
from facetools import FaceDetection, IdentityVerification, LivenessDetection
from facetools.utils import compute_identity_confidence, compute_liveness_confidence
root = Path(os.path.abspath(__file__)).parent.absolute()
load_dotenv((root / ".env").as_posix())
data_folder = root.parent / environ.get("DATA_FOLDER")
face_recognition_model = environ.get("FACE_RECOGNITION_MODEL")
liveness_model = environ.get("LIVENESS_MODEL")
face_database = environ.get("FACE_DATABASE")
face_model_path = data_folder / "checkpoints" / face_recognition_model
liveness_model_path = data_folder / "checkpoints" / liveness_model
face_database_path = data_folder / face_database
LIVENESS_THRESHOLD = 0.15
IDENTITY_THRESHOLD = 0.85
face_detector = FaceDetection()

identity_verifier = IdentityVerification(
    checkpoint_path=face_model_path.as_posix(),
    facebank_path=face_database_path.as_posix(),
)

liveness_detector = LivenessDetection(
    checkpoint_path=liveness_model_path.as_posix()
)
app = Flask(__name__)

app.config["PROJECT_NAME"] = "AI Secure Face Authentication API"
app.config["VERSION"] = "1.0"
@app.route("/authenticate", methods=["POST"])
def authenticate():
    image_bytes = request.data

    image_array = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    faces, boxes = face_detector(frame)

    if len(faces) == 0:

        response = {
            "status": "failed",
            "message": "No face detected.",
            "authentication": False,
            "liveness_score": None,
            "similarity_score": None,
            "liveness_confidence": None,
            "identity_confidence": None,
        }

        status = 400

    else:

        face = faces[0]

        min_score, mean_score, _ = identity_verifier(face)

        live_score = liveness_detector(face)

        # Use min_score (best match distance) for identity decision
        authenticated = bool(live_score > LIVENESS_THRESHOLD and min_score < IDENTITY_THRESHOLD)

        response = {
            "status": "success",
            "message": "Authentication completed.",
            "authentication": authenticated,
            "liveness_score": float(live_score),
            "similarity_score": float(min_score),
            "liveness_confidence": compute_liveness_confidence(live_score),
            "identity_confidence": compute_identity_confidence(min_score, IDENTITY_THRESHOLD),
        }

        status = 200

    return Response(
        response=jsonpickle.encode(response),
        status=status,
        mimetype="application/json",
    )
@app.route("/main", methods=["POST"])
def recognition():

    image_bytes = request.data

    image_array = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    faces, boxes = face_detector(frame)

    if len(faces) == 0:

        response = {
            "status": "failed",
            "message": "No face detected.",
            "similarity_score": None,
            "identity_confidence": None,
        }

        status = 400

    else:

        face = faces[0]

        min_score, mean_score, _ = identity_verifier(face)

        response = {
            "status": "success",
            "message": "Face verification completed.",
            "similarity_score": float(min_score),
            "identity_confidence": compute_identity_confidence(min_score, IDENTITY_THRESHOLD),
        }

        status = 200

    return Response(
        response=jsonpickle.encode(response),
        status=status,
        mimetype="application/json",
    )
@app.route("/liveness", methods=["POST"])
def liveness():

    image_bytes = request.data

    image_array = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    faces, boxes = face_detector(frame)

    if len(faces) == 0:

        response = {
            "status": "failed",
            "message": "No face detected.",
            "liveness_score": None,
            "liveness_confidence": None,
        }

        status = 400

    else:

        face = faces[0]

        live_score = liveness_detector(face)

        response = {
            "status": "success",
            "message": "Liveness verification completed.",
            "liveness_score": float(live_score),
            "liveness_confidence": compute_liveness_confidence(live_score),
        }

        status = 200

    return Response(
        response=jsonpickle.encode(response),
        status=status,
        mimetype="application/json",
    )
if __name__ == "__main__":
    print("=" * 55)
    print("AI Secure Face Authentication API")
    print("Version : 1.0")
    print("Developed by Arun Sharma")
    print("Server running at http://127.0.0.1:5000")
    print("=" * 55)

    app.run(host="0.0.0.0", port=5000)