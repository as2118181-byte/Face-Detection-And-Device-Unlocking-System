import os
from pathlib import Path
import random
import cv2
import numpy as np
from facetools import FaceDetection, IdentityVerification, LivenessDetection
# Paths
root = Path(os.path.abspath(__file__)).parent.absolute()
data_folder = root / "data"
resNet_checkpoint_path = data_folder / "checkpoints" / "InceptionResnetV1_vggface2.onnx"
facebank_path = data_folder / "face_database.csv"
deepPix_checkpoint_path = data_folder / "checkpoints" / "OULU_Protocol_2_model_0_0.onnx"
# Decision Thresholds
LIVENESS_THRESHOLD = 0.03
IDENTITY_THRESHOLD = 0.85
# Load Models
faceDetector = FaceDetection(max_num_faces=1)
identityChecker = IdentityVerification(
    checkpoint_path=resNet_checkpoint_path.as_posix(),
    facebank_path=facebank_path.as_posix(),
)
livenessDetector = LivenessDetection(
    checkpoint_path=deepPix_checkpoint_path.as_posix()
)
# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    canvas = frame.copy()
    faces, boxes = faceDetector(frame)
    for face_arr, box in zip(faces, boxes):
        # Raw scores
        min_sim_score, mean_sim_score, person_name = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)
        x1, y1 = map(int, box[0])
        x2, y2 = map(int, box[1])
        # Decision + Display Confidence (EXACT ranges you requested)
        is_live = liveness_score > LIVENESS_THRESHOLD
        is_authentic = mean_sim_score < IDENTITY_THRESHOLD
        if is_live and is_authentic:
            # Authentic + Live → confidence strictly in [80, 95]
            # Slight variation based on how good the match is (lower distance → higher conf)
            base = 80.0 + (1.0 - (mean_sim_score / IDENTITY_THRESHOLD)) * 15.0
            confidence = float(np.clip(base, 80.0, 95.0))
            status_text = "Liveliness Detected"
            status_color = (0, 255, 0)
            box_color = (0, 255, 0)
            result_text = f"Device Unlocked - {person_name}"
            result_color = (0, 255, 0)
        elif not is_live:
            # Spoof → confidence strictly in [5, 10]
            confidence = float(np.clip(5.0 + random.uniform(0, 5), 5.0, 10.0))
            status_text = "Spoof Detected"
            status_color = (0, 0, 255)
            box_color = (0, 0, 255)
            result_text = "Authentication Failed"
            result_color = (0, 0, 255)
        else:
            # Live but unknown person
            confidence = float(np.clip(20.0 + (1.0 - mean_sim_score) * 30.0, 15.0, 45.0))
            status_text = "Liveliness Detected"
            status_color = (0, 255, 255)
            box_color = (0, 255, 255)
            result_text = "ACCESS DENIED - Unknown"
            result_color = (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 3)
        panel_width = 340
        panel_height = 110
        panel_x1 = x1
        panel_y1 = max(0, y1 - panel_height - 10)
        panel_x2 = panel_x1 + panel_width
        panel_y2 = panel_y1 + panel_height

        cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (30, 30, 30), -1)

        cv2.putText(canvas, status_text, (panel_x1 + 10, panel_y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.putText(canvas, result_text, (panel_x1 + 10, panel_y1 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, result_color, 2)

        cv2.putText(canvas, f"Confidence: {confidence:0.1f}%", (panel_x1 + 10, panel_y1 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("AI Secure Face Authentication", canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()