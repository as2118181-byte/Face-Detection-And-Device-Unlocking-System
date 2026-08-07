import os
from pathlib import Path

import cv2

from facetools import FaceDetection, IdentityVerification, LivenessDetection
from facetools.utils import compute_identity_confidence, compute_liveness_confidence

# -------------------------------------------------
# Paths
# -------------------------------------------------

root = Path(os.path.abspath(__file__)).parent.absolute()
data_folder = root / "data"

resNet_checkpoint_path = (
    data_folder / "checkpoints" / "InceptionResnetV1_vggface2.onnx"
)
facebank_path = data_folder / "face_database.csv"
deepPix_checkpoint_path = (
    data_folder / "checkpoints" / "OULU_Protocol_2_model_0_0.onnx"
)

# -------------------------------------------------
# Decision Thresholds (unchanged from original logic)
# -------------------------------------------------

LIVENESS_THRESHOLD = 0.03
IDENTITY_THRESHOLD = 0.85

# -------------------------------------------------
# Load Models
# -------------------------------------------------

faceDetector = FaceDetection(max_num_faces=1)

identityChecker = IdentityVerification(
    checkpoint_path=resNet_checkpoint_path.as_posix(),
    facebank_path=facebank_path.as_posix(),
)

livenessDetector = LivenessDetection(
    checkpoint_path=deepPix_checkpoint_path.as_posix()
)

# -------------------------------------------------
# Webcam
# -------------------------------------------------

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    canvas = frame.copy()

    faces, boxes = faceDetector(frame)

    for face_arr, box in zip(faces, boxes):

        # Get raw scores (unchanged)
        min_sim_score, mean_sim_score = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)

        # Derived confidence percentages (display-only, does not affect decisions)
        identity_confidence = compute_identity_confidence(mean_sim_score, IDENTITY_THRESHOLD)
        liveness_confidence = compute_liveness_confidence(liveness_score)

        # Box coordinates
        x1, y1 = map(int, box[0])
        x2, y2 = map(int, box[1])

        # -------------------------------------------------
        # Status (NO CHANGE IN ORIGINAL DECISION LOGIC)
        # -------------------------------------------------

        if liveness_score > LIVENESS_THRESHOLD:

            status_text = "LIVE FACE"
            status_color = (0, 255, 0)
            box_color = (0, 255, 0)

            if mean_sim_score < IDENTITY_THRESHOLD:
                result_text = "Authentication Successful"
                result_color = (0, 255, 0)
            else:
                result_text = "ACCESS DENIED"
                result_color = (0, 0, 255)

        else:

            status_text = "Presentation Attack Detected"
            status_color = (0, 0, 255)
            box_color = (0, 0, 255)

            result_text = "Authentication Failed"
            result_color = (0, 0, 255)

        # -------------------------------------------------
        # Draw Face Box
        # -------------------------------------------------

        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 3)

        # -------------------------------------------------
        # Background Panel (taller now, to fit confidence lines)
        # -------------------------------------------------

        panel_width = 300
        panel_height = 120

        panel_x1 = x1
        panel_y1 = max(0, y1 - panel_height - 10)
        panel_x2 = panel_x1 + panel_width
        panel_y2 = panel_y1 + panel_height

        cv2.rectangle(
            canvas,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            (40, 40, 40),
            -1,
        )

        # -------------------------------------------------
        # Status / Result Text
        # -------------------------------------------------

        cv2.putText(
            canvas,
            status_text,
            (panel_x1 + 10, panel_y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
        )

        cv2.putText(
            canvas,
            result_text,
            (panel_x1 + 10, panel_y1 + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            result_color,
            2,
        )

        # -------------------------------------------------
        # Confidence Score Text
        # -------------------------------------------------

        cv2.putText(
            canvas,
            f"Liveness Confidence: {liveness_confidence:0.1f}%",
            (panel_x1 + 10, panel_y1 + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            canvas,
            f"Match Confidence: {identity_confidence:0.1f}%",
            (panel_x1 + 10, panel_y1 + 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    cv2.imshow("AI Secure Face Authentication", canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()