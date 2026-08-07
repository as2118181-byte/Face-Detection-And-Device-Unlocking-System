import os
from pathlib import Path

import cv2

from facetools import FaceDetection, IdentityVerification, LivenessDetection

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

        # Get scores
        min_sim_score, mean_sim_score = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)

        # Box coordinates
        x1, y1 = map(int, box[0])
        x2, y2 = map(int, box[1])

        # -------------------------------------------------
        # Status (NO CHANGE IN ORIGINAL LOGIC)
        # -------------------------------------------------

        if liveness_score > 0.03:

            status_text = "LIVE FACE"
            status_color = (0, 255, 0)
            box_color = (0, 255, 0)

            if mean_sim_score < 0.85:
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
        # Background Panel
        # -------------------------------------------------

        panel_width = 280
        panel_height = 70

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
        # Status Text
        # -------------------------------------------------

        cv2.putText(
            canvas,
            status_text,
            (panel_x1 + 10, panel_y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2,
        )

        cv2.putText(
            canvas,
            result_text,
            (panel_x1 + 10, panel_y1 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            result_color,
            2,
        )

    cv2.imshow("AI Secure Face Authentication", canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()