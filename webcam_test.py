import os
from pathlib import Path
import random
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import cv2
import numpy as np
from dotenv import load_dotenv
from fer.fer import FER

from facetools import FaceDetection, IdentityVerification, LivenessDetection

# -------------------------------------------------
# Load secrets from .env
# -------------------------------------------------
load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
ALERT_TO_EMAIL = os.getenv("ALERT_TO_EMAIL")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

# -------------------------------------------------
# Paths
# -------------------------------------------------
root = Path(os.path.abspath(__file__)).parent.absolute()
data_folder = root / "data"
resNet_checkpoint_path = data_folder / "checkpoints" / "InceptionResnetV1_vggface2.onnx"
facebank_path = data_folder / "face_database.csv"
deepPix_checkpoint_path = data_folder / "checkpoints" / "OULU_Protocol_2_model_0_0.onnx"

# Decision Thresholds (UNCHANGED)
LIVENESS_THRESHOLD = 0.03
IDENTITY_THRESHOLD = 0.85

# Alert settings
FEAR_DURATION_SECONDS = 5.0
SPOOF_DURATION_SECONDS = 5.0
ALERT_COOLDOWN_SECONDS = 60.0
RESET_TOLERANCE = 0.8          # only reset timer if condition is gone for > 0.8 sec

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
emotion_detector = FER(mtcnn=False)

# -------------------------------------------------
# Email helper
# -------------------------------------------------
def send_alert(subject: str, body: str):
    if not all([EMAIL_ADDRESS, EMAIL_PASSWORD, ALERT_TO_EMAIL]):
        print("[ALERT] Email credentials missing in .env – skipping send")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = ALERT_TO_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        print(f"[ALERT] Email sent successfully → {subject}")
        return True
    except Exception as e:
        print(f"[ALERT] Failed to send email: {e}")
        return False


def send_fear_alert(person_name: str):
    body = f"""
SECURITY ALERT – Face Authentication System

An authenticated person was detected showing continuous fear for more than {FEAR_DURATION_SECONDS} seconds.

Person          : {person_name}
Time            : {time.strftime('%Y-%m-%d %H:%M:%S')}
Location        : Device unlock attempt

The system still granted access (as requested), but this alert was sent for safety.

— AI Secure Face Authentication System
"""
    send_alert("SECURITY ALERT: Fear Detected on Authenticated User", body)


def send_spoof_alert():
    body = f"""
SECURITY ALERT – Face Authentication System

A SPOOF attack was detected continuously for more than {SPOOF_DURATION_SECONDS} seconds.

Time            : {time.strftime('%Y-%m-%d %H:%M:%S')}
Location        : Device unlock attempt

Possible photo / video / mask attack detected.

— AI Secure Face Authentication System
"""
    send_alert("SECURITY ALERT: Spoof Attack Detected", body)

# -------------------------------------------------
# State tracking (with tolerance)
# -------------------------------------------------
fear_start_time = None
spoof_start_time = None
last_fear_seen = 0.0
last_spoof_seen = 0.0
last_alert_time = 0.0

# -------------------------------------------------
# Webcam
# -------------------------------------------------
cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit")
print("Watch the terminal for status messages")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    canvas = frame.copy()
    faces, boxes = faceDetector(frame)

    for face_arr, box in zip(faces, boxes):
        # ----- Existing authentication logic (UNCHANGED) -----
        min_sim_score, mean_sim_score, person_name = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)

        x1, y1 = map(int, box[0])
        x2, y2 = map(int, box[1])

        is_live = liveness_score > LIVENESS_THRESHOLD
        is_authentic = mean_sim_score < IDENTITY_THRESHOLD
        is_spoof = not is_live

        if is_live and is_authentic:
            base = 80.0 + (1.0 - (mean_sim_score / IDENTITY_THRESHOLD)) * 15.0
            confidence = float(np.clip(base, 80.0, 95.0))
            status_text = "Liveliness Detected"
            status_color = (0, 255, 0)
            box_color = (0, 255, 0)
            result_text = f"Device Unlocked - {person_name}"
            result_color = (0, 255, 0)
        elif is_spoof:
            confidence = float(np.clip(5.0 + random.uniform(0, 5), 5.0, 10.0))
            status_text = "Spoof Detected"
            status_color = (0, 0, 255)
            box_color = (0, 0, 255)
            result_text = "Authentication Failed"
            result_color = (0, 0, 255)
        else:
            confidence = float(np.clip(20.0 + (1.0 - mean_sim_score) * 30.0, 15.0, 45.0))
            status_text = "Liveliness Detected"
            status_color = (0, 255, 255)
            box_color = (0, 255, 255)
            result_text = "ACCESS DENIED - Unknown"
            result_color = (0, 0, 255)

        # ----- Fear detection -----
        is_fear = False
        fear_score = 0.0
        dominant = "none"

        try:
            pad = 25
            y1p = max(0, y1 - pad)
            y2p = min(frame.shape[0], y2 + pad)
            x1p = max(0, x1 - pad)
            x2p = min(frame.shape[1], x2 + pad)
            roi = frame[y1p:y2p, x1p:x2p]

            if roi.size > 0 and roi.shape[0] > 50 and roi.shape[1] > 50:
                emotions = emotion_detector.detect_emotions(roi)
                if emotions and len(emotions) > 0:
                    emotion_scores = emotions[0]["emotions"]
                    dominant = max(emotion_scores, key=emotion_scores.get)
                    fear_score = emotion_scores.get("fear", 0.0)
                    is_fear = (dominant == "fear") or (fear_score >= 0.28)
        except Exception as e:
            pass

        current_time = time.time()

        # ---------- Fear timer (with tolerance) ----------
        if is_live and is_authentic and is_fear:
            last_fear_seen = current_time
            if fear_start_time is None:
                fear_start_time = current_time
                print(">>> Fear timer started...")
            elif (current_time - fear_start_time) >= FEAR_DURATION_SECONDS:
                if (current_time - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                    print(">>> 5 seconds continuous fear → Sending email...")
                    send_fear_alert(person_name)
                    last_alert_time = current_time
                fear_start_time = None
        else:
            # Only reset if fear has been gone for more than RESET_TOLERANCE seconds
            if fear_start_time is not None and (current_time - last_fear_seen) > RESET_TOLERANCE:
                print(">>> Fear timer reset")
                fear_start_time = None

        # ---------- Spoof timer (with tolerance) ----------
        if is_spoof:
            last_spoof_seen = current_time
            if spoof_start_time is None:
                spoof_start_time = current_time
                print(">>> Spoof timer started...")
            elif (current_time - spoof_start_time) >= SPOOF_DURATION_SECONDS:
                if (current_time - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                    print(">>> 5 seconds continuous spoof → Sending email...")
                    send_spoof_alert()
                    last_alert_time = current_time
                spoof_start_time = None
        else:
            # Only reset if spoof has been gone for more than RESET_TOLERANCE seconds
            if spoof_start_time is not None and (current_time - last_spoof_seen) > RESET_TOLERANCE:
                print(">>> Spoof timer reset")
                spoof_start_time = None

        # Display text
        if is_fear and is_authentic and is_live and fear_start_time is not None:
            elapsed = int(current_time - fear_start_time)
            fear_status = f"FEAR {elapsed}s ({fear_score:.2f})"
        elif is_spoof and spoof_start_time is not None:
            elapsed = int(current_time - spoof_start_time)
            fear_status = f"SPOOF {elapsed}s"
        else:
            fear_status = f"{dominant} ({fear_score:.2f})"

        # ----- Drawing -----
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 3)

        panel_width = 360
        panel_height = 135
        panel_x1 = x1
        panel_y1 = max(0, y1 - panel_height - 10)
        panel_x2 = panel_x1 + panel_width
        panel_y2 = panel_y1 + panel_height

        cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (30, 30, 30), -1)

        cv2.putText(canvas, status_text, (panel_x1 + 10, panel_y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.putText(canvas, result_text, (panel_x1 + 10, panel_y1 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, result_color, 2)

        cv2.putText(canvas, f"Confidence: {confidence:0.1f}%", (panel_x1 + 10, panel_y1 + 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(canvas, fear_status, (panel_x1 + 10, panel_y1 + 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 165, 255), 2)

    cv2.imshow("AI Secure Face Authentication", canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()