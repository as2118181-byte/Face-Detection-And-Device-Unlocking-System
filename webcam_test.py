import os
from pathlib import Path
import random
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from collections import deque
import json
import subprocess
import threading

import cv2
import numpy as np
from dotenv import load_dotenv
from fer.fer import FER
from flask import Flask, Response
import jsonpickle

from facetools import FaceDetection, IdentityVerification, LivenessDetection


def play_alarm(alarm_type="generic"):
    try:
        subprocess.Popen(["alarm.exe", alarm_type], shell=True)
    except Exception as e:
        print(f"Could not play alarm: {e}")


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

spoof_folder = root / "spoofs_detected"
spoof_folder.mkdir(exist_ok=True)
unknown_folder = root / "unknown_detected"
unknown_folder.mkdir(exist_ok=True)

STATUS_FILE = data_folder / "device_status.json"


def write_device_status(status_dict):
    status_dict["pc_online"] = True
    status_dict["timestamp"] = datetime.now().isoformat()
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_dict, f, indent=2)
    except Exception as e:
        print(f"[STATUS] write failed: {e}")


# -------------------------------------------------
# Embedded Flask server (so only one process is needed)
# -------------------------------------------------
flask_app = Flask(__name__)

@flask_app.route("/api/device-status", methods=["GET"])
def device_status():
    if not STATUS_FILE.exists():
        return Response(
            response=jsonpickle.encode({
                "status": "locked",
                "authentication": False,
                "user": None,
                "identity_confidence": 0.0,
                "liveness_confidence": 0.0,
                "liveness_score": 0.0,
                "similarity_score": 0.0,
                "message": "No status available yet",
                "pc_online": True,
                "timestamp": datetime.now().isoformat(),
                "failed_attempts": 0,
                "max_failed_attempts": 4,
            }),
            status=200,
            mimetype="application/json",
        )
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["pc_online"] = True
        return Response(
            response=jsonpickle.encode(data),
            status=200,
            mimetype="application/json",
        )
    except Exception as e:
        return Response(
            response=jsonpickle.encode({"error": str(e), "pc_online": False}),
            status=500,
            mimetype="application/json",
        )


def start_flask_server():
    # Runs in background thread
    flask_app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


# Decision Thresholds
LIVENESS_THRESHOLD = 0.12
IDENTITY_THRESHOLD = 0.85

FEAR_DURATION_SECONDS = 5.0
SPOOF_DURATION_SECONDS = 5.0
UNKNOWN_DURATION_SECONDS = 5.0
ALERT_COOLDOWN_SECONDS = 60.0
RESET_TOLERANCE = 0.8

MAX_FAILED_ATTEMPTS = 4
LOCK_DURATION_SECONDS = 5 * 60

HISTORY_LEN = 8
live_history = deque(maxlen=HISTORY_LEN)
auth_history = deque(maxlen=HISTORY_LEN)

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
# Email helpers
# -------------------------------------------------
def send_alert(subject: str, body: str):
    if not all([EMAIL_ADDRESS, EMAIL_PASSWORD, ALERT_TO_EMAIL]):
        print("[ALERT] Email credentials missing in .env")
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
        print(f"[ALERT] Email sent → {subject}")
        return True
    except Exception as e:
        print(f"[ALERT] Failed to send email: {e}")
        return False


def send_fear_alert(person_name: str):
    attack_time = time.strftime('%Y-%m-%d %H:%M:%S')
    body = f"""
SECURITY ALERT – Face Authentication System

An authenticated person showed continuous fear for more than {FEAR_DURATION_SECONDS} seconds.

Person      : {person_name}
Time of Attack : {attack_time}

— AI Secure Face Authentication System
"""
    send_alert("SECURITY ALERT: Fear Detected on Authenticated User", body)


def send_spoof_alert():
    attack_time = time.strftime('%Y-%m-%d %H:%M:%S')
    body = f"""
SECURITY ALERT – Face Authentication System

A SPOOF attack was detected continuously for more than {SPOOF_DURATION_SECONDS} seconds.
A photo of the spoof has also been saved.

Time of Attack : {attack_time}

— AI Secure Face Authentication System
"""
    send_alert("SECURITY ALERT: Spoof Attack Detected", body)


def send_unknown_alert():
    attack_time = time.strftime('%Y-%m-%d %H:%M:%S')
    body = f"""
SECURITY ALERT – Face Authentication System

An UNKNOWN / UNAUTHORIZED live person was continuously present for more than {UNKNOWN_DURATION_SECONDS} seconds.
A photo has been saved.

Time of Attack : {attack_time}

— AI Secure Face Authentication System
"""
    send_alert("SECURITY ALERT: Unauthorized Person Detected", body)


# -------------------------------------------------
# State
# -------------------------------------------------
fear_start_time = None
spoof_start_time = None
unknown_start_time = None
last_fear_seen = 0.0
last_spoof_seen = 0.0
last_unknown_seen = 0.0
last_alert_time = 0.0

failed_attempts = 0
lock_until = 0.0

# -------------------------------------------------
# Start Flask server in background
# -------------------------------------------------
print("=" * 55)
print("AI Secure Face Authentication + Dashboard Server")
print("Starting embedded Flask server on port 5000...")
print("=" * 55)

flask_thread = threading.Thread(target=start_flask_server, daemon=True)
flask_thread.start()

time.sleep(1.5)  # give Flask a moment to start

# -------------------------------------------------
# Webcam
# -------------------------------------------------
cap = cv2.VideoCapture(0)

print("System started... Press 'q' to quit")
print(f"Spoof photos  → {spoof_folder}")
print(f"Unknown photos → {unknown_folder}")
print("Flutter can now connect to http://127.0.0.1:5000/api/device-status")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    canvas = frame.copy()
    current_time = time.time()

    is_locked = current_time < lock_until
    remaining_lock = int(lock_until - current_time) if is_locked else 0

    faces, boxes = faceDetector(frame)

    if len(faces) == 0:
        write_device_status({
            "status": "locked",
            "authentication": False,
            "user": None,
            "identity_confidence": 0.0,
            "liveness_confidence": 0.0,
            "liveness_score": 0.0,
            "similarity_score": 0.0,
            "message": "No face detected",
            "failed_attempts": failed_attempts,
            "max_failed_attempts": MAX_FAILED_ATTEMPTS,
        })

    for face_arr, box in zip(faces, boxes):
        min_sim_score, mean_sim_score, person_name = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)

        x1, y1 = map(int, box[0])
        x2, y2 = map(int, box[1])

        raw_live = liveness_score > LIVENESS_THRESHOLD
        raw_auth = min_sim_score < IDENTITY_THRESHOLD

        live_history.append(raw_live)
        auth_history.append(raw_auth)

        is_live = sum(live_history) >= (HISTORY_LEN // 2 + 1)
        is_authentic = sum(auth_history) >= (HISTORY_LEN // 2 + 1)

        is_spoof = not is_live
        is_unknown = is_live and (not is_authentic)

        if is_locked:
            confidence = 0.0
            status_text = "SYSTEM LOCKED"
            status_color = (0, 0, 255)
            box_color = (0, 0, 255)
            result_text = f"Try after {remaining_lock}s"
            result_color = (0, 0, 255)
        elif is_live and is_authentic:
            base = 80.0 + (1.0 - (min_sim_score / IDENTITY_THRESHOLD)) * 15.0
            confidence = float(np.clip(base, 80.0, 95.0))
            status_text = "Liveliness Detected"
            status_color = (0, 255, 0)
            box_color = (0, 255, 0)
            result_text = f"Device Unlocked - {person_name}"
            result_color = (0, 255, 0)
            failed_attempts = 0
        elif is_spoof:
            confidence = float(np.clip(5.0 + random.uniform(0, 5), 5.0, 10.0))
            status_text = "Spoof Detected"
            status_color = (0, 0, 255)
            box_color = (0, 0, 255)
            result_text = "Authentication Failed"
            result_color = (0, 0, 255)
        else:
            confidence = float(np.clip(20.0 + (1.0 - min_sim_score) * 30.0, 15.0, 45.0))
            status_text = "Liveliness Detected"
            status_color = (0, 255, 255)
            box_color = (0, 255, 255)
            result_text = "ACCESS DENIED - Unknown"
            result_color = (0, 0, 255)

        # Fear detection
        is_fear = False
        fear_score = 0.0
        dominant = "none"
        try:
            pad = 25
            roi = frame[max(0, y1-pad):min(frame.shape[0], y2+pad),
                        max(0, x1-pad):min(frame.shape[1], x2+pad)]
            if roi.size > 0 and roi.shape[0] > 50:
                emotions = emotion_detector.detect_emotions(roi)
                if emotions:
                    scores = emotions[0]["emotions"]
                    dominant = max(scores, key=scores.get)
                    fear_score = scores.get("fear", 0.0)
                    is_fear = (dominant == "fear") or (fear_score >= 0.28)
        except:
            pass

        # Fear logic
        if is_live and is_authentic and is_fear and not is_locked:
            last_fear_seen = current_time
            if fear_start_time is None:
                fear_start_time = current_time
                print(">>> Fear timer started...")
            elif (current_time - fear_start_time) >= FEAR_DURATION_SECONDS:
                if (current_time - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                    print(">>> 5s Fear → Sending Email...")
                    play_alarm("fear")
                    send_fear_alert(person_name)
                    last_alert_time = current_time
                fear_start_time = None
        else:
            if fear_start_time and (current_time - last_fear_seen) > RESET_TOLERANCE:
                fear_start_time = None

        # Spoof logic
        if is_spoof and not is_locked:
            last_spoof_seen = current_time
            if spoof_start_time is None:
                spoof_start_time = current_time
                print(">>> Spoof timer started...")
            elif (current_time - spoof_start_time) >= SPOOF_DURATION_SECONDS:
                print(">>> 5s continuous Spoof detected!")
                play_alarm("spoof")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                spoof_path = spoof_folder / f"spoof_{timestamp}.jpg"
                cv2.imwrite(str(spoof_path), frame)
                print(f"Spoof photo saved: {spoof_path.name}")
                if (current_time - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                    send_spoof_alert()
                    last_alert_time = current_time
                failed_attempts += 1
                print(f"Failed attempt: {failed_attempts}/{MAX_FAILED_ATTEMPTS}")
                if failed_attempts >= MAX_FAILED_ATTEMPTS:
                    lock_until = current_time + LOCK_DURATION_SECONDS
                    failed_attempts = 0
                    print(f"SYSTEM LOCKED for {LOCK_DURATION_SECONDS // 60} minutes!")
                    send_alert(
                        "SECURITY ALERT: System Locked",
                        f"System locked for 5 minutes due to multiple spoof attacks.\nTime of Attack: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                spoof_start_time = None
        else:
            if spoof_start_time and (current_time - last_spoof_seen) > RESET_TOLERANCE:
                spoof_start_time = None

        # Unknown logic
        if is_unknown and not is_locked:
            last_unknown_seen = current_time
            if unknown_start_time is None:
                unknown_start_time = current_time
                print(">>> Unknown person timer started...")
            elif (current_time - unknown_start_time) >= UNKNOWN_DURATION_SECONDS:
                print(">>> 5s continuous Unknown person detected!")
                play_alarm("generic")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unknown_path = unknown_folder / f"unknown_{timestamp}.jpg"
                cv2.imwrite(str(unknown_path), frame)
                print(f"Unknown photo saved: {unknown_path.name}")
                if (current_time - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                    send_unknown_alert()
                    last_alert_time = current_time
                failed_attempts += 1
                print(f"Failed attempt: {failed_attempts}/{MAX_FAILED_ATTEMPTS}")
                if failed_attempts >= MAX_FAILED_ATTEMPTS:
                    lock_until = current_time + LOCK_DURATION_SECONDS
                    failed_attempts = 0
                    print(f"SYSTEM LOCKED for {LOCK_DURATION_SECONDS // 60} minutes!")
                    send_alert(
                        "SECURITY ALERT: System Locked",
                        f"System locked for 5 minutes due to multiple unauthorized attempts.\nTime of Attack: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                unknown_start_time = None
        else:
            if unknown_start_time and (current_time - last_unknown_seen) > RESET_TOLERANCE:
                unknown_start_time = None

        # Extra text
        if is_locked:
            extra = f"LOCKED ({remaining_lock}s left)"
        elif is_fear and fear_start_time:
            extra = f"FEAR {int(current_time - fear_start_time)}s"
        elif is_spoof and spoof_start_time:
            extra = f"SPOOF {int(current_time - spoof_start_time)}s"
        elif is_unknown and unknown_start_time:
            extra = f"UNKNOWN {int(current_time - unknown_start_time)}s"
        else:
            extra = dominant

        # Write status for Flutter
        if is_locked:
            st = "locked"
            auth = False
            usr = None
            msg = f"SYSTEM LOCKED - Try after {remaining_lock}s"
            id_conf = 0.0
            live_conf = 0.0
        elif is_live and is_authentic:
            st = "unlocked"
            auth = True
            usr = person_name
            msg = f"ACCESS GRANTED - {person_name}"
            id_conf = float(np.clip(80.0 + (1.0 - (min_sim_score / IDENTITY_THRESHOLD)) * 15.0, 80.0, 95.0))
            live_conf = float(np.clip(liveness_score * 100, 0, 100))
        elif is_spoof:
            st = "locked"
            auth = False
            usr = None
            msg = "ACCESS DENIED - Spoof"
            id_conf = 0.0
            live_conf = float(np.clip(liveness_score * 100, 0, 100))
        else:
            st = "locked"
            auth = False
            usr = None
            msg = "ACCESS DENIED - Unknown"
            id_conf = float(np.clip(20.0 + (1.0 - min_sim_score) * 30.0, 15.0, 45.0))
            live_conf = float(np.clip(liveness_score * 100, 0, 100))

        write_device_status({
            "status": st,
            "authentication": auth,
            "user": usr,
            "identity_confidence": id_conf,
            "liveness_confidence": live_conf,
            "liveness_score": float(liveness_score),
            "similarity_score": float(min_sim_score),
            "message": msg,
            "failed_attempts": failed_attempts,
            "max_failed_attempts": MAX_FAILED_ATTEMPTS,
        })

        # Drawing
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 3)
        panel_y1 = max(0, y1 - 150)
        cv2.rectangle(canvas, (x1, panel_y1), (x1 + 380, panel_y1 + 145), (25, 25, 25), -1)

        cv2.putText(canvas, status_text, (x1+10, panel_y1+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(canvas, result_text, (x1+10, panel_y1+58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, result_color, 2)
        cv2.putText(canvas, f"Confidence: {confidence:.1f}%", (x1+10, panel_y1+88), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)
        cv2.putText(canvas, extra, (x1+10, panel_y1+118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,165,255), 2)

    cv2.imshow("AI Secure Face Authentication", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()