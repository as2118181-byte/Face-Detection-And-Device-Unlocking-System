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
from queue import Queue, Empty

import cv2
import numpy as np
from dotenv import load_dotenv
from fer.fer import FER
from flask import Flask, Response, request, send_file
import jsonpickle

from facetools import FaceDetection, IdentityVerification, LivenessDetection
from security_events import (
    create_event, list_events, get_event, safe_image_path,
    delete_event, delete_all_events
)


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
AUTH_HISTORY_FILE = data_folder / "auth_history.json"


def write_device_status(status_dict):
    status_dict["pc_online"] = True
    status_dict["timestamp"] = datetime.now().isoformat()
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_dict, f, indent=2)
    except Exception as e:
        print(f"[STATUS] write failed: {e}")


def _ensure_auth_history():
    AUTH_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not AUTH_HISTORY_FILE.exists():
        with open(AUTH_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)


def append_auth_history(entry: dict):
    try:
        _ensure_auth_history()
        with open(AUTH_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
            if not isinstance(history, list):
                history = []
        history.append(entry)
        if len(history) > 200:
            history = history[-200:]
        with open(AUTH_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"[HISTORY] Saved authentication: {entry.get('name')} at {entry.get('timestamp')}")
    except Exception as e:
        print(f"[HISTORY] Failed to save: {e}")


# -------------------------------------------------
# Embedded Flask server
# -------------------------------------------------
flask_app = Flask(__name__)

_sse_clients = []
_sse_lock = threading.Lock()

# Live camera stream support
_latest_jpeg = None
_jpeg_lock = threading.Lock()
_live_camera_activated = False   # becomes True after first failed attempt and stays True


def _broadcast_event(event: dict):
    payload = json.dumps({
        "event_id": event["event_id"],
        "type": event["type"],
        "filename": event["filename"],
        "timestamp": event["timestamp"],
        "score": event.get("score"),
    })
    dead = []
    with _sse_lock:
        for q in _sse_clients:
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            if q in _sse_clients:
                _sse_clients.remove(q)


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
                "live_camera_active": False,
            }),
            status=200,
            mimetype="application/json",
        )
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["pc_online"] = True
        data["live_camera_active"] = _live_camera_activated
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


@flask_app.route("/api/live-frame")
def live_frame():
    """Returns the latest camera frame as JPEG. Used by Flutter for smooth live view."""
    with _jpeg_lock:
        data = _latest_jpeg
    if data is None:
        return Response(status=204)
    return Response(data, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                             "Pragma": "no-cache",
                             "Expires": "0"})


# ---------- Security Events API ----------
@flask_app.route("/api/security-events", methods=["GET"])
def api_list_events():
    limit = request.args.get("limit", 30, type=int)
    limit = max(1, min(limit, 100))
    events = list_events(limit=limit)
    safe = [{
        "event_id": e["event_id"],
        "type": e["type"],
        "filename": e["filename"],
        "timestamp": e["timestamp"],
        "score": e.get("score"),
    } for e in events]
    return Response(jsonpickle.encode(safe), status=200, mimetype="application/json")


@flask_app.route("/api/security-events/<event_id>", methods=["GET"])
def api_get_event(event_id):
    e = get_event(event_id)
    if not e:
        return Response(jsonpickle.encode({"error": "not found"}), status=404, mimetype="application/json")
    return Response(jsonpickle.encode({
        "event_id": e["event_id"],
        "type": e["type"],
        "filename": e["filename"],
        "timestamp": e["timestamp"],
        "score": e.get("score"),
    }), status=200, mimetype="application/json")


@flask_app.route("/api/security-events/<event_id>/image", methods=["GET"])
def api_event_image(event_id):
    path = safe_image_path(event_id)
    if path is None:
        print(f"[EVENT] Image request rejected or missing: {event_id}")
        return Response("Not found", status=404)
    print(f"[EVENT] Image requested: {event_id} → {path.name}")
    return send_file(path, mimetype="image/jpeg")


@flask_app.route("/api/security-events/<event_id>", methods=["DELETE"])
def api_delete_event(event_id):
    success = delete_event(event_id)
    if success:
        return Response(jsonpickle.encode({"status": "deleted", "event_id": event_id}),
                        status=200, mimetype="application/json")
    return Response(jsonpickle.encode({"error": "not found"}), status=404, mimetype="application/json")


@flask_app.route("/api/security-events", methods=["DELETE"])
def api_delete_all_events():
    count = delete_all_events()
    return Response(jsonpickle.encode({"status": "deleted_all", "count": count}),
                    status=200, mimetype="application/json")


@flask_app.route("/api/security-events/stream")
def api_event_stream():
    def generate():
        q = Queue(maxsize=32)
        with _sse_lock:
            _sse_clients.append(q)
        print("[EVENT] APK connected to event stream")
        try:
            yield "data: {\"type\":\"connected\"}\n\n"
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield f"data: {msg}\n\n"
                except Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)
            print("[EVENT] APK disconnected from event stream")

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def start_flask_server():
    flask_app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


# Decision Thresholds
LIVENESS_THRESHOLD = 0.12
IDENTITY_THRESHOLD = 0.85
LIVENESS_UPPER_BOUND = 0.90
SCORE_EMA_ALPHA = 0.25

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

sim_score_ema = None
liveness_score_ema = None


def update_ema(previous, new_value, alpha=SCORE_EMA_ALPHA):
    if previous is None:
        return new_value
    return alpha * new_value + (1.0 - alpha) * previous


def compute_identity_confidence(sim_score: float, authentic: bool) -> float:
    if authentic:
        frac = float(np.clip(1.0 - (sim_score / IDENTITY_THRESHOLD), 0.0, 1.0))
        return float(np.clip(80.0 + frac * 18.0, 80.0, 98.0))
    frac = float(np.clip(1.0 - sim_score, 0.0, 1.0))
    return float(np.clip(20.0 + frac * 30.0, 15.0, 45.0))


def compute_liveness_confidence(live_score: float, live_flag: bool) -> float:
    if live_flag:
        frac = float(np.clip(
            (live_score - LIVENESS_THRESHOLD) / (LIVENESS_UPPER_BOUND - LIVENESS_THRESHOLD),
            0.0, 1.0,
        ))
        return float(np.clip(80.0 + frac * 18.0, 80.0, 98.0))
    return float(np.clip(5.0 + random.uniform(0, 5), 5.0, 10.0))

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

last_auth_user = None
last_auth_log_time = 0.0
AUTH_HISTORY_COOLDOWN = 30.0

# -------------------------------------------------
# Start Flask server in background
# -------------------------------------------------
print("=" * 55)
print("AI Secure Face Authentication + Dashboard Server")
print("Starting embedded Flask server on port 5000...")
print("=" * 55)

flask_thread = threading.Thread(target=start_flask_server, daemon=True)
flask_thread.start()

time.sleep(1.5)

# -------------------------------------------------
# Webcam
# -------------------------------------------------
cap = cv2.VideoCapture(0)

print("System started... Press 'q' to quit")
print(f"Spoof photos  → {spoof_folder}")
print(f"Unknown photos → {unknown_folder}")
print("Flutter can now connect to http://127.0.0.1:5000/api/device-status")
print("Live camera stream available at http://127.0.0.1:5000/api/live-frame")

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
            "live_camera_active": _live_camera_activated,
        })

    for face_arr, box in zip(faces, boxes):
        min_sim_score, mean_sim_score, person_name = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)

        sim_score_ema = update_ema(sim_score_ema, min_sim_score)
        liveness_score_ema = update_ema(liveness_score_ema, liveness_score)

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
            confidence = compute_identity_confidence(sim_score_ema, authentic=True)
            status_text = "Liveliness Detected"
            status_color = (0, 255, 0)
            box_color = (0, 255, 0)
            result_text = f"Device Unlocked - {person_name}"
            result_color = (0, 255, 0)
            failed_attempts = 0
        elif is_spoof:
            confidence = compute_liveness_confidence(liveness_score_ema, live_flag=False)
            status_text = "Spoof Detected"
            status_color = (0, 0, 255)
            box_color = (0, 0, 255)
            result_text = "Authentication Failed"
            result_color = (0, 0, 255)
        else:
            confidence = compute_identity_confidence(sim_score_ema, authentic=False)
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

                try:
                    evt = create_event(
                        event_type="spoof",
                        filename=spoof_path.name,
                        image_path=spoof_path,
                        score=float(liveness_score_ema) if liveness_score_ema is not None else float(liveness_score),
                    )
                    if evt:
                        _broadcast_event(evt)
                except Exception as ex:
                    print(f"[EVENT] Failed to create spoof event: {ex}")

                if (current_time - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                    send_spoof_alert()
                    last_alert_time = current_time
                failed_attempts += 1
                print(f"Failed attempt: {failed_attempts}/{MAX_FAILED_ATTEMPTS}")
                if failed_attempts >= 1:
                    _live_camera_activated = True
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

                try:
                    evt = create_event(
                        event_type="unknown",
                        filename=unknown_path.name,
                        image_path=unknown_path,
                        score=float(sim_score_ema) if sim_score_ema is not None else float(min_sim_score),
                    )
                    if evt:
                        _broadcast_event(evt)
                except Exception as ex:
                    print(f"[EVENT] Failed to create unknown event: {ex}")

                if (current_time - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                    send_unknown_alert()
                    last_alert_time = current_time
                failed_attempts += 1
                print(f"Failed attempt: {failed_attempts}/{MAX_FAILED_ATTEMPTS}")
                if failed_attempts >= 1:
                    _live_camera_activated = True
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
            id_conf = compute_identity_confidence(sim_score_ema, authentic=True)
            live_conf = compute_liveness_confidence(liveness_score_ema, live_flag=True)

            should_log = (
                person_name != last_auth_user or
                (current_time - last_auth_log_time) >= AUTH_HISTORY_COOLDOWN
            )
            if should_log:
                append_auth_history({
                    "name": person_name,
                    "timestamp": datetime.now().isoformat(),
                    "identity_confidence": round(id_conf, 2),
                    "liveness_confidence": round(live_conf, 2),
                    "liveness_score": round(float(liveness_score_ema), 4),
                    "similarity_score": round(float(sim_score_ema), 4),
                    "message": msg,
                })
                last_auth_user = person_name
                last_auth_log_time = current_time

        elif is_spoof:
            st = "locked"
            auth = False
            usr = None
            msg = "ACCESS DENIED - Spoof"
            id_conf = 0.0
            live_conf = compute_liveness_confidence(liveness_score_ema, live_flag=False)
        else:
            st = "locked"
            auth = False
            usr = None
            msg = "ACCESS DENIED - Unknown"
            id_conf = compute_identity_confidence(sim_score_ema, authentic=False)
            live_conf = compute_liveness_confidence(liveness_score_ema, live_flag=True)

        write_device_status({
            "status": st,
            "authentication": auth,
            "user": usr,
            "identity_confidence": id_conf,
            "liveness_confidence": live_conf,
            "liveness_score": float(liveness_score_ema) if liveness_score_ema is not None else 0.0,
            "similarity_score": float(sim_score_ema) if sim_score_ema is not None else 0.0,
            "message": msg,
            "failed_attempts": failed_attempts,
            "max_failed_attempts": MAX_FAILED_ATTEMPTS,
            "live_camera_active": _live_camera_activated,
        })

        # Drawing
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 3)
        panel_y1 = max(0, y1 - 150)
        cv2.rectangle(canvas, (x1, panel_y1), (x1 + 380, panel_y1 + 145), (25, 25, 25), -1)

        cv2.putText(canvas, status_text, (x1+10, panel_y1+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(canvas, result_text, (x1+10, panel_y1+58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, result_color, 2)
        cv2.putText(canvas, f"Confidence: {confidence:.1f}%", (x1+10, panel_y1+88), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)
        cv2.putText(canvas, extra, (x1+10, panel_y1+118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,165,255), 2)

    # ---- Update live frame for Flutter (always, so stream is ready) ----
    try:
        # Resize a bit for faster transfer over ADB (still clear)
        stream_frame = cv2.resize(canvas, (640, 480))
        ret_enc, buf = cv2.imencode(".jpg", stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ret_enc:
            with _jpeg_lock:
                _latest_jpeg = buf.tobytes()
    except Exception:
        pass

    cv2.imshow("AI Secure Face Authentication", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()