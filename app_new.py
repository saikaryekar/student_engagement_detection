import cv2
import torch
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request
from tensorflow import keras
import collections
import warnings
import os
import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Flask App
app = Flask(__name__)

# Load MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load TensorFlow Model
model_tf = keras.models.load_model("student_engagement_model.h5")

# Engagement Levels
engagement_levels = ["Low", "High"]

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("1", cv2.CAP_DSHOW)

ema_alpha = 0.2
face_smoothing = {}
prediction_queues = {}
frame_count = 0

# Engagement Data Storage
engagement_data = []
is_tracking = False  # Control variable for tracking

def smooth_bbox(new_bbox, face_index):
    if face_index not in face_smoothing:
        face_smoothing[face_index] = new_bbox
    else:
        for i in range(4):
            face_smoothing[face_index][i] = int(
                ema_alpha * new_bbox[i] + (1 - ema_alpha) * face_smoothing[face_index][i]
            )
    return face_smoothing[face_index]

def detect_and_classify():
    global frame_count, is_tracking

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_bbox, h_bbox = (
                    int(bboxC.xmin * w), int(bboxC.ymin * h),
                    int(bboxC.width * w), int(bboxC.height * h)
                )

                x, y = max(0, x), max(0, y)
                w_bbox, h_bbox = min(w - x, w_bbox), min(h - y, h_bbox)
                x, y, w_bbox, h_bbox = smooth_bbox([x, y, w_bbox, h_bbox], i)

                face_roi = frame[y:y + h_bbox, x:x + w_bbox]
                if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                    continue

                if i not in prediction_queues:
                    prediction_queues[i] = collections.deque(maxlen=5)

                if frame_count % 8 == 0:
                    # TensorFlow model prediction
                    face_resized = cv2.resize(face_roi, (256, 256))
                    face_resized = (face_resized - 0.5) * 2
                    face_resized = np.expand_dims(face_resized, axis=0)
                    prediction_tf = model_tf.predict(face_resized)
                    label_tf = 1 if prediction_tf[0][0] >= 0.5 else 0
                    prediction_queues[i].append(label_tf)
                    

                if len(prediction_queues[i]) > 0:
                    most_common_class = max(set(prediction_queues[i]), key=prediction_queues[i].count)
                    engagement_label = engagement_levels[most_common_class]
                else:
                    engagement_label = "Unknown"

                # Set bounding box color based on engagement level
                color = (0, 255, 0) if engagement_label == "High" else (0, 0, 255)  # Green for High, Red for Low
                cv2.rectangle(frame, (x, y), (x + w_bbox, y + h_bbox), color, 2)
                cv2.putText(frame, engagement_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)

                # Log data if tracking is enabled
                if is_tracking:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    engagement_data.append({"timestamp": timestamp, "engagement": engagement_label})

        frame_count += 1

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(detect_and_classify(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_engagement_data")
def get_engagement_data():
    return jsonify(engagement_data)

@app.route("/toggle_tracking", methods=["POST"])
def toggle_tracking():
    global is_tracking
    is_tracking = request.json.get("track", False)
    return jsonify({"tracking": is_tracking})

@app.route("/engagement_summary")
def engagement_summary():
    return render_template("engagement_summary.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
