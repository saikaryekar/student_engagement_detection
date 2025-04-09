import os
import cv2
import numpy as np
import datetime
import warnings
import collections
import base64
from io import BytesIO
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import mediapipe as mp

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="student_engagement_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Engagement levels
engagement_levels = ["Low", "High"]

# Face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Data tracking
engagement_data = []
is_tracking = False
prediction_queues = {}

@app.route("/")
def index():
    return render_template("index_new.html")

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

@app.route("/process_frame_chunk", methods=["POST"])
def process_frame_chunk():
    annotated_frames = []
    global is_tracking

    for key in sorted(request.files.keys()):
        file = request.files[key]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for i, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)

                x, y = max(0, x), max(0, y)
                w_box, h_box = min(w - x, w_box), min(h - y, h_box)

                face_roi = frame[y:y + h_box, x:x + w_box]
                if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                    continue

                face_resized = cv2.resize(face_roi, (256, 256))
                face_resized = (face_resized - 0.5) * 2
                face_resized = np.expand_dims(face_resized, axis=0).astype(np.float32)

                interpreter.set_tensor(input_details[0]['index'], face_resized)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                label_idx = 1 if output[0][0] >= 0.5 else 0
                label = engagement_levels[label_idx]
                color = (0, 255, 0) if label == "High" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if is_tracking:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    engagement_data.append({"timestamp": timestamp, "engagement": label})

        _, buffer = cv2.imencode('.jpg', frame)
        base64_frame = base64.b64encode(buffer).decode('utf-8')
        annotated_frames.append(base64_frame)

    return jsonify({"annotated_frames": annotated_frames})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


# import cv2
# import torch
# import numpy as np
# import mediapipe as mp
# from flask import Flask, render_template, Response, jsonify, request, send_file
# from tensorflow import keras
# import collections
# import warnings
# import os
# import datetime
# from io import BytesIO
# import tensorflow as tf

# # Suppress warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# app = Flask(__name__)

# # Load models
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
# # model_tf = keras.models.load_model("student_engagement_model.h5")
# interpreter = tf.lite.Interpreter(model_path="student_engagement_model.tflite")
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Config
# engagement_levels = ["Low", "High"]
# face_smoothing = {}
# prediction_queues = {}
# ema_alpha = 0.2
# engagement_data = []
# is_tracking = False

# def smooth_bbox(new_bbox, face_index):
#     if face_index not in face_smoothing:
#         face_smoothing[face_index] = new_bbox
#     else:
#         for i in range(4):
#             face_smoothing[face_index][i] = int(
#                 ema_alpha * new_bbox[i] + (1 - ema_alpha) * face_smoothing[face_index][i]
#             )
#     return face_smoothing[face_index]

# @app.route("/")
# def index():
#     return render_template("index_new.html")

# @app.route("/process_frame", methods=["POST"])
# def process_frame():
#     global is_tracking

#     file = request.files.get("frame")
#     if not file:
#         return "No frame uploaded", 400

#     file_bytes = np.frombuffer(file.read(), np.uint8)
#     frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_detection.process(rgb_frame)

#     if results.detections:
#         for i, detection in enumerate(results.detections):
#             bboxC = detection.location_data.relative_bounding_box
#             h, w, _ = frame.shape
#             x, y, w_bbox, h_bbox = (
#                 int(bboxC.xmin * w), int(bboxC.ymin * h),
#                 int(bboxC.width * w), int(bboxC.height * h)
#             )
#             x, y = max(0, x), max(0, y)
#             w_bbox, h_bbox = min(w - x, w_bbox), min(h - y, h_bbox)
#             x, y, w_bbox, h_bbox = smooth_bbox([x, y, w_bbox, h_bbox], i)

#             face_roi = frame[y:y + h_bbox, x:x + w_bbox]
#             if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
#                 continue

#             if i not in prediction_queues:
#                 prediction_queues[i] = collections.deque(maxlen=5)

#             face_resized = cv2.resize(face_roi, (256, 256))
#             face_resized = (face_resized - 0.5) * 2
#             face_resized = np.expand_dims(face_resized, axis=0)
#             # prediction_tf = model_tf.predict(face_resized, verbose=0)
#             # label_tf = 1 if prediction_tf[0][0] >= 0.5 else 0
#             input_data = face_resized.astype(np.float32)  # or .astype(np.uint8) if fully quantized
#             interpreter.set_tensor(input_details[0]['index'], input_data)
#             interpreter.invoke()
#             output_data = interpreter.get_tensor(output_details[0]['index'])
#             label_tf = 1 if output_data[0][0] >= 0.5 else 0
#             prediction_queues[i].append(label_tf)
#             prediction_queues[i].append(label_tf)

#             most_common_class = max(set(prediction_queues[i]), key=prediction_queues[i].count)
#             engagement_label = engagement_levels[most_common_class]
#             color = (0, 255, 0) if engagement_label == "High" else (0, 0, 255)

#             cv2.rectangle(frame, (x, y), (x + w_bbox, y + h_bbox), color, 2)
#             cv2.putText(frame, engagement_label, (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#             if is_tracking:
#                 timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 engagement_data.append({"timestamp": timestamp, "engagement": engagement_label})

#     _, buffer = cv2.imencode('.jpg', frame)
#     return send_file(BytesIO(buffer), mimetype='image/jpeg')

# @app.route("/toggle_tracking", methods=["POST"])
# def toggle_tracking():
#     global is_tracking
#     is_tracking = request.json.get("track", False)
#     return jsonify({"tracking": is_tracking})

# @app.route("/get_engagement_data")
# def get_engagement_data():
#     return jsonify(engagement_data)

# @app.route("/engagement_summary")
# def engagement_summary():
#     return render_template("engagement_summary.html")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
