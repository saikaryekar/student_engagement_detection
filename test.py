import cv2
import numpy as np
from tensorflow import keras
import mediapipe as mp

# Load the trained model
model = keras.models.load_model("student_engagement_model.h5")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Start Video Capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape

                # Convert bbox coordinates
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                                     int(bboxC.width * w), int(bboxC.height * h)

                # Crop face region
                face_roi = frame[y:y + h_box, x:x + w_box]
                if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                    face_resized = cv2.resize(face_roi, (256, 256))  # Resize to match model input
                    # face_resized = face_resized / 255.0  # Normalize
                    face_resized = (face_resized - 0.5) * 2
                    face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension

                    # Predict engagement level
                    prediction = model.predict(face_resized)
                    label = "Engaged" if prediction[0][0] <= 0.5 else "Not Engaged"

                    # Draw bounding box & label
                    color = (0, 255, 0) if label == "Engaged" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show the frame
        cv2.imshow("Engagement Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
