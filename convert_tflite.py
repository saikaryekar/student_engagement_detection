import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model("student_engagement_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open("student_engagement_model.tflite", "wb") as f:
    f.write(tflite_model)
