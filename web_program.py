from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import base64
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, 
                             QLineEdit, QComboBox, QStackedWidget, QHBoxLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

app = Flask(__name__)

# Load the pose detection model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

def update_frame(self):
    ret, frame = self.cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))
    else:
        self.timer.stop()
        self.cap.release()

def detect_pose(frame):
    input_image = tf.expand_dims(frame, axis=0)
    input_image = tf.image.resize_with_pad(input_image, 192, 192)
    input_image = tf.cast(input_image, dtype=tf.int32)
    
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy().squeeze()
    
    return keypoints

def analyze_pose(keypoints):
    # Simplified analysis logic
    feedback = []
    
    # Example: Check elbow angle
    shoulder = keypoints[6][:2]
    elbow = keypoints[8][:2]
    wrist = keypoints[10][:2]
    
    if all(k[2] > 0.3 for k in [keypoints[6], keypoints[8], keypoints[10]]):
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        if elbow_angle < 90:
            feedback.append("Keep your elbow higher during the swing.")
    
    # Add more analysis here based on original logic
    
    if not feedback:
        feedback.append("Your form looks good! Keep practicing.")
    
    return " ".join(feedback)

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video = request.files['video']
    height = request.form['height']
    stroke_type = request.form['stroke_type']
    
    # Save the uploaded video
    video_path = 'temp_video.mp4'
    video.save(video_path)
    
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = detect_pose(frame)
        frames.append(keypoints)
    
    cap.release()
    
    feedback = analyze_pose(np.mean(frames, axis=0))
    
    return jsonify({'feedback': feedback})

if __name__ == '__main__':
    app.run(debug=True)