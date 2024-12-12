# current version: v01.04

import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, 
                             QLineEdit, QComboBox, QStackedWidget, QHBoxLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

# cache clearing method

import shutil
import os

def clear_tfhub_cache():
    tfhub_cache = os.path.join(os.path.expanduser("~"), ".cache", "tensorflow_hub")
    if os.path.exists(tfhub_cache):
        shutil.rmtree(tfhub_cache)

clear_tfhub_cache()

# homepage
class HomePage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()

    def initUI(self):
        self.setWindowTitle('MatchPoint AI')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: white;")
        layout = QVBoxLayout()

        self.logo_label = QLabel(self)
        self.logo_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap('C:/Users/rupal/Downloads/matchpoint_logo_white.png')
        self.logo_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        layout.addWidget(self.logo_label)

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.go_to_input_page)
        layout.addWidget(self.start_button)
        self.setLayout(layout)

    def go_to_input_page(self):
        self.stacked_widget.setCurrentIndex(1)

# input information page
class InputPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()

    def initUI(self):
        self.setWindowTitle('MatchPoint AI')
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()

        # height input
        self.height_label = QLabel('Enter your height (cm):', self)
        layout.addWidget(self.height_label)
        self.height_input = QLineEdit(self)
        layout.addWidget(self.height_input)

        # stroke input
        self.stroke_label = QLabel('Select stroke type:', self)
        layout.addWidget(self.stroke_label)
        self.stroke_combo = QComboBox(self)
        self.stroke_combo.addItems(['Forehand', 'Backhand'])
        layout.addWidget(self.stroke_combo)

        self.next_button = QPushButton('Next', self)
        self.next_button.clicked.connect(self.go_to_analysis_page)
        layout.addWidget(self.next_button)
        self.setLayout(layout)

    def go_to_analysis_page(self):
        height = self.height_input.text()
        stroke_type = self.stroke_combo.currentText()
        self.stacked_widget.widget(2).set_user_input(height, stroke_type)
        self.stacked_widget.setCurrentIndex(2)

# MatchPoint.ai analysis page
class MatchPointAI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadPoseModel()
        self.height = None
        self.stroke_type = None

    def initUI(self):
        self.setWindowTitle('MatchPoint AI')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLabel {
                font-size: 20px;
            }
        """)

        layout = QVBoxLayout()

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        self.upload_button = QPushButton('Upload Video', self)
        self.upload_button.clicked.connect(self.upload_video)
        layout.addWidget(self.upload_button)

        self.analyze_button = QPushButton('Analyze Swing', self)
        self.analyze_button.clicked.connect(self.analyze_swing)
        self.analyze_button.setEnabled(False)
        layout.addWidget(self.analyze_button)

        self.pause_button = QPushButton('Pause', self)
        self.pause_button.clicked.connect(self.pause_video)
        self.pause_button.setEnabled(False)
        layout.addWidget(self.pause_button)

        self.resume_button = QPushButton('Resume', self)
        self.resume_button.clicked.connect(self.resume_video)
        self.resume_button.setEnabled(False)
        layout.addWidget(self.resume_button)

        self.feedback_label = QLabel('Upload a video and click "Analyze Swing" to get feedback.', self)
        self.feedback_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.feedback_label)

        self.setLayout(layout)

        self.video_path = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def loadPoseModel(self):
        self.model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self.movenet = self.model.signatures['serving_default']

    def set_user_input(self, height, stroke_type):
        self.height = height
        self.stroke_type = stroke_type

    def upload_video(self):
        file_dialog = QFileDialog()
        self.video_path, _ = file_dialog.getOpenFileName(self, 'Open Video File', '', 'Video Files (*.mp4 *.avi)')
        
        if self.video_path:
            self.analyze_button.setEnabled(True)
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.feedback_label.setText('Video uploaded. Click "Analyze Swing" to get feedback.')
            self.start_video()

    def start_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.timer.start(30)  # Update every 30 ms

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.detect_pose(frame)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))
        else:
            self.timer.stop()
            self.cap.release()

    def detect_pose(self, frame):
        # Resize and pad the image to keep the aspect ratio and fit the expected size
        input_image = tf.expand_dims(frame, axis=0)
        input_image = tf.image.resize_with_pad(input_image, 192, 192)
        input_image = tf.cast(input_image, dtype=tf.int32)

        # Run model inference
        outputs = self.movenet(input_image)
        keypoints = outputs['output_0'].numpy().squeeze()

        # Draw the skeleton
        y, x, _ = frame.shape
        for keypoint in keypoints:
            ky, kx, kp_conf = keypoint
            if kp_conf > 0.3:
                cv2.circle(frame, (int(kx * x), int(ky * y)), 5, (0, 0, 255), -1)

        # Draw lines between keypoints
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head to shoulders and arms
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Body and legs
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Feet
        ]
        for connection in connections:
            start_point = keypoints[connection[0]]
            end_point = keypoints[connection[1]]
            if start_point[2] > 0.3 and end_point[2] > 0.3:
                cv2.line(frame, 
                         (int(start_point[1] * x), int(start_point[0] * y)), 
                         (int(end_point[1] * x), int(end_point[0] * y)), 
                         (0, 0, 255), 2)

        # Track the racket (assuming it's the highest point in the frame)
        racket_point = keypoints[10]  # Assuming wrist is the racket point
        if racket_point[2] > 0.3:
            cv2.circle(frame, (int(racket_point[1] * x), int(racket_point[0] * y)), 5, (255, 0, 0), -1)

        return frame

    def analyze_swing(self):
        feedback = self.analyze_pose()
        self.feedback_label.setText(f"Coach's Feedback: {feedback}")

    def analyze_pose(self):
        # Reset video to start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        elbow_angles = []
        knee_angles = []
        racket_positions = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            input_image = tf.expand_dims(frame, axis=0)
            input_image = tf.image.resize_with_pad(input_image, 192, 192)
            input_image = tf.cast(input_image, dtype=tf.int32)

            # Run model inference
            outputs = self.movenet(input_image)
            keypoints = outputs['output_0'].numpy().squeeze()

            # Calculate elbow angle (using right arm)
            shoulder = keypoints[6][:2]
            elbow = keypoints[8][:2]
            wrist = keypoints[10][:2]
            if all(k[2] > 0.3 for k in [keypoints[6], keypoints[8], keypoints[10]]):
                elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                elbow_angles.append(elbow_angle)

            # Calculate knee angle (using right leg)
            hip = keypoints[12][:2]
            knee = keypoints[14][:2]
            ankle = keypoints[16][:2]
            if all(k[2] > 0.3 for k in [keypoints[12], keypoints[14], keypoints[16]]):
                knee_angle = self.calculate_angle(hip, knee, ankle)
                knee_angles.append(knee_angle)

            # Track racket position
            racket_position = keypoints[10][:2]
            if keypoints[10][2] > 0.3:
                racket_positions.append(racket_position)

        # Analyze the angles
        avg_elbow_angle = np.mean(elbow_angles) if elbow_angles else 0
        avg_knee_angle = np.mean(knee_angles) if knee_angles else 0

        feedback = []
        if avg_elbow_angle < 90:
            feedback.append("Keep your elbow higher during the swing.")
        if avg_knee_angle > 160:
            feedback.append("Bend your knees more for better balance and power.")
        
        # Analyze racket velocity and path
        if racket_positions:
            velocities = [np.linalg.norm(racket_positions[i] - racket_positions[i-1]) for i in range(1, len(racket_positions))]
            avg_velocity = np.mean(velocities)
            if avg_velocity < 5:
                feedback.append("Accelerate through the ball for more power.")
            else:
                feedback.append("Good acceleration through the ball.")
            
            # Check for stable/linear path
            path_variation = np.std([pos[1] for pos in racket_positions])
            if path_variation > 10:
                feedback.append("Try to keep the racket path more stable and linear.")
            else:
                feedback.append("Good stable racket path.")

        if not feedback:
            feedback.append("Your form looks good! Keep practicing.")

        return " ".join(feedback)

    def calculate_angle(self, a, b, c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def pause_video(self):
        self.timer.stop()
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(True)

    def resume_video(self):
        self.timer.start(30)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    stacked_widget = QStackedWidget()

    home_page = HomePage(stacked_widget)
    input_page = InputPage(stacked_widget)
    match_point_ai = MatchPointAI()

    stacked_widget.addWidget(home_page)
    stacked_widget.addWidget(input_page)
    stacked_widget.addWidget(match_point_ai)

    stacked_widget.setCurrentIndex(0)

    stacked_widget.show()
    sys.exit(app.exec_())