# MatchPoint.ai - Balanced Performance Version
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import tempfile
import shutil
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration - Balanced for quality and performance
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-jwt-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///matchpoint.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB for 3-minute videos

# Create directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

# TensorFlow configuration
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

# Global variables
movenet = None
model_loaded = False

def load_movenet_model():
    """Load MoveNet model."""
    global movenet, model_loaded
    try:
        if movenet is None:
            logging.info("Loading MoveNet model...")
            movenet = hub.KerasLayer("https://tfhub.dev/google/movenet/singlepose/lightning/4",
                                     signature="serving_default", signature_outputs_as_dict=True)
            model_loaded = True
            logging.info("MoveNet model loaded successfully")
        return movenet
    except Exception as e:
        logging.error(f"Error loading MoveNet model: {e}")
        model_loaded = False
        return None

# Initialize JWT and SQLAlchemy
jwt = JWTManager(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    elbow_angles = db.Column(db.JSON, nullable=False)
    knee_angles = db.Column(db.JSON, nullable=False)
    racket_velocity = db.Column(db.Float, nullable=False)
    feedback_text = db.Column(db.Text, nullable=False)

def analyze_swing_balanced(video_path, height, stroke_type):
    """Balanced swing analysis - quality focused with moderate optimizations."""
    model = load_movenet_model()
    if model is None:
        return "Error: MoveNet model not available. Please try again later.", None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file at {video_path}")
        return "Error: Could not open video file.", None
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Balanced optimization settings
    frame_skip = 3  # Process every 3rd frame for good quality
    max_frames = min(75, total_frames)  # Process up to 75 frames or all frames if less
    
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
    try:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), fps, (frame_width, frame_height))
    except Exception as e:
        logging.error(f"Error creating VideoWriter: {e}")
        cap.release()
        return "Error creating video. Check codec and file permissions.", None
    
    elbow_angles = []
    knee_angles = []
    racket_positions = []
    all_frames = []

    frame_count = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret or processed_frames >= max_frames:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        processed_frames += 1
        
        # Resize frame for faster processing (maintain aspect ratio)
        scale_factor = 0.6  # Slightly smaller for better performance
        new_width = int(frame_width * scale_factor)
        new_height = int(frame_height * scale_factor)
        frame_resized = cv2.resize(frame, (new_width, new_height))
        
        annotated_frame, keypoints = detect_pose_balanced(frame_resized, model)
        
        # Resize back to original size for output
        annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))
        all_frames.append(annotated_frame)

        # Calculate elbow angle (using right arm)
        if all(k[2] > 0.3 for k in [keypoints[6], keypoints[8], keypoints[10]]):
            shoulder = keypoints[6][:2]
            elbow = keypoints[8][:2]
            wrist = keypoints[10][:2]
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            elbow_angles.append(elbow_angle)

        # Calculate knee angle (using right leg)
        if all(k[2] > 0.3 for k in [keypoints[12], keypoints[14], keypoints[16]]):
            hip = keypoints[12][:2]
            knee = keypoints[14][:2]
            ankle = keypoints[16][:2]
            knee_angle = calculate_angle(hip, knee, ankle)
            knee_angles.append(knee_angle)

        # Track racket position
        if keypoints[10][2] > 0.3:
            racket_position = keypoints[10][:2]
            racket_positions.append(racket_position)

    # Write all frames
    for frame in all_frames:
        out.write(frame)
    cap.release()
    out.release()

    # Analyze the angles
    avg_elbow_angle = np.mean(elbow_angles) if elbow_angles else 0
    avg_knee_angle = np.mean(knee_angles) if knee_angles else 0

    feedback = []
    if avg_elbow_angle < 50:
        feedback.append(
            "🟥 Your elbow is extremely low, severely limiting power and increasing injury risk. "
            "Focus on drills to keep your elbow raised at shoulder height during the swing."
        )
    elif avg_elbow_angle < 70:
        feedback.append(
            "🟧 Elbow is low, restricting swing speed and control. Keep practicing raising your elbow during your swings."
        )
    elif avg_elbow_angle < 90:
        feedback.append(
            "🟨 Elbow position is slightly low. Raising it will help increase power and consistency."
        )
    elif avg_elbow_angle <= 110:
        feedback.append(
            "🟩 Great elbow position! This helps generate power and consistent contact."
        )
    else:
        feedback.append(
            "🟨 Your elbow is a bit too high, which can cause late contact or overextension. "
            "Work on smooth, relaxed swings with good follow-through."
        )

    if avg_knee_angle > 180:
        feedback.append(
            "🟥 Your knees are fully locked, which kills balance and power. Bend more before swinging."
        )
    elif avg_knee_angle > 170:
        feedback.append(
            "🟧 Knees too straight, limiting balance and power generation. Try to lower your center of gravity."
        )
    elif avg_knee_angle > 150:
        feedback.append(
            "🟨 Knee bend is okay but could be improved. Lower your stance slightly for better stability."
        )
    elif avg_knee_angle > 120:
        feedback.append(
            "🟩 Good knee bend, supporting power and balance."
        )
    else:
        feedback.append(
            "🟦 Excellent knee bend! This maximizes power and stability during your swings."
        )

    if racket_positions and len(racket_positions) > 1:
        velocities = [np.linalg.norm(np.array(racket_positions[i]) - np.array(racket_positions[i - 1]))
                      for i in range(1, len(racket_positions))]
        avg_velocity = np.mean(velocities)

        if avg_velocity < 2:
            feedback.append(
                "🟥 Your swing speed is very slow. Focus on accelerating through contact using your hips and shoulders."
            )
        elif avg_velocity < 4:
            feedback.append(
                "🟧 Swing speed is below average. Work on generating more racquet head speed through proper technique."
            )
        elif avg_velocity < 6:
            feedback.append(
                "🟨 Swing speed is decent. Focus on timing and contact point for better results."
            )
        else:
            feedback.append(
                "🟩 Good swing speed! Keep working on consistency and accuracy."
            )

    # Add stroke-specific feedback
    if stroke_type.lower() == 'forehand':
        feedback.append("🎾 Forehand Analysis: Focus on keeping your elbow at shoulder height and following through.")
    elif stroke_type.lower() == 'backhand':
        feedback.append("🎾 Backhand Analysis: Maintain good form with proper grip and follow-through.")
    elif stroke_type.lower() == 'serve':
        feedback.append("🎾 Serve Analysis: Work on your toss consistency and racquet head speed.")

    # Add general tips
    feedback.append("✅ Your form looks good! Keep practicing.")
    feedback.append(f"📊 Processed {processed_frames} frames for analysis.")

    return " ".join(feedback), '/static/output.mp4'

def detect_pose_balanced(frame, movenet):
    """Balanced pose detection with good quality."""
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    img = tf.cast(img, dtype=tf.int32)
    try:
        outputs = movenet(img)
        keypoints = outputs['output_0'].numpy().squeeze()
    except Exception as e:
        logging.error(f"Error during MoveNet inference: {e}")
        return frame, np.zeros((17,3))

    y, x, _ = frame.shape
    for keypoint in keypoints:
        ky, kx, kp_conf = keypoint
        if kp_conf > 0.3:
            cv2.circle(frame, (int(kx * x), int(ky * y)), 5, (0, 0, 255), -1)

    # Draw lines between keypoints
    connections = [(0, 1), (0, 2), (1, 3), (2, 4),  # Head to shoulders and arms
                   (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Body and legs
                   (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]  # Feet
    for connection in connections:
        start_point = keypoints[connection[0]]
        end_point = keypoints[connection[1]]
        if start_point[2] > 0.3 and end_point[2] > 0.3:
            cv2.line(frame,
                     (int(start_point[1] * x), int(start_point[0] * y)),
                     (int(end_point[1] * x), int(end_point[0] * y)),
                     (0, 0, 255), 2)

    # Track the racket (assuming it's the wrist point)
    racket_point = keypoints[10]  # Assuming wrist is the racket point
    if racket_point[2] > 0.3:
        cv2.circle(frame, (int(racket_point[1] * x), int(racket_point[0] * y)), 5, (255, 0, 0), -1)

    return frame, keypoints

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/ai-platform')
def ai_platform():
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    height = request.form.get('height', '5.8')
    stroke_type = request.form.get('stroke_type', 'forehand')
    
    # Validate file size
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB.'}), 400
    
    # Save file
    filename = f"upload_{int(time.time())}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        feedback, video_path = analyze_swing_balanced(file_path, height, stroke_type)
        
        return jsonify({
            'feedback': feedback,
            'video_path': video_path,
            'analysis_type': 'balanced_analysis'
        })
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed. Please try again with a different video.'}), 500
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.error(f"Error cleaning up file: {e}")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

@app.route('/health')
def health_check():
    model_status = load_movenet_model() is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_status,
        'platform': 'DigitalOcean App Platform - Balanced Mode',
        'features': [
            'Balanced performance and quality',
            'Process every 3rd frame',
            'Up to 75 frames maximum',
            'Full skeleton overlay',
            'Comprehensive feedback',
            '3-minute video support'
        ],
        'performance': 'Balanced Mode'
    })

# Initialize database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
