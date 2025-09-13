# MatchPoint.ai - Highly Optimized Version
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
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
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration for maximum performance
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-jwt-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///matchpoint.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Increased to 100MB for 3-minute videos

# Create directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
CACHE_FOLDER = 'cache'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['CACHE_FOLDER'] = CACHE_FOLDER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

# Optimized TensorFlow configuration
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.set_visible_devices([], 'GPU')  # Use CPU only for consistency

# Global variables with caching
movenet = None
model_loaded = False
processing_cache = {}
executor = ThreadPoolExecutor(max_workers=2)

def load_movenet_model():
    """Load MoveNet model with caching."""
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

# Database models (same as before)
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

class AnalysisCache(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_hash = db.Column(db.String(64), unique=True, nullable=False)
    height = db.Column(db.Float, nullable=False)
    stroke_type = db.Column(db.String(50), nullable=False)
    feedback = db.Column(db.Text, nullable=False)
    video_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

def get_video_hash(video_path, height, stroke_type):
    """Generate hash for video caching."""
    with open(video_path, 'rb') as f:
        video_content = f.read()
    content_hash = hashlib.sha256(video_content).hexdigest()
    return hashlib.sha256(f"{content_hash}_{height}_{stroke_type}".encode()).hexdigest()

def check_cache(video_hash):
    """Check if analysis is cached."""
    cached = AnalysisCache.query.filter_by(video_hash=video_hash).first()
    if cached:
        # Check if video file still exists
        if os.path.exists(cached.video_path):
            logging.info(f"Cache hit for video hash: {video_hash}")
            return cached.feedback, cached.video_path
    return None, None

def cache_result(video_hash, height, stroke_type, feedback, video_path):
    """Cache analysis result."""
    try:
        # Remove old cache entry if exists
        old_cache = AnalysisCache.query.filter_by(video_hash=video_hash).first()
        if old_cache:
            if os.path.exists(old_cache.video_path):
                os.remove(old_cache.video_path)
            db.session.delete(old_cache)
        
        # Create new cache entry
        cache_entry = AnalysisCache(
            video_hash=video_hash,
            height=height,
            stroke_type=stroke_type,
            feedback=feedback,
            video_path=video_path
        )
        db.session.add(cache_entry)
        db.session.commit()
        logging.info(f"Cached result for video hash: {video_hash}")
    except Exception as e:
        logging.error(f"Error caching result: {e}")

def analyze_swing_ultra_fast(video_path, height, stroke_type):
    """Ultra-fast swing analysis with aggressive optimizations."""
    
    # Check cache first
    video_hash = get_video_hash(video_path, height, stroke_type)
    cached_feedback, cached_video = check_cache(video_hash)
    if cached_feedback:
        return cached_feedback, cached_video
    
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
    
    # Balanced settings for longer videos (up to 3 minutes)
    frame_skip = 4  # Process every 4th frame for better pose detection
    max_frames = 50  # Increased to 50 frames for comprehensive analysis
    processed_frames = 0
    
    # Process frames from the beginning for better pose detection
    
    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_output_path = temp_output.name
    temp_output.close()
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps//frame_skip, (640, 480))  # Better resolution for pose detection
    except Exception as e:
        logging.error(f"Error creating VideoWriter: {e}")
        cap.release()
        return "Error creating video output.", None
    
    elbow_angles = []
    knee_angles = []
    racket_positions = []
    processed_frames_list = []
    
    frame_count = 0
    while processed_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
            
        processed_frames += 1
        
        # Balanced frame resizing for pose detection accuracy
        frame = cv2.resize(frame, (640, 480))
        
        try:
            annotated_frame, keypoints = detect_pose_ultra_fast(frame, model)
            processed_frames_list.append(annotated_frame)

            # Calculate angles if keypoints are detected
            if np.any(keypoints) and not np.all(keypoints == 0):
                # Elbow angle (balanced threshold)
                if all(k[2] > 0.3 for k in [keypoints[6], keypoints[8], keypoints[10]]):
                    shoulder = keypoints[6][:2]
                    elbow = keypoints[8][:2]
                    wrist = keypoints[10][:2]
                    elbow_angle = calculate_angle_fast(shoulder, elbow, wrist)
                    elbow_angles.append(elbow_angle)

                # Knee angle (balanced threshold)
                if all(k[2] > 0.3 for k in [keypoints[12], keypoints[14], keypoints[16]]):
                    hip = keypoints[12][:2]
                    knee = keypoints[14][:2]
                    ankle = keypoints[16][:2]
                    knee_angle = calculate_angle_fast(hip, knee, ankle)
                    knee_angles.append(knee_angle)

                # Racket position
                if keypoints[10][2] > 0.3:
                    racket_positions.append(keypoints[10][:2])
        except Exception as e:
            logging.error(f"Error processing frame {frame_count}: {e}")
            continue

    # Write processed frames
    for frame in processed_frames_list:
        out.write(frame)
    
    cap.release()
    out.release()

    # Move to final location
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'output_{video_hash[:8]}.mp4')
    shutil.move(temp_output_path, output_path)

    # Generate feedback
    feedback = generate_feedback_fast(elbow_angles, knee_angles, racket_positions)
    
    # Cache the result
    cache_result(video_hash, height, stroke_type, feedback, output_path)
    
    return " ".join(feedback), f'/static/output_{video_hash[:8]}.mp4'

def detect_pose_ultra_fast(frame, model):
    """Ultra-fast pose detection with minimal processing."""
    try:
        # Resize for faster processing
        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
        img = tf.cast(img, dtype=tf.int32)
        
        outputs = model(img)
        keypoints = outputs['output_0'].numpy().squeeze()
        
        # Draw skeleton for better visualization
        y, x, _ = frame.shape
        for keypoint in keypoints:
            ky, kx, kp_conf = keypoint
            if kp_conf > 0.3:  # Balanced confidence threshold
                cv2.circle(frame, (int(kx * x), int(ky * y)), 3, (0, 0, 255), -1)
        
        # Draw connections between keypoints
        connections = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
        for connection in connections:
            start_point = keypoints[connection[0]]
            end_point = keypoints[connection[1]]
            if start_point[2] > 0.3 and end_point[2] > 0.3:
                cv2.line(frame,
                         (int(start_point[1] * x), int(start_point[0] * y)),
                         (int(end_point[1] * x), int(end_point[0] * y)),
                         (0, 255, 0), 2)
        
        return frame, keypoints
    except Exception as e:
        logging.error(f"Error during pose detection: {e}")
        return frame, np.zeros((17,3))

def calculate_angle_fast(a, b, c):
    """Fast angle calculation with error handling."""
    try:
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    except Exception as e:
        logging.error(f"Error calculating angle: {e}")
        return 0.0

def generate_feedback_fast(elbow_angles, knee_angles, racket_positions):
    """Generate fast feedback."""
    feedback = []
    
    if not elbow_angles and not knee_angles:
        return ["⚠️ Limited pose data detected. Try a clearer video for better analysis."]
    
    # Simplified feedback generation
    if elbow_angles:
        avg_elbow_angle = np.mean(elbow_angles)
        if avg_elbow_angle < 80:
            feedback.append("🟧 Keep your elbow higher during the swing for better power.")
        else:
            feedback.append("🟩 Good elbow position!")
    
    if knee_angles:
        avg_knee_angle = np.mean(knee_angles)
        if avg_knee_angle > 160:
            feedback.append("🟧 Bend your knees more for better balance.")
        else:
            feedback.append("🟩 Good knee bend!")
    
    if racket_positions and len(racket_positions) > 1:
        try:
            velocities = [np.linalg.norm(np.array(racket_positions[i]) - np.array(racket_positions[i - 1]))
                          for i in range(1, len(racket_positions))]
            avg_velocity = np.mean(velocities)
            
            if avg_velocity < 5:
                feedback.append("🟧 Work on accelerating through your swing.")
            else:
                feedback.append("🟩 Good swing speed!")
        except Exception:
            feedback.append("🟨 Swing analysis completed.")
    
    if not feedback:
        feedback.append("✅ Analysis complete! Keep practicing.")
    
    return feedback

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
    start_time = time.time()
    
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
    
    # Save file with timestamp
    filename = f"upload_{int(time.time())}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        feedback, video_path = analyze_swing_ultra_fast(file_path, height, stroke_type)
        
        processing_time = time.time() - start_time
        logging.info(f"Analysis completed in {processing_time:.2f} seconds")
        
        return jsonify({
            'feedback': feedback,
            'video_path': video_path,
            'processing_time': round(processing_time, 2),
            'optimization': 'ultra_fast_mode'
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
    cache_size = len(os.listdir(app.config['CACHE_FOLDER'])) if os.path.exists(app.config['CACHE_FOLDER']) else 0
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_status,
        'platform': 'DigitalOcean App Platform - Ultra Optimized',
        'optimizations': [
            'Aggressive frame skipping (every 8th frame)',
            'Limited to 15 frames maximum',
            '320x240 output resolution',
            'Intelligent caching system',
            '25MB file size limit',
            'Focused frame selection (middle portion)',
            'Parallel processing support'
        ],
        'cache_entries': cache_size,
        'performance': 'Ultra Fast Mode'
    })

# Initialize database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
