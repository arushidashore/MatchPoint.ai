# MatchPoint.ai - Streaming Progress Version
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
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-jwt-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///matchpoint.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB for 3-minute videos

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

# Global variables
movenet = None
model_loaded = False
processing_jobs = {}  # Store active processing jobs
executor = ThreadPoolExecutor(max_workers=2)

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

def analyze_swing_with_progress(job_id, video_path, height, stroke_type, progress_queue):
    """Analyze swing with progress updates."""
    try:
        model = load_movenet_model()
        if model is None:
            progress_queue.put({'status': 'error', 'message': 'Model not available'})
            return
        
        progress_queue.put({'status': 'progress', 'progress': 10, 'message': 'Loading video...'})
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            progress_queue.put({'status': 'error', 'message': 'Could not open video file'})
            return
        
        # Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_queue.put({'status': 'progress', 'progress': 20, 'message': f'Video loaded: {total_frames} frames'})
        
        # Optimization settings for longer videos (up to 3 minutes)
        frame_skip = 8  # Process every 8th frame for longer videos
        max_frames = 25  # Increased to 25 frames for better analysis
        processed_frames = 0
        
        # Create temporary output file
        temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_output_path = temp_output.name
        temp_output.close()
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps//frame_skip, (480, 360))
        except Exception as e:
            progress_queue.put({'status': 'error', 'message': 'Error creating video output'})
            cap.release()
            return
        
        progress_queue.put({'status': 'progress', 'progress': 30, 'message': 'Starting pose detection...'})
        
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
            
            # Update progress
            progress = 30 + (processed_frames / max_frames) * 60
            progress_queue.put({
                'status': 'progress', 
                'progress': int(progress), 
                'message': f'Processing frame {processed_frames}/{max_frames}'
            })
            
            # Resize frame for faster processing
            frame = cv2.resize(frame, (480, 360))
            
            try:
                annotated_frame, keypoints = detect_pose_with_progress(frame, model)
                processed_frames_list.append(annotated_frame)

                # Calculate angles if keypoints are detected
                if np.any(keypoints) and not np.all(keypoints == 0):
                    # Elbow angle
                    if all(k[2] > 0.2 for k in [keypoints[6], keypoints[8], keypoints[10]]):
                        shoulder = keypoints[6][:2]
                        elbow = keypoints[8][:2]
                        wrist = keypoints[10][:2]
                        elbow_angle = calculate_angle_fast(shoulder, elbow, wrist)
                        elbow_angles.append(elbow_angle)

                    # Knee angle
                    if all(k[2] > 0.2 for k in [keypoints[12], keypoints[14], keypoints[16]]):
                        hip = keypoints[12][:2]
                        knee = keypoints[14][:2]
                        ankle = keypoints[16][:2]
                        knee_angle = calculate_angle_fast(hip, knee, ankle)
                        knee_angles.append(knee_angle)

                    # Racket position
                    if keypoints[10][2] > 0.2:
                        racket_positions.append(keypoints[10][:2])
            except Exception as e:
                logging.error(f"Error processing frame {frame_count}: {e}")
                continue

        progress_queue.put({'status': 'progress', 'progress': 90, 'message': 'Generating output video...'})
        
        # Write processed frames
        for frame in processed_frames_list:
            out.write(frame)
        
        cap.release()
        out.release()

        # Move to final location
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'output_{job_id}.mp4')
        shutil.move(temp_output_path, output_path)

        # Generate feedback
        feedback = generate_feedback_fast(elbow_angles, knee_angles, racket_positions)
        
        progress_queue.put({
            'status': 'complete', 
            'progress': 100, 
            'message': 'Analysis complete!',
            'feedback': feedback,
            'video_path': f'/static/output_{job_id}.mp4'
        })
        
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        progress_queue.put({'status': 'error', 'message': str(e)})

def detect_pose_with_progress(frame, model):
    """Pose detection with progress tracking."""
    try:
        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
        img = tf.cast(img, dtype=tf.int32)
        
        outputs = model(img)
        keypoints = outputs['output_0'].numpy().squeeze()
        
        # Draw skeleton
        y, x, _ = frame.shape
        for keypoint in keypoints:
            ky, kx, kp_conf = keypoint
            if kp_conf > 0.2:
                cv2.circle(frame, (int(kx * x), int(ky * y)), 2, (0, 0, 255), -1)
        
        return frame, keypoints
    except Exception as e:
        logging.error(f"Error during pose detection: {e}")
        return frame, np.zeros((17,3))

def calculate_angle_fast(a, b, c):
    """Fast angle calculation."""
    try:
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    except Exception:
        return 0.0

def generate_feedback_fast(elbow_angles, knee_angles, racket_positions):
    """Generate feedback."""
    feedback = []
    
    if not elbow_angles and not knee_angles:
        return ["⚠️ Limited pose data detected. Try a clearer video for better analysis."]
    
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
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save file
    filename = f"upload_{job_id}_{file.filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Create progress queue
    progress_queue = queue.Queue()
    processing_jobs[job_id] = {
        'status': 'processing',
        'progress_queue': progress_queue,
        'start_time': time.time()
    }
    
    # Start analysis in background
    executor.submit(analyze_swing_with_progress, job_id, file_path, height, stroke_type, progress_queue)
    
    return jsonify({
        'job_id': job_id,
        'status': 'started',
        'message': 'Analysis started'
    })

@app.route('/progress/<job_id>')
def get_progress(job_id):
    """Get progress updates for a job."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    progress_queue = job['progress_queue']
    
    try:
        # Get latest progress update
        progress_data = progress_queue.get_nowait()
        
        if progress_data['status'] == 'complete':
            # Clean up job
            del processing_jobs[job_id]
            
            # Clean up uploaded file
            try:
                file_pattern = f"upload_{job_id}_"
                for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                    if filename.startswith(file_pattern):
                        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            except Exception as e:
                logging.error(f"Error cleaning up file: {e}")
        
        return jsonify(progress_data)
    
    except queue.Empty:
        return jsonify({
            'status': 'processing',
            'progress': 0,
            'message': 'Processing...'
        })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

@app.route('/health')
def health_check():
    model_status = load_movenet_model() is not None
    active_jobs = len([job for job in processing_jobs.values() if job['status'] == 'processing'])
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_status,
        'platform': 'DigitalOcean App Platform - Streaming Mode',
        'active_jobs': active_jobs,
        'features': [
            'Real-time progress streaming',
            'Background processing',
            'Job queue management',
            'Optimized frame processing',
            'Progress tracking'
        ],
        'performance': 'Streaming Mode'
    })

# Initialize database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
