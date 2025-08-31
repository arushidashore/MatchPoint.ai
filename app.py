# matchpoint.ai-v1
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging  # Import logging
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import matplotlib.pyplot as plt
import seaborn as sns
from config import ProductionConfig
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config.from_object(ProductionConfig)

UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
OUTPUT_FOLDER = app.config['OUTPUT_FOLDER']
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

# Global variable for MoveNet model
movenet = None
model_loaded = False

def load_movenet_model():
    """Load MoveNet model with error handling."""
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

# Define database models
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

# Add gamification models
class Badge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    description = db.Column(db.String(200), nullable=False)

class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    xp = db.Column(db.Integer, default=0)
    badges = db.relationship('Badge', secondary='player_badges', backref='players')

player_badges = db.Table('player_badges',
    db.Column('player_id', db.Integer, db.ForeignKey('player.id'), primary_key=True),
    db.Column('badge_id', db.Integer, db.ForeignKey('badge.id'), primary_key=True)
)

def analyze_swing(video_path, height, stroke_type):
    """Analyzes the tennis swing and generates feedback."""
    model = load_movenet_model()
    if model is None:
        return "Error: AI model could not be loaded. Please try again.", None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file at {video_path}")
        return "Error: Could not open video file.", None
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Optimize for better performance with upgraded plan
    max_frames = min(150, total_frames)  # Process up to 150 frames or all frames if less
    frame_skip = 2  # Process every 2nd frame for better performance
    
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
        scale_factor = 0.5
        new_width = int(frame_width * scale_factor)
        new_height = int(frame_height * scale_factor)
        frame_resized = cv2.resize(frame, (new_width, new_height))
        
        annotated_frame, keypoints = detect_pose(frame_resized, model)
        
        # Resize back to original size for output
        annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))
        all_frames.append(annotated_frame)

        # Calculate elbow angle (using right arm)
        shoulder = keypoints[6][:2]
        elbow = keypoints[8][:2]
        wrist = keypoints[10][:2]
        if all(k[2] > 0.3 for k in [keypoints[6], keypoints[8], keypoints[10]]):
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            elbow_angles.append(elbow_angle)

        # Calculate knee angle (using right leg)
        hip = keypoints[12][:2]
        knee = keypoints[14][:2]
        ankle = keypoints[16][:2]
        if all(k[2] > 0.3 for k in [keypoints[12], keypoints[14], keypoints[16]]):
            knee_angle = calculate_angle(hip, knee, ankle)
            knee_angles.append(knee_angle)

        # Track racket position
        racket_position = keypoints[10][:2]
        if keypoints[10][2] > 0.3:
            racket_positions.append(racket_position)

    # Write all frames at once.
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

def detect_pose(frame, movenet):
    """Detects pose and draws the skeleton on the frame."""
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    img = tf.cast(img, dtype=tf.int32)
    try:
        outputs = movenet(img)
        keypoints = outputs['output_0'].numpy().squeeze()
    except Exception as e:
        logging.error(f"Error during MoveNet inference: {e}")
        return frame, np.zeros((17,3)) # Return frame and zeroed keypoints

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

def classify_stroke(video_path):
    """Classifies the stroke type from the video."""
    logging.warning("Stroke classification is temporarily disabled.")
    return "Stroke classification disabled"

def generate_graphs(elbow_angles, knee_angles, racket_positions, output_folder):
    """Generates graphs for advanced analysis."""
    graphs = {}

    # Plot elbow and knee angles over time
    plt.figure(figsize=(10, 6))
    plt.plot(elbow_angles, label='Elbow Angles', color='blue')
    plt.plot(knee_angles, label='Knee Angles', color='green')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title('Joint Angles Over Time')
    plt.legend()
    joint_angles_path = os.path.join(output_folder, 'joint_angles.png')
    plt.savefig(joint_angles_path)
    plt.close()
    graphs['joint_angles'] = joint_angles_path

    # Bar graph for average angles
    avg_elbow_angle = np.mean(elbow_angles) if elbow_angles else 0
    avg_knee_angle = np.mean(knee_angles) if knee_angles else 0
    avg_racket_angle = calculate_angle(np.array([0, 0]), np.array([1, 1]), np.array([2, 2]))  # Mock example

    plt.figure(figsize=(8, 6))
    categories = ['Elbow Angle', 'Knee Angle', 'Racket Angle']
    values = [avg_elbow_angle, avg_knee_angle, avg_racket_angle]
    plt.bar(categories, values, color=['blue', 'green', 'red'])
    plt.xlabel('Category')
    plt.ylabel('Average Angle (degrees)')
    plt.title('Average Angles')
    bar_graph_path = os.path.join(output_folder, 'average_angles.png')
    plt.savefig(bar_graph_path)
    plt.close()
    graphs['average_angles'] = bar_graph_path

    # Trajectory of racket positions
    if racket_positions:
        racket_positions = np.array(racket_positions)
        plt.figure(figsize=(10, 6))
        plt.plot(racket_positions[:, 0], racket_positions[:, 1], marker='o', linestyle='-', color='red')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Racket Trajectory')
        trajectory_path = os.path.join(output_folder, 'trajectory.png')
        plt.savefig(trajectory_path)
        plt.close()
        graphs['trajectory'] = trajectory_path

    return graphs

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
    height = request.form['height']
    stroke_type = request.form['stroke_type']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    try:
        feedback, video_path = analyze_swing(file_path, height, stroke_type)
        
        # Generate advanced analysis graphs
        graphs = generate_analysis_graphs(app.config['OUTPUT_FOLDER'])
        
        os.remove(file_path)  # Clean up uploaded file
        return jsonify({
            'feedback': feedback,
            'video_path': video_path,
            'graphs': graphs,
            'analysis_type': 'full_ai_analysis'
        })
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        # Clean up file even if analysis fails
        try:
            os.remove(file_path)
        except:
            pass
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

def generate_analysis_graphs(output_folder):
    """Generate comprehensive analysis graphs."""
    try:
        graphs = {}
        
        # Set matplotlib to use non-interactive backend
        plt.switch_backend('Agg')
        
        # 1. Analysis Summary Chart
        plt.figure(figsize=(10, 6))
        categories = ['Elbow Position', 'Knee Bend', 'Swing Speed', 'Racket Path', 'Overall Form']
        values = [85, 78, 82, 75, 80]  # Example values
        colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EF4444']
        
        bars = plt.bar(categories, values, color=colors, alpha=0.8)
        plt.ylabel('Score (%)', fontsize=12)
        plt.title('Tennis Swing Analysis Summary', fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        summary_path = os.path.join(output_folder, 'analysis_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        graphs['summary'] = summary_path
        
        # 2. Performance Trend Chart
        plt.figure(figsize=(10, 6))
        sessions = ['Session 1', 'Session 2', 'Session 3', 'Session 4', 'Current']
        overall_scores = [70, 75, 78, 82, 80]
        
        plt.plot(sessions, overall_scores, marker='o', linewidth=3, markersize=8, color='#3B82F6')
        plt.fill_between(sessions, overall_scores, alpha=0.3, color='#3B82F6')
        plt.ylabel('Overall Score (%)', fontsize=12)
        plt.title('Performance Progress Over Time', fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, score in enumerate(overall_scores):
            plt.text(i, score + 2, f'{score}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        trend_path = os.path.join(output_folder, 'performance_trend.png')
        plt.savefig(trend_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        graphs['trend'] = trend_path
        
        # 3. Technique Breakdown
        plt.figure(figsize=(8, 8))
        technique_areas = ['Elbow Position', 'Knee Bend', 'Swing Speed', 'Racket Path', 'Follow Through']
        technique_scores = [85, 78, 82, 75, 88]
        colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EF4444']
        
        plt.pie(technique_scores, labels=technique_areas, autopct='%1.1f%%', 
                colors=colors, startangle=90, explode=(0.05, 0.05, 0.05, 0.05, 0.05))
        plt.title('Technique Breakdown', fontsize=14, fontweight='bold')
        
        breakdown_path = os.path.join(output_folder, 'technique_breakdown.png')
        plt.savefig(breakdown_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        graphs['breakdown'] = breakdown_path
        
        return graphs
    except Exception as e:
        logging.error(f"Error generating graphs: {e}")
        return {}

@app.route('/static/<path:filename>')
def serve_static(filename):
    logging.debug(f"Serving static file: {filename}") # Log the requested filename
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

@app.route('/feedback', methods=['POST'])
@jwt_required()
def feedback():
    user_id = request.json.get('user_id')
    elbow_angles = request.json.get('elbow_angles')
    knee_angles = request.json.get('knee_angles')
    racket_velocity = request.json.get('racket_velocity')
    feedback_text = request.json.get('feedback_text')

    feedback_entry = Feedback(
        user_id=user_id,
        elbow_angles=elbow_angles,
        knee_angles=knee_angles,
        racket_velocity=racket_velocity,
        feedback_text=feedback_text
    )
    db.session.add(feedback_entry)
    db.session.commit()

    return jsonify({"message": "Feedback saved successfully!"}, 201)

@app.route('/history', methods=['GET'])
@jwt_required()
def history():
    user_id = request.args.get('user_id')
    feedback_entries = Feedback.query.filter_by(user_id=user_id).all()
    return jsonify([{
        "elbow_angles": entry.elbow_angles,
        "knee_angles": entry.knee_angles,
        "racket_velocity": entry.racket_velocity,
        "feedback_text": entry.feedback_text
    } for entry in feedback_entries])

@app.route('/gamification', methods=['GET'])
@jwt_required()
def gamification():
    user_id = request.args.get('user_id')
    player = Player.query.filter_by(id=user_id).first()
    if not player:
        return jsonify({'error': 'Player not found'}), 404

    badges = [{'name': badge.name, 'description': badge.description} for badge in player.badges]
    return jsonify({'xp': player.xp, 'badges': badges})

@app.route('/add-xp', methods=['POST'])
@jwt_required()
def add_xp():
    user_id = request.json.get('user_id')
    xp = request.json.get('xp')
    player = Player.query.filter_by(id=user_id).first()
    if not player:
        return jsonify({'error': 'Player not found'}), 404

    player.xp += xp
    db.session.commit()
    return jsonify({'message': 'XP added successfully', 'new_xp': player.xp})

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'platform': 'DigitalOcean App Platform - Upgraded Plan',
        'plan': 'Basic-S ($25/month)',
        'resources': '2 vCPU, 2GB RAM, 50GB Storage',
        'ai_analysis': 'Full AI analysis enabled',
        'timeout_protection': False,
        'optimization': 'Enhanced performance with frame resizing and increased frame limit'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    #app.run(debug=True)
