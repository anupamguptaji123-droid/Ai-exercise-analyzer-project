from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

app = Flask(__name__)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Global variables for tracking
rep_count = 0
exercise_type = 'squat'
form_quality = 'neutral'
stability_score = 0
angle_history = deque(maxlen=30)

# Color palette for visualization
COLOR_GOOD = (0, 255, 0)      # Green
COLOR_WARNING = (0, 165, 255)  # Orange
COLOR_BAD = (0, 0, 255)        # Red
COLOR_SKELETON = (0, 255, 255) # Cyan
COLOR_JOINTS = (255, 0, 255)   # Magenta

@app.route('/')
def home():
    return render_template('index.html')

def calculate_angle(point_a, point_b, point_c):
    """Calculate angle between three points"""
    a = np.array([point_a.x, point_a.y])
    b = np.array([point_b.x, point_b.y])
    c = np.array([point_c.x, point_c.y])
    
    ba = a - b
    bc = c - b
    
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return int(np.degrees(angle))

def draw_colorful_landmarks(frame, landmarks, connections):
    """Draw colored landmarks and connections on frame"""
    h, w, c = frame.shape
    
    # Define colors for different body parts
    colors = {
        'head': (255, 0, 255),      # Magenta
        'arms': (0, 255, 255),      # Cyan
        'torso': (255, 255, 0),     # Yellow
        'legs': (0, 255, 0)         # Green
    }
    
    # Body part mappings
    head_points = [0, 1, 2, 3, 4]
    arm_points = [5, 6, 7, 8, 9, 10]
    torso_points = [11, 12]
    leg_points = [13, 14, 15, 16]
    
    # Draw connections with gradients
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if start.visibility > 0.5 and end.visibility > 0.5:
                start_x = int(start.x * w)
                start_y = int(start.y * h)
                end_x = int(end.x * w)
                end_y = int(end.y * h)
                
                # Choose color based on body part
                if start_idx in head_points:
                    color = colors['head']
                elif start_idx in arm_points:
                    color = colors['arms']
                elif start_idx in torso_points:
                    color = colors['torso']
                else:
                    color = colors['legs']
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 3)
    
    # Draw keypoints with circles
    for idx, landmark in enumerate(landmarks):
        if landmark.visibility > 0.5:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            if idx in head_points:
                color = colors['head']
            elif idx in arm_points:
                color = colors['arms']
            elif idx in torso_points:
                color = colors['torso']
            else:
                color = colors['legs']
            
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.circle(frame, (x, y), 8, color, 2)

def analyze_squat(landmarks):
    """Analyze squat form"""
    left_knee = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
    right_knee = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
    
    avg_knee = (left_knee + right_knee) / 2
    
    if avg_knee < 90:
        return 'good', f'Great squat! Depth: {int(avg_knee)}°'
    elif avg_knee < 110:
        return 'warning', f'Squat deeper! Current: {int(avg_knee)}°'
    else:
        return 'bad', f'Go lower! Current: {int(avg_knee)}°'

def analyze_pushup(landmarks):
    """Analyze push-up form"""
    left_elbow = calculate_angle(landmarks[5], landmarks[7], landmarks[9])
    right_elbow = calculate_angle(landmarks[6], landmarks[8], landmarks[10])
    
    avg_elbow = (left_elbow + right_elbow) / 2
    
    if avg_elbow < 50:
        return 'good', f'Perfect form! Elbow angle: {int(avg_elbow)}°'
    elif avg_elbow < 70:
        return 'warning', f'Lower your body more! Angle: {int(avg_elbow)}°'
    else:
        return 'bad', f'Much deeper needed! Angle: {int(avg_elbow)}°'

def generate_frames():
    """Generate video frames with pose detection"""
    global rep_count, form_quality
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Add colorful background
        overlay = frame.copy()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        # Add title with gradient effect
        cv2.putText(frame, '🏋️  EXERCISE ANALYZER  🏋️', (50, 50),
                   cv2.FONT_HERSHEY_BOLD, 1.5, (0, 255, 255), 2)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Draw colorful landmarks
            draw_colorful_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Analyze form
            if exercise_type == 'squat':
                form_quality, feedback = analyze_squat(landmarks)
            elif exercise_type == 'pushup':
                form_quality, feedback = analyze_pushup(landmarks)
            else:
                form_quality = 'neutral'
                feedback = 'Analyzing form...'
            
            # Display feedback with color
            color = COLOR_GOOD if form_quality == 'good' else COLOR_WARNING if form_quality == 'warning' else COLOR_BAD
            cv2.putText(frame, feedback, (50, 700),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Add frame info
        frame_count += 1
        cv2.putText(frame, f'FPS: {int(cap.get(cv2.CAP_PROP_FPS))}', (w-200, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_exercise/<exercise>')
def set_exercise(exercise):
    """Set the exercise type"""
    global exercise_type
    exercise_type = exercise
    return jsonify({'status': 'success', 'exercise': exercise_type})

@app.route('/get_stats')
def get_stats():
    """Get current statistics"""
    return jsonify({
        'rep_count': rep_count,
        'form_quality': form_quality,
        'exercise_type': exercise_type
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)