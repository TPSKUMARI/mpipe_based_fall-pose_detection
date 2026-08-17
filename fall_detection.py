import cv2
import mediapipe as mp
import numpy as np
import math
import time
import sqlite3
import pytz
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from datetime import datetime

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# System configuration
SYSTEM_CONFIG = {
    'frame_rate': 30,
    'confidence_threshold': 0.5,
    'temporal_window': 90,
    'fall_confirmation_time': 2.0,
    'recovery_detection_time': 0.5,
    'bbox_ratio_threshold': 0.5,
    'torso_angle_threshold': 60,
    'head_drop_threshold': 0.3,
    'velocity_threshold': 800,
    'min_active_rules': 2,
    'head_angle_threshold': 45,
    'height_reduction_threshold': 0.6,
    'head_displacement_threshold': 0.3,
}

# MediaPipe keypoint indices
CRITICAL_KEYPOINTS = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_hip': 23,
    'right_hip': 24,
    'left_ankle': 27,
    'right_ankle': 28
}

@dataclass
class KeypointData:
    position: Tuple[float, float]
    confidence: float

class FallDetectionHistory:
    def __init__(self, max_size=90):
        self.max_size = max_size
        self.bbox_history = deque(maxlen=max_size)
        self.keypoints_history = deque(maxlen=max_size)
        self.fall_detections = deque(maxlen=max_size)
        self.head_positions = deque(maxlen=max_size)
        self.velocity_history = deque(maxlen=max_size)
        self.angle_history = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.person_heights = deque(maxlen=max_size)
        self.min_head_y = float('inf')

class EnhancedFallDetector:
    def __init__(self):
        self.history = FallDetectionHistory()

    def detect_fall_bbox_ratio(self, bbox, bbox_history, frame_count=5):
        if not bbox:
            return False
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0:
            return False
        ratio = height / width
        if ratio < SYSTEM_CONFIG['bbox_ratio_threshold']:
            if len(bbox_history) >= frame_count:
                recent = list(bbox_history)[-frame_count:]
                recent_ratios = []
                for hist_bbox in recent:
                    if hist_bbox and hist_bbox[2] - hist_bbox[0] > 0:
                        r = (hist_bbox[3] - hist_bbox[1]) / (hist_bbox[2] - hist_bbox[0])
                        recent_ratios.append(r)
                return sum(1 for r in recent_ratios if r < SYSTEM_CONFIG['bbox_ratio_threshold']) >= 3
            return True
        return False

    def calculate_torso_angle(self, keypoints):
        try:
            need = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
            if not all(key in keypoints for key in need):
                return None, False
            mid_shoulder = [
                (keypoints['left_shoulder'].position[0] + keypoints['right_shoulder'].position[0]) / 2,
                (keypoints['left_shoulder'].position[1] + keypoints['right_shoulder'].position[1]) / 2,
            ]
            mid_hip = [
                (keypoints['left_hip'].position[0] + keypoints['right_hip'].position[0]) / 2,
                (keypoints['left_hip'].position[1] + keypoints['right_hip'].position[1]) / 2,
            ]
            dx = mid_hip[0] - mid_shoulder[0]
            dy = mid_hip[1] - mid_shoulder[1]
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return None, False
            angle = math.degrees(math.atan2(abs(dy), abs(dx)))
            angle_from_vertical = abs(90 - angle)
            return angle_from_vertical, angle_from_vertical > SYSTEM_CONFIG['torso_angle_threshold']
        except Exception:
            return None, False

    def calculate_head_angle(self, keypoints):
        if 'nose' not in keypoints or 'left_shoulder' not in keypoints or 'right_shoulder' not in keypoints:
            return None, False
        neck = [
            (keypoints['left_shoulder'].position[0] + keypoints['right_shoulder'].position[0]) / 2,
            (keypoints['left_shoulder'].position[1] + keypoints['right_shoulder'].position[1]) / 2,
        ]
        nose = keypoints['nose'].position
        dx = nose[0] - neck[0]
        dy = nose[1] - neck[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None, False
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        angle_from_vertical = abs(90 - angle)
        return angle_from_vertical, angle_from_vertical > SYSTEM_CONFIG['head_angle_threshold']

    def detect_head_drop(self, keypoints, person_height):
        head_kp = keypoints.get('nose')
        if not head_kp:
            if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
                head_kp = KeypointData(
                    position=[
                        (keypoints['left_shoulder'].position[0] + keypoints['right_shoulder'].position[0]) / 2,
                        (keypoints['left_shoulder'].position[1] + keypoints['right_shoulder'].position[1]) / 2,
                    ],
                    confidence=min(keypoints['left_shoulder'].confidence, keypoints['right_shoulder'].confidence)
                )
        if not head_kp or person_height <= 0:
            return False
        self.history.head_positions.append(head_kp.position)
        self.history.min_head_y = min(self.history.min_head_y, head_kp.position[1])
        if len(self.history.head_positions) < 2:
            return False
        y_drop = self.history.head_positions[-1][1] - self.history.head_positions[-2][1]
        drop_ratio = y_drop / person_height
        return drop_ratio > SYSTEM_CONFIG['head_drop_threshold']

    def detect_head_displacement(self, head_y, person_height):
        if person_height <= 0 or self.history.min_head_y == float('inf'):
            return False
        displacement = (head_y - self.history.min_head_y) / person_height
        return displacement > SYSTEM_CONFIG['head_displacement_threshold']

    def detect_height_reduction(self, current_height):
        self.history.person_heights.append(current_height)
        if len(self.history.person_heights) < 1 or current_height <= 0:
            return False
        max_height = max(self.history.person_heights)
        return current_height < SYSTEM_CONFIG['height_reduction_threshold'] * max_height

    def calculate_keypoint_velocity(self, keypoints, frame_width, frame_height):
        if len(self.history.keypoints_history) < 1:
            return 0
        velocities = []
        for key in ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
            current_kp = keypoints.get(key)
            prev_kp = self.history.keypoints_history[-1].get(key)
            if current_kp and prev_kp:
                dx = current_kp.position[0] - prev_kp.position[0]
                dy = current_kp.position[1] - prev_kp.position[1]
                dx_norm = dx / frame_width
                dy_norm = dy / frame_height
                velocity = math.sqrt(dx_norm**2 + dy_norm**2) * SYSTEM_CONFIG['frame_rate'] * 1000
                velocities.append(velocity)
        return sum(velocities) / len(velocities) if velocities else 0

    def detect_rapid_movement(self, velocity):
        self.history.velocity_history.append(velocity)
        return velocity > SYSTEM_CONFIG['velocity_threshold']

    def estimate_person_height(self, keypoints):
        nose = keypoints.get('nose')
        left_ankle = keypoints.get('left_ankle')
        right_ankle = keypoints.get('right_ankle')
        if not nose:
            return 0
        ankle = left_ankle if left_ankle and (not right_ankle or left_ankle.position[1] > right_ankle.position[1]) else right_ankle
        if not ankle:
            return 0
        return abs(ankle.position[1] - nose.position[1])

    def validate_fall_duration(self):
        if len(self.history.fall_detections) < 10:
            return False
        frame_rate = SYSTEM_CONFIG['frame_rate']
        min_duration_frames = int(2 * frame_rate)
        max_duration_frames = int(10 * frame_rate)
        consecutive = 0
        for detection in reversed(self.history.fall_detections):
            if detection:
                consecutive += 1
            else:
                break
        if min_duration_frames <= consecutive <= max_duration_frames:
            return True
        if consecutive > max_duration_frames:
            detection_window = list(self.history.fall_detections)[-max_duration_frames:]
            for i in range(len(detection_window) - 15 + 1):
                if sum(1 for d in detection_window[i:i+15]) == 0:
                    return False
            return True
        return False

    def detect_fall(self, keypoints, bbox, frame_width, frame_height):
        current_time = time.time()
        self.history.timestamps.append(current_time)
        self.history.bbox_history.append(bbox)

        bbox_fall = self.detect_fall_bbox_ratio(bbox, self.history.bbox_history)
        angle_value, torso_fall = self.calculate_torso_angle(keypoints)
        person_height = self.estimate_person_height(keypoints)
        head_fall = self.detect_head_drop(keypoints, person_height) if person_height > 0 else False
        velocity = self.calculate_keypoint_velocity(keypoints, frame_width, frame_height)
        velocity_fall = self.detect_rapid_movement(velocity)
        duration_valid = self.validate_fall_duration()

        head_angle_value, head_angle_fall = self.calculate_head_angle(keypoints)
        height_reduction_fall = self.detect_height_reduction(person_height)
        head_y = keypoints.get('nose').position[1] if 'nose' in keypoints else 0
        head_disp_fall = self.detect_head_displacement(head_y, person_height) if person_height > 0 else False

        scores = {
            'bbox_ratio': bool(bbox_fall),
            'torso_angle': bool(torso_fall),
            'head_drop': bool(head_fall),
            'velocity': bool(velocity_fall),
            'duration': bool(duration_valid),
            'head_angle': bool(head_angle_fall),
            'height_reduction': bool(height_reduction_fall),
            'head_displacement': bool(head_disp_fall),
        }

        active_rules = sum(int(v) for v in scores.values())
        is_fall = active_rules >= SYSTEM_CONFIG['min_active_rules']

        self.history.keypoints_history.append(keypoints)
        self.history.fall_detections.append(is_fall)

        drop_ratio = 0.0
        if person_height > 0 and len(self.history.head_positions) >= 2:
            y_drop = self.history.head_positions[-1][1] - self.history.head_positions[-2][1]
            drop_ratio = y_drop / person_height

        triggered_rules = [rule for rule, v in scores.items() if v]

        max_height = max(self.history.person_heights) if self.history.person_heights else 1.0
        height_reduction_value = person_height / max_height if max_height > 0 else 0.0
        head_displacement_value = (head_y - self.history.min_head_y) / person_height if person_height > 0 and self.history.min_head_y != float('inf') else 0.0

        return {
            'is_fall': is_fall,
            'confidence': active_rules / len(scores),
            'individual_scores': {k: int(v) for k, v in scores.items()},
            'triggered_rules': triggered_rules,
            'person_height': person_height,
            'velocity': velocity,
            'torso_angle_value': angle_value if angle_value is not None else 0.0,
            'head_drop_value': drop_ratio,
            'head_angle_value': head_angle_value if head_angle_value is not None else 0.0,
            'height_reduction_value': height_reduction_value,
            'head_displacement_value': head_displacement_value,
        }

def filter_keypoints_by_confidence(landmarks, w, h, model_type='mediapipe'):
    threshold = {'mediapipe': 0.5, 'openpose': 0.3, 'yolo': 0.4}.get(model_type, 0.5)
    filtered_keypoints = {}
    for name, idx in CRITICAL_KEYPOINTS.items():
        if idx < len(landmarks) and landmarks[idx].visibility >= threshold:
            filtered_keypoints[name] = KeypointData(
                position=(landmarks[idx].x * w, landmarks[idx].y * h),
                confidence=landmarks[idx].visibility
            )
    return filtered_keypoints

def draw_checkbox_row(img, x, y, label, active, box_size=14):
    top_left = (x, y - box_size)
    bottom_right = (x + box_size, y)
    color_box = (0, 0, 0)
    cv2.rectangle(img, top_left, bottom_right, color_box, 1)

    if active:
        p1 = (x + int(0.2 * box_size), y - int(0.5 * box_size))
        p2 = (x + int(0.45 * box_size), y - int(0.2 * box_size))
        p3 = (x + int(0.85 * box_size), y - int(0.9 * box_size))
        cv2.line(img, p1, p2, (0, 0, 180), 2)
        cv2.line(img, p2, p3, (0, 0, 180), 2)
        text_color = (0, 0, 180)
    else:
        p1 = (x + int(0.2 * box_size), y - int(0.2 * box_size))
        p2 = (x + int(0.8 * box_size), y - int(0.8 * box_size))
        p3 = (x + int(0.2 * box_size), y - int(0.8 * box_size))
        p4 = (x + int(0.8 * box_size), y - int(0.2 * box_size))
        cv2.line(img, p1, p2, (0, 180, 0), 2)
        cv2.line(img, p3, p4, (0, 180, 0), 2)
        text_color = (0, 180, 0)

    cv2.putText(img, f"{label}", (x + box_size + 6, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

# Trackbar callbacks
def update_bbox_ratio(val):
    SYSTEM_CONFIG['bbox_ratio_threshold'] = val / 100.0

def update_torso_angle(val):
    SYSTEM_CONFIG['torso_angle_threshold'] = val

def update_head_drop(val):
    SYSTEM_CONFIG['head_drop_threshold'] = val / 100.0

def update_velocity(val):
    SYSTEM_CONFIG['velocity_threshold'] = val

def update_min_active_rules(val):
    SYSTEM_CONFIG['min_active_rules'] = max(1, min(8, val))

def update_head_angle(val):
    SYSTEM_CONFIG['head_angle_threshold'] = val

def update_height_reduction(val):
    SYSTEM_CONFIG['height_reduction_threshold'] = val / 100.0

def update_head_displacement(val):
    SYSTEM_CONFIG['head_displacement_threshold'] = val / 100.0

class FallDetectionSystem:
    """Fall Detection System that can be integrated with the orchestrator"""
    
    def __init__(self, window_name="Fall Detection", show_gui=True):
        self.window_name = window_name
        self.show_gui = show_gui
        self.fall_detector = EnhancedFallDetector()
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Initialize database
        self.init_database()
        
        # GUI setup
        if self.show_gui:
            self.setup_gui()
        
        # Performance tracking
        self.fall_alert_frames = 0
        self.last_fall_log_time = 0
        self.frame_count = 0
        
        # Status tracking
        self.current_status = {
            'fall_detected': False,
            'confidence': 0.0,
            'triggered_rules': [],
            'velocity': 0.0,
            'frames_processed': 0
        }
        
        print(f"✅ Fall Detection System initialized (GUI: {show_gui})")
    
    def init_database(self):
        """Initialize database for fall detection logs"""
        try:
            self.conn = sqlite3.connect('fall_detection_logs.db', check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS fall_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fall_detected BOOLEAN NOT NULL,
                    confidence REAL NOT NULL,
                    triggered_rules TEXT,
                    velocity REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
            print("📁 Fall detection database initialized")
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
    
    def setup_gui(self):
        """Setup GUI window and trackbars"""
        cv2.namedWindow(self.window_name)
        
        # Create trackbars for parameter adjustment
        cv2.createTrackbar("BBox Ratio (x100)", self.window_name, 
                          int(SYSTEM_CONFIG['bbox_ratio_threshold'] * 100), 100, update_bbox_ratio)
        cv2.createTrackbar("Torso Angle", self.window_name, 
                          int(SYSTEM_CONFIG['torso_angle_threshold']), 90, update_torso_angle)
        cv2.createTrackbar("Head Drop (x100)", self.window_name, 
                          int(SYSTEM_CONFIG['head_drop_threshold'] * 100), 100, update_head_drop)
        cv2.createTrackbar("Velocity", self.window_name, 
                          int(SYSTEM_CONFIG['velocity_threshold']), 2000, update_velocity)
        cv2.createTrackbar("Min Active Rules", self.window_name, 
                          int(SYSTEM_CONFIG['min_active_rules']), 8, update_min_active_rules)
        cv2.createTrackbar("Head Angle", self.window_name, 
                          int(SYSTEM_CONFIG['head_angle_threshold']), 90, update_head_angle)
        cv2.createTrackbar("Height Reduction (x100)", self.window_name, 
                          int(SYSTEM_CONFIG['height_reduction_threshold'] * 100), 100, update_height_reduction)
        cv2.createTrackbar("Head Displacement (x100)", self.window_name, 
                          int(SYSTEM_CONFIG['head_displacement_threshold'] * 100), 100, update_head_displacement)
    
    def log_fall_event(self, fall_info):
        """Log fall detection event to database"""
        try:
            slst_tz = pytz.timezone('Asia/Colombo')
            current_time = datetime.now(slst_tz).strftime('%Y-%m-%d %H:%M:%S')
            
            self.cursor.execute('''
                INSERT INTO fall_logs (fall_detected, confidence, triggered_rules, velocity, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                fall_info['is_fall'],
                fall_info['confidence'],
                str(fall_info['triggered_rules']),
                fall_info['velocity'],
                current_time
            ))
            self.conn.commit()
        except Exception as e:
            print(f"❌ Fall logging error: {e}")
    
    def process_frame(self, frame_data):
        """Process a single frame for fall detection"""
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_count = frame_data['frame_count']
        
        self.frame_count = frame_count
        
        # Process with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        orig_h, orig_w = frame.shape[:2]
        fall_detected = False
        fall_info = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Compute bounding box
            bbox = None
            valid_landmarks = [lm for lm in landmarks if lm.visibility >= SYSTEM_CONFIG['confidence_threshold']]
            if valid_landmarks:
                x_coords = [lm.x * orig_w for lm in valid_landmarks]
                y_coords = [lm.y * orig_h for lm in valid_landmarks]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                bbox = (x1, y1, x2, y2)
            
            # Process keypoints for fall detection
            filtered_keypoints = filter_keypoints_by_confidence(landmarks, orig_w, orig_h, 'mediapipe')
            if filtered_keypoints and len(filtered_keypoints) >= 4:
                fall_info = self.fall_detector.detect_fall(filtered_keypoints, bbox, orig_w, orig_h)
                fall_detected = fall_info['is_fall']
                
                # Update current status
                self.current_status.update({
                    'fall_detected': fall_detected,
                    'confidence': fall_info['confidence'],
                    'triggered_rules': fall_info['triggered_rules'],
                    'velocity': fall_info['velocity'],
                    'frames_processed': frame_count
                })
                
                # Log fall event (with rate limiting)
                if fall_detected and (timestamp - self.last_fall_log_time) > 5.0:
                    self.log_fall_event(fall_info)
                    self.last_fall_log_time = timestamp
                    print(f"🚨 FALL DETECTED! Frame {frame_count} - Confidence: {fall_info['confidence']:.3f}")
        
        # Return results for combined display
        result = {
            'fall_detected': fall_detected,
            'confidence': fall_info['confidence'] if fall_info else 0.0,
            'pose_landmarks': results.pose_landmarks,
            'bbox': bbox if 'bbox' in locals() else None,
            'fall_info': fall_info
        }
        
        # Display GUI if enabled (for standalone mode)
        if self.show_gui:
            self.render_gui(image, results, fall_detected, fall_info, bbox if 'bbox' in locals() else None)
        
        return result
    
    def render_gui(self, image, results, fall_detected, fall_info, bbox):
        """Render the GUI display"""
        orig_h, orig_w = image.shape[:2]
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66) if not fall_detected else (245, 66, 230), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230) if not fall_detected else (245, 117, 66), thickness=2, circle_radius=2)
            )
        
        # Draw bounding box
        if bbox:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 0, 255) if fall_detected else (0, 255, 0), 2)
        
        # Main status
        status_text = "FALL DETECTED!" if fall_detected else "Normal Activity"
        status_color = (0, 0, 255) if fall_detected else (0, 255, 0)
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
        
        # Fall detection details
        if fall_info:
            y_offset = 60
            cv2.putText(image, f"Confidence: {fall_info['confidence']:.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 25
            
            cv2.putText(image, f"Velocity: {fall_info['velocity']:.0f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y_offset += 15
            cv2.putText(image, f"Torso: {fall_info['torso_angle_value']:.1f}°", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y_offset += 15
            cv2.putText(image, f"Head Drop: {fall_info['head_drop_value']:.2f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y_offset += 20
            
            # Rule Status with checkboxes
            rule_order = ['bbox_ratio', 'torso_angle', 'head_drop', 'velocity', 
                         'duration', 'head_angle', 'height_reduction', 'head_displacement']
            pretty = {
                'bbox_ratio': 'BBox', 'torso_angle': 'Torso', 'head_drop': 'Head', 'velocity': 'Vel',
                'duration': 'Dur', 'head_angle': 'H.Ang', 'height_reduction': 'Height', 'head_displacement': 'H.Disp',
            }
            
            x0 = 10
            for i, rule in enumerate(rule_order):
                if i % 4 == 0 and i > 0:  # New row every 4 items
                    y_offset += 18
                    x0 = 10
                val = fall_info['individual_scores'].get(rule, 0) > 0
                draw_checkbox_row(image, x0, y_offset, pretty[rule], val, box_size=10)
                x0 += 80
        
        # Fall alert flash effect
        if fall_detected:
            self.fall_alert_frames = min(self.fall_alert_frames + 1, 30)
            if self.fall_alert_frames % 10 < 5:
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (orig_w, orig_h), (0, 0, 255), -1)
                image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)
        else:
            self.fall_alert_frames = max(self.fall_alert_frames - 1, 0)
        
        # System info
        cv2.putText(image, f"Frame: {self.frame_count}", (10, orig_h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "Fall Detection System", (10, orig_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.imshow(self.window_name, image)
    
    def get_status(self):
        """Get current system status"""
        return self.current_status.copy()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
            if hasattr(self, 'temp_pose'):
                self.temp_pose.close()
            # Don't destroy windows here - let main orchestrator handle it
            print("Fall Detection System cleaned up")
        except Exception as e:
            print(f"Cleanup error: {e}")

# For standalone testing
if __name__ == "__main__":
    print("🔧 Testing Fall Detection System standalone...")
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera available for testing")
        exit(1)
    
    system = FallDetectionSystem(show_gui=True)
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_data = {
                'frame': cv2.flip(frame, 1),
                'timestamp': time.time(),
                'frame_count': frame_count
            }
            
            system.process_frame(frame_data)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⚡ Interrupted by user")
    finally:
        cap.release()
        system.cleanup()
        cv2.destroyAllWindows()
        print("👋 Fall Detection System test complete")