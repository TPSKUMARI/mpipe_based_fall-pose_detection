import cv2
import mediapipe as mp
import numpy as np
import time
import sqlite3
import pytz
from datetime import datetime
from typing import Tuple

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a: list, b: list, c: list) -> float:
    """Calculate the angle between three points in 3D."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def nothing(x):
    """Trackbar callback placeholder"""
    pass

class PoseEstimationSystem:
    """Pose Estimation System that can be integrated with the orchestrator"""
    
    def __init__(self, window_name="Pose Estimation", show_gui=True):
        self.window_name = window_name
        self.show_gui = show_gui
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Initialize database
        self.init_database()
        
        # GUI setup
        if self.show_gui:
            self.setup_gui()
        
        # Pose estimation variables
        self.ankle_history_left = []
        self.ankle_history_right = []
        self.status_history = []
        
        # Timing variables
        self.last_status = None
        self.status_start_time = time.time()
        self.frame_count = 0
        
        # Configuration constants
        self.STABILITY_WINDOW = 3.0       # seconds to consider status stable
        self.MAX_LOG_DURATION = 300       # max log duration in seconds
        self.HISTORY_LENGTH = 40          # default history length
        
        # Current status tracking
        self.current_status = {
            'pose': 'No Detection',
            'confidence': 0.0,
            'knee_angle': 0.0,
            'body_span': 0.0,
            'movement_std': 0.0,
            'frames_processed': 0
        }
        
        print(f"✅ Pose Estimation System initialized (GUI: {show_gui})")
    
    def init_database(self):
        """Initialize database for pose estimation logs"""
        try:
            self.conn = sqlite3.connect('pose_estimation_logs.db', check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS pose_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detected_result TEXT NOT NULL,
                    duration_of_result_in_seconds REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
            print("📁 Pose estimation database initialized")
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
    
    def setup_gui(self):
        """Setup GUI window and trackbars"""
        cv2.namedWindow(self.window_name)
        
        # Create trackbars for parameter adjustment
        cv2.createTrackbar("Sleep Thresh (x100)", self.window_name, 22, 100, nothing)
        cv2.createTrackbar("Knee Sit Thresh", self.window_name, 123, 180, nothing)
        cv2.createTrackbar("Move Std Thresh (x1000)", self.window_name, 30, 50, nothing)
        cv2.createTrackbar("History Length", self.window_name, 40, 100, nothing)
    
    def log_status(self, status: str, duration: float):
        """Log the pose status and its duration to the database"""
        try:
            slst_tz = pytz.timezone('Asia/Colombo')
            current_time = datetime.now(slst_tz).strftime('%Y-%m-%d %H:%M:%S')
            
            for attempt in range(3):
                try:
                    self.cursor.execute('''
                        INSERT INTO pose_logs (detected_result, duration_of_result_in_seconds, timestamp)
                        VALUES (?, ?, ?)
                    ''', (status, duration, current_time))
                    self.conn.commit()
                    return
                except sqlite3.OperationalError as e:
                    if "locked" in str(e):
                        time.sleep(0.1)
                        continue
                    else:
                        raise
            print("⚠️ Failed to save pose record after retries.")
        except Exception as e:
            print(f"❌ Pose logging error: {e}")
    
    def get_trackbar_values(self):
        """Get current trackbar values"""
        if not self.show_gui:
            return {
                'sleep_threshold': 0.22,
                'knee_sit_threshold': 123,
                'move_std_threshold': 0.030,
                'history_length': 40
            }
        
        return {
            'sleep_threshold': cv2.getTrackbarPos("Sleep Thresh (x100)", self.window_name) / 100.0,
            'knee_sit_threshold': cv2.getTrackbarPos("Knee Sit Thresh", self.window_name),
            'move_std_threshold': cv2.getTrackbarPos("Move Std Thresh (x1000)", self.window_name) / 1000.0,
            'history_length': max(1, cv2.getTrackbarPos("History Length", self.window_name))
        }
    
    def analyze_pose(self, landmarks, params):
        """Analyze pose from landmarks and return pose status"""
        try:
            # Extract keypoints
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP].z]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].z]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE].x,
                   landmarks[mp_pose.PoseLandmark.NOSE].y,
                   landmarks[mp_pose.PoseLandmark.NOSE].z]
            
            # Calculate body span
            head_y = min(nose[1], left_shoulder[1], right_shoulder[1])
            feet_y = max(left_ankle[1], right_ankle[1])
            body_y_span = feet_y - head_y
            
            # Ankle movement history
            self.ankle_history_left.append(left_ankle[:2])
            self.ankle_history_right.append(right_ankle[:2])
            
            if len(self.ankle_history_left) > params['history_length']:
                self.ankle_history_left = self.ankle_history_left[-params['history_length']:]
                self.ankle_history_right = self.ankle_history_right[-params['history_length']:]
            
            # Calculate movement
            movement_std = 0.0
            is_moving = False
            if len(self.ankle_history_left) > 1:
                std_left_x = np.std([p[0] for p in self.ankle_history_left])
                std_left_y = np.std([p[1] for p in self.ankle_history_left])
                std_left = np.sqrt(std_left_x**2 + std_left_y**2)
                std_right_x = np.std([p[0] for p in self.ankle_history_right])
                std_right_y = np.std([p[1] for p in self.ankle_history_right])
                std_right = np.sqrt(std_right_x**2 + std_right_y**2)
                movement_std = (std_left + std_right) / 2
                is_moving = movement_std > params['move_std_threshold']
            
            # Determine pose status
            if body_y_span < params['sleep_threshold']:
                status = "Sleeping"
                confidence = 1.0 - abs(body_y_span - params['sleep_threshold']) / params['sleep_threshold']
            else:
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
                
                if avg_knee_angle < params['knee_sit_threshold']:
                    status = "Sitting"
                    confidence = 1.0 - avg_knee_angle / params['knee_sit_threshold']
                else:
                    if is_moving:
                        status = "Walking"
                        confidence = min(1.0, movement_std / params['move_std_threshold'])
                    else:
                        status = "Standing"
                        confidence = 1.0 - min(1.0, movement_std / params['move_std_threshold'])
            
            # Update current status
            self.current_status.update({
                'pose': status,
                'confidence': max(0.0, min(1.0, confidence)),
                'knee_angle': avg_knee_angle if 'avg_knee_angle' in locals() else 0.0,
                'body_span': body_y_span,
                'movement_std': movement_std
            })
            
            return status
            
        except Exception as e:
            print(f"Error analyzing pose: {e}")
            return "Processing error"
    
    def process_frame(self, frame_data):
        """Process a single frame for pose estimation"""
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_count = frame_data['frame_count']
        
        self.frame_count = frame_count
        self.current_status['frames_processed'] = frame_count
        
        # Get trackbar values
        params = self.get_trackbar_values()
        
        # Process with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        status = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            status = self.analyze_pose(landmarks, params)
        else:
            status = "No Detection"
            self.current_status['pose'] = status
        
        # Handle status logging
        if status:
            duration = time.time() - self.status_start_time
            if status != self.last_status or duration >= self.MAX_LOG_DURATION:
                if self.last_status and duration >= self.STABILITY_WINDOW:
                    self.log_status(self.last_status, min(duration, self.MAX_LOG_DURATION))
                    print(f"📝 Logged pose: {self.last_status} for {duration:.1f}s")
                self.status_start_time = time.time()
            self.last_status = status
            
            # Smooth display
            self.status_history.append(status)
            if len(self.status_history) > 10:
                self.status_history = self.status_history[-10:]
            display_status = max(set(self.status_history), key=self.status_history.count)
            self.current_status['pose'] = display_status
        
        # Return results for combined display
        result = {
            'pose': self.current_status['pose'],
            'confidence': self.current_status['confidence'],
            'pose_landmarks': results.pose_landmarks,
            'knee_angle': self.current_status['knee_angle'],
            'body_span': self.current_status['body_span'],
            'movement_std': self.current_status['movement_std']
        }
        
        # Display GUI if enabled (for standalone mode)
        if self.show_gui:
            self.render_gui(image, results, display_status if status else "No Detection", params)
        
        return result
    
    def render_gui(self, image, results, display_status, params):
        """Render the GUI display"""
        orig_h, orig_w = image.shape[:2]
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        # Status colors
        status_colors = {
            'Sleeping': (255, 0, 255),    # Magenta
            'Sitting': (0, 255, 255),     # Cyan
            'Standing': (0, 255, 0),      # Green
            'Walking': (255, 255, 0),     # Yellow
            'No Detection': (0, 0, 255),  # Red
            'Processing error': (128, 128, 128)  # Gray
        }
        
        status_color = status_colors.get(display_status, (255, 255, 255))
        
        # Main pose status
        cv2.putText(image, f"Pose: {display_status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
        
        # Confidence and metrics
        cv2.putText(image, f"Confidence: {self.current_status['confidence']:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(image, f"Knee Angle: {self.current_status['knee_angle']:.1f}°", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.putText(image, f"Body Span: {self.current_status['body_span']:.3f}", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.putText(image, f"Movement: {self.current_status['movement_std']:.4f}", (10, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Parameter display
        cv2.putText(image, f"Sleep Thresh: {params['sleep_threshold']:.2f}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        cv2.putText(image, f"Knee Thresh: {params['knee_sit_threshold']}", (10, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        cv2.putText(image, f"Move Thresh: {params['move_std_threshold']:.3f}", (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        # System info
        cv2.putText(image, f"Frame: {self.frame_count}", (10, orig_h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(image, "Pose Estimation System", (10, orig_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Legend
        legend_y = 220
        cv2.putText(image, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        legend_items = [
            ("Sleeping", (255, 0, 255)),
            ("Sitting", (0, 255, 255)),
            ("Standing", (0, 255, 0)),
            ("Walking", (255, 255, 0))
        ]
        
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + 20 + (i * 15)
            cv2.circle(image, (15, y_pos - 5), 4, color, -1)
            cv2.putText(image, label, (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        cv2.imshow(self.window_name, image)
    
    def get_status(self):
        """Get current system status"""
        return self.current_status.copy()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Log final status before cleanup
            if self.last_status:
                duration = time.time() - self.status_start_time
                if duration >= self.STABILITY_WINDOW:
                    self.log_status(self.last_status, duration)
            
            if hasattr(self, 'conn'):
                self.conn.close()
            if hasattr(self, 'temp_pose'):
                self.temp_pose.close()
            # Don't destroy windows here - let main orchestrator handle it
            print("Pose Estimation System cleaned up")
        except Exception as e:
            print(f"Cleanup error: {e}")

# For standalone testing
if __name__ == "__main__":
    print("🔧 Testing Pose Estimation System standalone...")
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera available for testing")
        exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    system = PoseEstimationSystem(show_gui=True)
    
    try:
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        print("🎬 Starting pose estimation test...")
        print("Controls: Press 'q' to quit, 'i' for info")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            
            frame_data = {
                'frame': cv2.flip(frame, 1),
                'timestamp': time.time(),
                'frame_count': frame_count
            }
            
            system.process_frame(frame_data)
            
            # Print periodic status
            if frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                status = system.get_status()
                print(f"📊 Status - FPS: {current_fps:.1f} | Frame: {frame_count} | "
                      f"Pose: {status['pose']} | Confidence: {status['confidence']:.2f}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                status = system.get_status()
                print("\n" + "="*40)
                print("📋 POSE ESTIMATION STATUS")
                print("="*40)
                for key, value in status.items():
                    print(f"{key}: {value}")
                print("="*40 + "\n")
                
    except KeyboardInterrupt:
        print("\n⚡ Interrupted by user")
    finally:
        cap.release()
        system.cleanup()
        cv2.destroyAllWindows()
        print("👋 Pose Estimation System test complete")