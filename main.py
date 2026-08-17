import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame

# Initialize MediaPipe once for shared use
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Import the existing detection systems
try:
    from fall_detection import FallDetectionSystem, filter_keypoints_by_confidence, SYSTEM_CONFIG, CRITICAL_KEYPOINTS
    from pose_estimation import PoseEstimationSystem
except ImportError as e:
    print(f"Error importing detection modules: {e}")
    print("Make sure fall_detection.py and pose_estimation.py are in the same directory")
    sys.exit(1)

POSE_COLORS = {
    'Sleeping': '#c084fc',
    'Sitting': '#22d3ee',
    'Standing': '#4ade80',
    'Walking': '#facc15',
    'No Detection': '#f87171',
}
FALL_COLOR = '#f87171'
NO_FALL_COLOR = '#4ade80'
NEUTRAL_COLOR = '#e5e7eb'
DIM_COLOR = '#9ca3af'


class DetectionEngine:
    """Runs shared MediaPipe pose inference plus the fall/pose detection algorithms."""

    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        print("Initializing detection systems...")
        self.fall_system = FallDetectionSystem(show_gui=False)
        self.pose_system = PoseEstimationSystem(show_gui=False)
        print("Detection systems initialized successfully!")

        self.frame_count = 0

    def process_frame(self, frame):
        """Process frame once with MediaPipe and run both detection algorithms"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True

        frame_height, frame_width = frame.shape[:2]

        fall_result = {
            'is_fall': False, 'confidence': 0.0, 'triggered_rules': [],
            'velocity': 0, 'bbox': None, 'fall_info': None
        }
        pose_result = {
            'pose': 'No Detection', 'confidence': 0.0, 'knee_angle': 0.0,
            'body_span': 0.0, 'movement_std': 0.0
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            try:
                bbox = None
                valid_landmarks = [lm for lm in landmarks if lm.visibility >= SYSTEM_CONFIG['confidence_threshold']]
                if valid_landmarks:
                    x_coords = [lm.x * frame_width for lm in valid_landmarks]
                    y_coords = [lm.y * frame_height for lm in valid_landmarks]
                    x1, x2 = int(min(x_coords)), int(max(x_coords))
                    y1, y2 = int(min(y_coords)), int(max(y_coords))
                    bbox = (x1, y1, x2, y2)

                filtered_keypoints = filter_keypoints_by_confidence(landmarks, frame_width, frame_height, 'mediapipe')

                if filtered_keypoints and len(filtered_keypoints) >= 4:
                    fall_info = self.fall_system.fall_detector.detect_fall(
                        filtered_keypoints, bbox, frame_width, frame_height
                    )

                    fall_result = {
                        'is_fall': fall_info['is_fall'],
                        'confidence': fall_info['confidence'],
                        'triggered_rules': fall_info['triggered_rules'],
                        'velocity': fall_info['velocity'],
                        'bbox': bbox,
                        'fall_info': fall_info
                    }

                    self.fall_system.current_status.update({
                        'fall_detected': fall_info['is_fall'],
                        'confidence': fall_info['confidence'],
                        'triggered_rules': fall_info['triggered_rules'],
                        'velocity': fall_info['velocity'],
                        'frames_processed': self.frame_count
                    })

                    current_time = time.time()
                    if fall_info['is_fall'] and (current_time - self.fall_system.last_fall_log_time) > 5.0:
                        self.fall_system.log_fall_event(fall_info)
                        self.fall_system.last_fall_log_time = current_time
                        print(f"FALL DETECTED! Frame {self.frame_count} - Confidence: {fall_info['confidence']:.3f} - Rules: {fall_info['triggered_rules']}")

            except Exception as e:
                print(f"Error in fall detection: {e}")

            try:
                params = {
                    'sleep_threshold': 0.22,
                    'knee_sit_threshold': 123,
                    'move_std_threshold': 0.030,
                    'history_length': 40
                }

                pose_status = self.pose_system.analyze_pose(landmarks, params)

                if pose_status:
                    pose_result = {
                        'pose': pose_status,
                        'confidence': self.pose_system.current_status.get('confidence', 0.0),
                        'knee_angle': self.pose_system.current_status.get('knee_angle', 0.0),
                        'body_span': self.pose_system.current_status.get('body_span', 0.0),
                        'movement_std': self.pose_system.current_status.get('movement_std', 0.0)
                    }

                    current_time = time.time()
                    duration = current_time - self.pose_system.status_start_time

                    if pose_status != self.pose_system.last_status or duration >= self.pose_system.MAX_LOG_DURATION:
                        if self.pose_system.last_status and duration >= self.pose_system.STABILITY_WINDOW:
                            self.pose_system.log_status(self.pose_system.last_status, min(duration, self.pose_system.MAX_LOG_DURATION))
                            print(f"Logged pose: {self.pose_system.last_status} for {duration:.1f}s")
                        self.pose_system.status_start_time = current_time

                    self.pose_system.last_status = pose_status

                    self.pose_system.status_history.append(pose_status)
                    if len(self.pose_system.status_history) > 10:
                        self.pose_system.status_history = self.pose_system.status_history[-10:]
                    display_status = max(set(self.pose_system.status_history), key=self.pose_system.status_history.count)
                    pose_result['pose'] = display_status

            except Exception as e:
                print(f"Error in pose estimation: {e}")

        return results, fall_result, pose_result

    def render_video_frame(self, frame, results, fall_result):
        """Draw only the pose skeleton and bounding box on the video frame (no text)."""
        display_frame = frame.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        bbox = fall_result.get('bbox')
        if bbox:
            fall_detected = fall_result.get('is_fall', False)
            color = (0, 0, 255) if fall_detected else (0, 255, 0)
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        return display_frame

    def cleanup(self):
        print("Cleaning up...")
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'fall_system'):
            self.fall_system.cleanup()
        if hasattr(self, 'pose_system'):
            self.pose_system.cleanup()
        print("Cleanup complete")


class StatChip(QFrame):
    """A small labeled value chip used in the top stats bar."""

    def __init__(self, title):
        super().__init__()
        self.setObjectName("statChip")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(2)

        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("chipTitle")

        self.value_label = QLabel("--")
        self.value_label.setObjectName("chipValue")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, text, color=NEUTRAL_COLOR):
        self.value_label.setText(text)
        self.value_label.setStyleSheet(f"color: {color};")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Combined Detection System")
        self.resize(960, 720)

        self.engine = DetectionEngine()
        self.cap = None

        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        self._build_ui()
        self._apply_styles()

        if not self._initialize_camera():
            sys.exit(1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(1)

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.status_bar_widget = QWidget()
        self.status_bar_widget.setObjectName("statusBar")
        bar_layout = QHBoxLayout(self.status_bar_widget)
        bar_layout.setContentsMargins(16, 10, 16, 10)
        bar_layout.setSpacing(10)

        self.chip_fall = StatChip("Fall Status")
        self.chip_fall_conf = StatChip("Fall Confidence")
        self.chip_rules = StatChip("Active Rules")
        self.chip_pose = StatChip("Pose")
        self.chip_pose_conf = StatChip("Pose Confidence")
        self.chip_fps = StatChip("FPS")

        for chip in (self.chip_fall, self.chip_fall_conf, self.chip_rules,
                     self.chip_pose, self.chip_pose_conf, self.chip_fps):
            bar_layout.addWidget(chip)

        bar_layout.addStretch()

        self.video_label = QLabel()
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)

        root.addWidget(self.status_bar_widget)
        root.addWidget(self.video_label, stretch=1)

        self.setCentralWidget(central)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0f14; }
            #statusBar { background-color: #12181f; border-bottom: 1px solid #232b34; }
            #videoLabel { background-color: #000000; }
            #statChip { background-color: #1a2129; border-radius: 8px; }
            #chipTitle { color: #6b7684; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; }
            #chipValue { color: #e5e7eb; font-size: 16px; font-weight: 700; }
        """)

    def _initialize_camera(self):
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Camera 0 not available, trying camera 1...")
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                print("No cameras found!")
                return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("Camera initialized successfully!")
        return True

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame")
            return

        frame = cv2.flip(frame, 1)
        self.engine.frame_count += 1

        results, fall_result, pose_result = self.engine.process_frame(frame)
        display_frame = self.engine.render_video_frame(frame, results, fall_result)

        self._calculate_fps()
        self._update_stats_bar(fall_result, pose_result)
        self._show_frame(display_frame)

    def _calculate_fps(self):
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time

    def _update_stats_bar(self, fall_result, pose_result):
        fall_detected = fall_result.get('is_fall', False)
        self.chip_fall.set_value(
            "FALL DETECTED" if fall_detected else "No Fall",
            FALL_COLOR if fall_detected else NO_FALL_COLOR
        )
        self.chip_fall_conf.set_value(f"{fall_result.get('confidence', 0.0):.2f}", NEUTRAL_COLOR)

        rules = fall_result.get('triggered_rules', [])
        self.chip_rules.set_value(", ".join(rules[:3]) if rules else "None", DIM_COLOR if not rules else "#facc15")

        pose_status = pose_result.get('pose', 'No Detection')
        self.chip_pose.set_value(pose_status, POSE_COLORS.get(pose_status, NEUTRAL_COLOR))
        self.chip_pose_conf.set_value(f"{pose_result.get('confidence', 0.0):.2f}", NEUTRAL_COLOR)

        self.chip_fps.set_value(f"{self.current_fps:.1f}", NEUTRAL_COLOR)

        # Flash the status bar background when a fall is detected
        bg = "#3a1414" if fall_detected else "#12181f"
        self.status_bar_widget.setStyleSheet(f"#statusBar {{ background-color: {bg}; border-bottom: 1px solid #232b34; }}")

    def _show_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.engine.cleanup()
        event.accept()


def main():
    print("Starting Combined Detection System...")
    print("Fall Detection + Pose Estimation")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    try:
        sys.exit(app.exec())
    finally:
        print("System shutdown complete")


if __name__ == "__main__":
    main()
