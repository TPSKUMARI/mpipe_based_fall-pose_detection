# Fall and Pose Detection Project

This project is a real-time computer vision application that detects a person's pose and checks whether a fall has happened using a webcam.

## Overview

The system uses MediaPipe Pose Estimation to track body landmarks and analyze the person in front of the camera. It can identify basic states such as:

- Sleeping
- Sitting
- Standing
- Walking
- No Detection

It also monitors body movement and posture changes to detect possible fall events.

## Main Features

- Live video processing from webcam
- Human pose estimation using MediaPipe
- Fall detection based on body posture and motion
- GUI-based interface built with PySide6
- Logging of pose and fall results
- Real-time status monitoring

## Project Structure

- `main.py` – main application and GUI
- `fall_detection.py` – fall detection logic and rules
- `pose_estimation.py` – pose classification and logging
- `requirements.txt` – Python dependencies

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- PySide6
- SQLite (for logs)

## Setup

1. Open a terminal in the project folder.
2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python main.py
```

## Notes

- A webcam is required to run the application.
- The system is designed for demonstration and research purposes.
- Detection accuracy depends on lighting, camera angle, and body visibility.

## Purpose

This project is useful for understanding how computer vision can be used for health and safety monitoring, especially in environments where fall detection is important.
