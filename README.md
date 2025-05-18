# Multi-Camera Live Face Recognition System

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV Version](https://img.shields.io/badge/opencv-4.8.1-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a real-time, multi-camera face recognition system using Python, OpenCV, and the `face_recognition` library. It can detect and identify known individuals from multiple video streams simultaneously, logging each recognition event with a timestamp, date, person's name, and the camera ID. The system is designed to be configurable and resilient, with features like automatic reloading of known faces from a cache, dynamic camera detection, and robust error handling for camera feeds.

## Features

* **Multi-Camera Support**: Simultaneously process feeds from multiple connected cameras (configurable up to `MAX_CAMERAS`).
* **Real-time Face Detection**: Utilizes Haar Cascades for efficient face detection in video frames.
* **Face Recognition**: Identifies known individuals by comparing detected faces against a library of pre-encoded known faces.
* **Known Faces Management**:
    * Loads known faces from a structured directory (`known_faces`).
    * Caches face encodings (`known_faces_encodings.pkl`) for faster startup.
    * Automatically re-processes images if the cache is missing or invalid.
* **Event Logging**: Logs all successful recognition events to a CSV file (`face_recognition_log.csv`) with timestamp, date, name, and camera ID.
* **Dynamic Camera Grid Display**: Shows a grid of all active camera feeds, with recognized faces highlighted and labeled.
* **Configurable Parameters**: Many aspects of the system can be tuned via constants in `main.py` (e.g., resolution, FPS, tolerance).
* **Threaded Processing**:
    * Each camera feed is processed in a separate thread for improved performance.
    * Face processing tasks are submitted to a thread pool executor.
* **Resilient Camera Handling**:
    * Attempts to open cameras using multiple OpenCV backends (DSHOW, MSMF for Windows).
    * Attempts to re-open camera feeds if they fail during operation.
    * Configurable camera properties (resolution, FPS, auto-focus, auto-exposure).
* **Frame Buffering**: Manages frames from cameras using queues and a display buffer for smoother output.

## Project Structure
