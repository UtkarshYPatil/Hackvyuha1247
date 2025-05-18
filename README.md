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
.
├── known_faces/ # Directory to store images of known individuals
│ └── [Person_Name_1]/
│ ├── image1.jpg
│ └── image2.png
│ └── [Person_Name_2]/
│ └── image1.jpeg
├── Face_cascade.xml # Haar cascade file for face detection
├── main.py # Main application script
├── requirements.txt # Python dependencies
├── face_recognition_log.csv # Log file for recognitions (auto-generated)
├── known_faces_encodings.pkl # Cache for known face encodings (auto-generated)
├── image_4d0293.png # Example image, replace as needed
└── README.md

## Requirements

* Python 3.x
* OpenCV
* A C++ compiler (required by `dlib`, a dependency of `face-recognition`)
* CMake (required to build `dlib`)

The specific Python package versions are listed in `requirements.txt`:
* opencv-python==4.8.1
* numpy==1.24.3
* face-recognition==1.3.0
* dlib==19.24.2

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install system dependencies (dlib):**
    * **Linux (Debian/Ubuntu):**
        ```bash
        sudo apt-get update
        sudo apt-get install build-essential cmake
        sudo apt-get install libopenblas-dev liblapack-dev # Optional, for dlib performance
        sudo apt-get install libx11-dev libgtk-3-dev # For GUI if not present
        ```
    * **macOS:**
        ```bash
        brew install cmake
        brew install openblas # Optional, for dlib performance
        ```
    * **Windows:**
        * Install Visual Studio with C++ build tools.
        * Install CMake and add it to your PATH.

4.  **Install Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing `dlib` can sometimes be tricky. If you encounter issues, refer to the official `dlib` installation guide or the `face-recognition` library's troubleshooting tips.*

5.  **Download Haar Cascade File:**
    Ensure `Face_cascade.xml` (e.g., `haarcascade_frontalface_default.xml`) is present in the root directory of the project. You can download it from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades). The script will check for this file on startup.

## Usage

1.  **Prepare Known Faces:**
    * Create a directory named `known_faces` in the project's root.
    * Inside `known_faces`, create subdirectories for each person you want to recognize. The name of the subdirectory will be used as the person's identified name.
    * Place one or more images (JPG, JPEG, PNG) of that person in their respective subdirectory. Ensure faces are clear and reasonably well-lit for best results.

2.  **Run the Application:**
    ```bash
    python main.py
    ```

3.  **Interface:**
    * A window titled "Multi-Camera Face Recognition" will appear, displaying a grid of camera feeds.
    * Detected faces will be highlighted with a red rectangle.
    * Recognized individuals will have their name displayed below the rectangle.
    * The camera ID will be displayed on each feed.
    * Press 'q' to quit the application.

4.  **Logs:**
    * Recognition events are logged in `face_recognition_log.csv`.

## Configuration

Several parameters can be adjusted directly in the `main.py` script:

* `KNOWN_FACES_DIR`: Directory containing images of known individuals.
* `HAAR_CASCADE_FILE`: Path to the Haar Cascade XML file for face detection.
* `RECOGNITION_TOLERANCE`: Threshold for face recognition (lower is stricter). Default is `0.6`.
* `LOG_FILE`: Name of the CSV file for logging recognitions.
* `MAX_CAMERAS`: Maximum number of cameras the system will attempt to use.
* `FRAME_QUEUE_SIZE`: Size of the frame queue for each camera.
* `TARGET_WIDTH`, `TARGET_HEIGHT`: Target resolution for camera frames.
* `PROCESS_EVERY_N_FRAMES`: Process every Nth frame to manage load.
* `MAX_WORKERS`: Maximum number of threads in the ThreadPoolExecutor for face processing.
* `FACE_DETECTION_SCALE_FACTOR`, `FACE_DETECTION_MIN_NEIGHBORS`, `MIN_FACE_SIZE`: Parameters for the Haar Cascade face detector.
* `TARGET_FPS`: Desired frames per second for camera capture.
* `ENCODINGS_CACHE_FILE`: File to store cached face encodings.

## How It Works

1.  **Initialization**:
    * The log file (`face_recognition_log.csv`) is initialized.
    * The Haar Cascade classifier is loaded.
    * Known faces are loaded:
        * It first tries to load pre-computed encodings from `known_faces_encodings.pkl`.
        * If the cache is not found or invalid, it processes images from subdirectories in `KNOWN_FACES_DIR`. Each subdirectory name is taken as a person's name. Face encodings are generated and then cached.

2.  **Camera Detection & Threading**:
    * The system attempts to detect and open up to `MAX_CAMERAS`.
    * For each successfully detected camera, a new thread is started (`process_camera_feed`).
    * This thread continuously captures frames, attempts to set desired camera properties (resolution, FPS), and puts frames into a dedicated queue.

3.  **Main Loop & Frame Processing**:
    * The main thread retrieves frames from each camera's queue.
    * For each frame:
        * It's converted to grayscale for face detection using the Haar Cascade classifier.
        * For each detected face, a task is submitted to a `ThreadPoolExecutor` (`process_face`).

4.  **Face Recognition (`process_face` function)**:
    * The detected face region is extracted and converted to RGB.
    * A face encoding is generated for the detected face.
    * This encoding is compared against all known face encodings.
    * If a match is found within the `RECOGNITION_TOLERANCE`, the corresponding person's name is identified. "Unknown" is used otherwise.
    * If a known person is recognized and hasn't been logged for that camera recently (e.g., within the last 5 seconds), the recognition event is logged.

5.  **Display**:
    * The `FrameBuffer` class manages a grid display.
    * Individual camera frames (with drawn rectangles and names from recognition results) are updated in their respective positions in the grid.
    * The combined grid is displayed in the OpenCV window. Camera IDs are overlaid on each feed.

6.  **Termination**:
    * Pressing 'q' sets a stop event, signaling all threads to terminate gracefully.
    * Camera resources are released, and OpenCV windows are destroyed.

## Troubleshooting

* **`dlib` installation issues:** This is the most common problem. Ensure you have a C++ compiler and CMake installed correctly. Consult `face-recognition` or `dlib` documentation for detailed, OS-specific instructions.
* **Camera not detected/opened:**
    * Ensure cameras are connected properly and drivers are installed.
    * Permissions: On Linux, you might need to be part of the `video` group.
    * Try different `camera_id` values if default (0, 1, ...) doesn't work.
    * The script tries `cv2.CAP_DSHOW` and `cv2.CAP_MSMF` on Windows, which often helps.
* **Poor recognition accuracy:**
    * Ensure good quality, well-lit images in `known_faces` directory.
    * Adjust `RECOGNITION_TOLERANCE` (lower for stricter, higher for more lenient).
    * Ensure `TARGET_WIDTH` and `TARGET_HEIGHT` provide sufficient resolution for faces.
* **"Haar Cascade file not found"**: Download `haarcascade_frontalface_default.xml` and place it as `Face_cascade.xml` or update `HAAR_CASCADE_FILE` in `main.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs, feature requests, or improvements.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if you choose to add one, otherwise you can remove this or state "No license provided").

## Acknowledgements

* [face_recognition library](https://github.com/ageitgey/face_recognition)
* [OpenCV](https://opencv.org/)
