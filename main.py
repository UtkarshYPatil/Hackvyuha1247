import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime
import threading
from queue import Queue, Empty as QueueEmpty # Explicit import for clarity
from concurrent.futures import ThreadPoolExecutor
import time
import pickle # Added for caching
import platform # To check OS for backend selection

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
HAAR_CASCADE_FILE = "Face_cascade.xml" # Ensure this file exists or provide the full path
RECOGNITION_TOLERANCE = 0.6
LOG_FILE = "face_recognition_log.csv"
MAX_CAMERAS = 2
FRAME_QUEUE_SIZE = 7
TARGET_WIDTH = 480 # Increased resolution for better face recognition accuracy
TARGET_HEIGHT = 360 # Increased resolution for better face recognition accuracy
PROCESS_EVERY_N_FRAMES = 10 # Keep high to manage load at higher resolution
MAX_WORKERS = 30
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (30, 30)
TARGET_FPS = 20 # Further reduced target FPS for smoother multi-camera feed on less powerful systems
ENCODINGS_CACHE_FILE = "known_faces_encodings.pkl"

# Global variables for thread synchronization
# frame_locks = {} # Not used in the provided snippet, consider if needed for specific shared resources
# last_processed_times = {} # Not used
last_recognitions = {}

# OpenCV Video Capture Backend
CV_CAP_ANY = cv2.CAP_ANY # Default
CV_CAP_DSHOW = cv2.CAP_DSHOW # DirectShow (Windows)
CV_CAP_MSMF = cv2.CAP_MSMF # Media Foundation (Windows)


class FrameBuffer:
    def __init__(self, width, height, num_cameras):
        self.width = width
        self.height = height
        self.num_cameras = num_cameras
        if num_cameras <= 0: # Handle case with no active cameras
            self.grid_size = 1
            self.buffer = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            self.grid_size = int(np.ceil(np.sqrt(num_cameras)))
            self.buffer = np.zeros((height * self.grid_size, width * self.grid_size, 3), dtype=np.uint8)

        self.frame_buffers = {}
        for i in range(num_cameras):
            self.frame_buffers[i] = np.zeros((height, width, 3), dtype=np.uint8)

    def clear(self):
        self.buffer.fill(0)
        for buf in self.frame_buffers.values():
            buf.fill(0)

    def update_frame(self, camera_id, frame):
        if camera_id in self.frame_buffers:
            if frame.shape[:2] != (self.height, self.width):
                try:
                    frame = cv2.resize(frame, (self.width, self.height))
                except cv2.error as e:
                    print(f"Error resizing frame for camera {camera_id}: {e}. Frame shape: {frame.shape}")
                    return # Skip update if resize fails
            self.frame_buffers[camera_id] = frame.copy()

    def draw_to_grid(self):
        if not self.frame_buffers: # If no cameras, do nothing or show a placeholder
             # Create a placeholder message if the main buffer is for a single "no camera" view
            if self.num_cameras == 0 and self.buffer.shape[:2] == (self.height, self.width):
                 cv2.putText(self.buffer, "No Cameras Active", (50, self.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return

        for idx, (camera_id, frame) in enumerate(self.frame_buffers.items()):
            # This assumes camera_id maps sequentially for grid placement.
            # If camera_ids can be sparse (e.g., 0 and 2, but not 1),
            # this grid logic might need adjustment or ensure active_cameras corresponds to sequential IDs.
            row = idx // self.grid_size
            col = idx % self.grid_size
            
            # Ensure frame is exactly the right size before copying
            # This check might be redundant if update_frame already resized it, but good for safety
            if frame.shape[:2] != (self.height, self.width):
                try:
                    frame = cv2.resize(frame, (self.width, self.height))
                except cv2.error as e: # Should not happen if update_frame worked
                    print(f"Error resizing frame for grid (camera {camera_id}): {e}")
                    # Fill with black if resize fails
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Boundary checks for drawing onto the main buffer
            y_start, y_end = row * self.height, (row + 1) * self.height
            x_start, x_end = col * self.width, (col + 1) * self.width

            if y_end <= self.buffer.shape[0] and x_end <= self.buffer.shape[1]:
                 self.buffer[y_start:y_end, x_start:x_end] = frame
            else:
                print(f"Warning: Frame for camera {camera_id} at grid ({row},{col}) is out of main buffer bounds.")


    def get_grid(self):
        return self.buffer.copy()


def initialize_log_file():
    """
    Creates a new CSV log file with headers if it doesn't exist
    """
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Date', 'Person Name', 'Camera ID'])

def log_recognition(person_name, camera_id):
    """
    Logs the recognition event to CSV file including camera ID
    """
    current_time = datetime.now()
    timestamp = current_time.strftime("%H:%M:%S")
    date = current_time.strftime("%Y-%m-%d")

    with open(LOG_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, date, person_name, camera_id])

def load_known_faces_from_folders():
    known_face_encodings = []
    known_face_names = []

    if os.path.exists(ENCODINGS_CACHE_FILE):
        print(f"Loading known faces from cache ('{ENCODINGS_CACHE_FILE}')...")
        try:
            with open(ENCODINGS_CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
                known_face_encodings = cached_data['encodings']
                known_face_names = cached_data['names']
                if known_face_names:
                    print(f"Successfully loaded encodings for {len(set(known_face_names))} unique person(s) from cache.")
                    return known_face_encodings, known_face_names
                else:
                    print("Cache was empty or invalid. Re-processing from folders.")
        except Exception as e:
            print(f"Error loading from cache: {e}. Re-processing from folders.")

    print(f"Loading known faces from '{KNOWN_FACES_DIR}' using folder structure...")
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Error: Directory '{KNOWN_FACES_DIR}' not found. Please create it.")
        return known_face_encodings, known_face_names

    persons_processed_count = 0
    images_with_faces_count = 0

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_folder_path = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_folder_path):
            print(f"  Processing folder for: {person_name}")
            images_processed_for_person = 0
            for filename in os.listdir(person_folder_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(person_folder_path, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_encodings_in_image = face_recognition.face_encodings(image)
                        if face_encodings_in_image:
                            known_face_encodings.append(face_encodings_in_image[0])
                            known_face_names.append(person_name)
                            images_processed_for_person += 1
                            images_with_faces_count += 1
                        else:
                            print(f"    Warning: No face found in {image_path}")
                    except Exception as e:
                        print(f"    Error loading or processing {image_path}: {str(e)}")
            if images_processed_for_person > 0:
                print(f"    Successfully loaded {images_processed_for_person} image(s) for {person_name}.")
                persons_processed_count +=1
            else:
                print(f"    No images with detectable faces found for {person_name}.")

    if not known_face_names:
        print(f"No known faces were loaded. Face recognition will not be effective.")
        print(f"Please ensure subfolders in '{KNOWN_FACES_DIR}' contain images with detectable faces.")
    else:
        print(f"Successfully processed and loaded encodings for {persons_processed_count} unique person(s) from {images_with_faces_count} images.")
        try:
            with open(ENCODINGS_CACHE_FILE, 'wb') as f:
                pickle.dump({'encodings': known_face_encodings, 'names': known_face_names}, f)
            print(f"Saved known face encodings to cache ('{ENCODINGS_CACHE_FILE}').")
        except Exception as e:
            print(f"Error saving encodings to cache: {e}")

    return known_face_encodings, known_face_names

def process_face(frame, face_location, known_face_encodings, known_face_names, camera_id):
    try:
        x, y, w, h = face_location
        # Ensure coordinates are within frame boundaries before slicing
        y_start, y_end = max(0, y), min(frame.shape[0], y + h)
        x_start, x_end = max(0, x), min(frame.shape[1], x + w)

        # Margin can be added more carefully if desired, but ensure it doesn't go out of bounds
        # For simplicity, using the direct face location first
        face_img_bgr = frame[y_start:y_end, x_start:x_end]

        if face_img_bgr.size == 0:
            return "Unknown", None

        # Resize for consistency, but ensure it's not upscaling tiny detections too much
        # if face_img_bgr.shape[0] < 30 or face_img_bgr.shape[1] < 30: # If too small, recognition might be poor
        #     return "Too small", None
        try:
            face_img_bgr = cv2.resize(face_img_bgr, (160, 160)) # Standard size for some models
        except cv2.error: # Handle potential resize error if face_img_bgr is degenerate
            return "Unknown (resize err)", None

        face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        current_face_encodings = face_recognition.face_encodings(face_img_rgb, num_jitters=1) # num_jitters=1 for speed

        if not current_face_encodings or not known_face_encodings:
            return "Unknown", None

        face_encoding = current_face_encodings[0]
        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding,
            tolerance=RECOGNITION_TOLERANCE
        )
        name = "Unknown"
        log_time = None

        if any(matches):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < RECOGNITION_TOLERANCE:
                name = known_face_names[best_match_index]
                current_time = datetime.now()
                last_recognition_info = last_recognitions.get(camera_id, {}).get(name)

                if (last_recognition_info is None or
                        (current_time - last_recognition_info).total_seconds() >= 5): # Log every 5s per person per camera
                    log_recognition(name, camera_id) # Pass camera_id here
                    if camera_id not in last_recognitions:
                        last_recognitions[camera_id] = {}
                    last_recognitions[camera_id][name] = current_time
                    log_time = current_time # Return the time it was logged
        return name, log_time
    except Exception as e:
        print(f"Error in process_face (cam {camera_id}): {str(e)}")
        return "Error", None

def process_camera_feed(camera_id, frame_queue, stop_event):
    cap = None
    preferred_backends = []
    if platform.system() == "Windows":
        preferred_backends = [CV_CAP_DSHOW, CV_CAP_MSMF, CV_CAP_ANY]
    else: # Linux, macOS, etc.
        preferred_backends = [CV_CAP_ANY] # Or add V4L2 if specific: cv2.CAP_V4L2

    for backend_idx, backend in enumerate(preferred_backends):
        print(f"Camera {camera_id}: Trying to open with backend {backend}...")
        cap = cv2.VideoCapture(camera_id, backend)
        if cap.isOpened():
            print(f"Camera {camera_id}: Successfully opened with backend {backend}.")
            break
        else:
            print(f"Camera {camera_id}: Failed to open with backend {backend}.")
            if cap: cap.release() # Ensure it's released if open failed partially

    if not cap or not cap.isOpened():
        print(f"Error: Could not open camera {camera_id} with any backend. Stopping feed for this camera.")
        return

    # --- Attempt to set camera properties ---
    print(f"Camera {camera_id}: Attempting to set properties...")
    # Resolution (important to set before other properties like FPS on some cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    # FPS
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # Buffer size
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) # Keep a small buffer on the camera side

    # Attempt to disable auto-focus
    # (0 means off, 1 means on for CAP_PROP_AUTOFOCUS)
    if cap.set(cv2.CAP_PROP_AUTOFOCUS, 1):
        print(f"Camera {camera_id}: Auto-focus disabled.")
    else:
        print(f"Camera {camera_id}: Could not disable auto-focus (or not supported).")

    # Attempt to disable auto-exposure
    # (Common values: 0 for manual, 1 for auto, 0.25 for some manual modes on certain APIs)
    # This is highly camera/driver dependent.
    # For DirectShow, 0 might not be manual; you might need to set CAP_PROP_AUTO_EXPOSURE to 1 (aperture priority)
    # or 0.25 (manual) then set CAP_PROP_EXPOSURE. Let's try 0 first.
    if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1): # Try to set to full manual
         print(f"Camera {camera_id}: Auto-exposure set to manual mode (attempted).")
         # If manual exposure is set, you might need to set a specific exposure value.
         # This value is camera-dependent. Too low = dark, too high = bright.
         # E.g., cap.set(cv2.CAP_PROP_EXPOSURE, -6) # Values typically range e.g. -13 to 1
         # print(f"Camera {camera_id}: Current exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    else:
        # Try another common way for some APIs (like UVC cameras on Linux via V4L2)
        # where 1 = manual, 3 = auto/aperture priority
        if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3): # Try another value often used for manual
             print(f"Camera {camera_id}: Auto-exposure set to manual mode (value 1, attempted).")
        else:
             print(f"Camera {camera_id}: Could not disable auto-exposure (or not supported).")


    # --- Log actual camera properties ---
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_autofocus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    actual_autoexposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)

    print(f"Camera {camera_id} - Actual settings: "
          f"Resolution: {actual_width}x{actual_height}, FPS: {actual_fps:.2f}, "
          f"Autofocus: {actual_autofocus}, AutoExposure: {actual_autoexposure}, Exposure: {actual_exposure}")

    if actual_fps == 0:
        print(f"Warning: Camera {camera_id} reports 0 FPS. Frame capture might be unstable or use default rate.")
        # min_frame_interval will be problematic if actual_fps is 0
        min_frame_interval = 1.0 / 15 # Fallback to a default reasonable FPS like 15
    else:
        min_frame_interval = 1.0 / actual_fps if actual_fps > 0 else 1.0 / TARGET_FPS


    frame_count = 0
    last_frame_time = time.perf_counter() # Use perf_counter for more precise timing

    while not stop_event.is_set():
        current_time = time.perf_counter()
        elapsed = current_time - last_frame_time

        if elapsed < min_frame_interval:
            sleep_duration = min_frame_interval - elapsed
            # Avoid sleeping for too short or negative durations if timing is off
            if sleep_duration > 0.0001:
                time.sleep(sleep_duration)
            continue # Re-check time for precise frame grabbing interval

        last_frame_time = time.perf_counter() # Reset timer just before read
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Failed to capture frame from camera {camera_id}. Attempting to re-open...")
            cap.release()
            time.sleep(0.5)
            # Re-attempt opening with the same backend logic
            reopened = False
            for backend in preferred_backends:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    reopened = True
                    print(f"Camera {camera_id}: Successfully re-opened with backend {backend}.")
                    # Re-apply settings if needed, though some might persist, others might reset
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS) # Re-attempt
                    # ... other settings if necessary
                    break
            if not reopened:
                print(f"Error: Failed to reopen camera {camera_id} after capture failure. Stopping feed.")
                stop_event.set() # Signal other parts of this thread to stop if critical
                break
            continue

        frame_count += 1
        if PROCESS_EVERY_N_FRAMES == 1 or frame_count % PROCESS_EVERY_N_FRAMES == 0:
            if frame.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH) and frame.size > 0 :
                try:
                    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                except cv2.error as e:
                    print(f"Camera {camera_id}: Error resizing frame in loop: {e}. Skipping frame.")
                    continue
            
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()  # Discard oldest frame
                except QueueEmpty:
                    pass # Should not happen if full() is true, but good practice
            
            try:
                frame_queue.put_nowait((frame.copy(), camera_id)) # Put a copy to avoid modification issues
            except Queue.Full: # Should be handled by the previous get_nowait, but as a safeguard
                 pass


    print(f"Camera {camera_id}: Releasing capture...")
    if cap:
        cap.release()
    print(f"Camera {camera_id}: Feed stopped.")


def main():
    initialize_log_file()

    if not os.path.exists(HAAR_CASCADE_FILE):
        print(f"Error: Haar Cascade file '{HAAR_CASCADE_FILE}' not found. "
              "Please download it (e.g., haarcascade_frontalface_default.xml) "
              "and place it in the correct path or update HAAR_CASCADE_FILE.")
        return

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)
    if face_cascade.empty():
        print(f"Error: Could not load Haar Cascade classifier from '{HAAR_CASCADE_FILE}'.")
        return

    known_face_encodings, known_face_names = load_known_faces_from_folders()
    if not known_face_encodings:
        print("Warning: No known face encodings loaded. Recognition will default to 'Unknown'.")

    camera_threads = []
    frame_queues = []
    stop_event = threading.Event()
    active_camera_indices = [] # Store indices of successfully opened cameras

    print("\nAttempting to connect to cameras...")
    for camera_id_test in range(MAX_CAMERAS):
        # Quick check if camera can be opened. Actual opening with settings is in the thread.
        temp_cap = cv2.VideoCapture(camera_id_test, cv2.CAP_ANY) # Use default for quick check
        if temp_cap.isOpened():
            active_camera_indices.append(camera_id_test)
            temp_cap.release()
            print(f"  Camera {camera_id_test} detected.")
        else:
            print(f"  Camera {camera_id_test} could not be detected or opened for initial check.")
            if temp_cap: temp_cap.release()

    if not active_camera_indices:
        print("Error: No cameras could be detected. Please check connections and permissions.")
        return

    print(f"\nInitializing feeds for {len(active_camera_indices)} detected camera(s): {active_camera_indices}")
    for camera_id in active_camera_indices:
        frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
        frame_queues.append(frame_queue)
        thread = threading.Thread(
            target=process_camera_feed,
            args=(camera_id, frame_queue, stop_event)
        )
        thread.daemon = True
        camera_threads.append(thread)
        thread.start()

    # Wait a moment for camera threads to initialize and report status
    time.sleep(3) # Allow time for initial connection attempts and printouts

    # Check if any threads are alive (i.e., cameras actually working)
    # This is a bit indirect; ideally, process_camera_feed would signal success/failure back
    if not any(t.is_alive() for t in camera_threads):
        print("Error: None of the detected cameras could be started successfully. Exiting.")
        stop_event.set()
        for thread in camera_threads:
            thread.join(timeout=2.0)
        return

    print(f"\nStarting live recognition with active cameras. Press 'q' to quit.")
    print(f"Recognition logs will be saved to '{LOG_FILE}'")

    # Initialize frame buffer with the number of successfully started cameras
    # This relies on camera_threads corresponding to active_camera_indices
    num_active_display_cameras = len(active_camera_indices) # Assume all started threads are for display
    frame_buffer = FrameBuffer(TARGET_WIDTH, TARGET_HEIGHT, num_active_display_cameras)


    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        try:
            while True:
                if stop_event.is_set() and all(not t.is_alive() for t in camera_threads):
                    print("All camera feeds stopped. Exiting main loop.")
                    break

                frames_from_queues = []
                for i, queue in enumerate(frame_queues):
                    try:
                        # Use camera_id from active_camera_indices to map queue to camera_id
                        original_camera_id = active_camera_indices[i]
                        frame_data, _ = queue.get_nowait() # Get (frame, cam_id_from_thread)
                        frames_from_queues.append((frame_data, original_camera_id))
                    except QueueEmpty:
                        pass # Queue is empty, no frame from this camera yet
                    except IndexError:
                        print(f"Warning: Mismatch between frame_queues and active_camera_indices at index {i}")


                if not frames_from_queues and not any(t.is_alive() for t in camera_threads):
                    # If no frames and all camera threads are dead, break
                    if not stop_event.is_set(): # If not already stopping, indicate an issue
                         print("No frames and all camera threads are stopped unexpectedly.")
                    break
                elif not frames_from_queues:
                    time.sleep(0.01) # Wait briefly if no frames yet but threads might still be running
                    continue

                # frame_buffer.clear() # Cleared implicitly by overwriting with new frames

                future_processing_info = []

                # Update individual buffers first
                for frame, camera_id in frames_from_queues:
                    frame_buffer.update_frame(camera_id, frame) # camera_id here is the original index

                    # Process face detection on this frame for this camera_id
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces_haar = face_cascade.detectMultiScale(
                        gray_frame,
                        scaleFactor=FACE_DETECTION_SCALE_FACTOR,
                        minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
                        minSize=MIN_FACE_SIZE,
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    for face_location in faces_haar:
                        future = executor.submit(
                            process_face,
                            frame.copy(), # Send a copy for thread safety
                            face_location,
                            known_face_encodings,
                            known_face_names,
                            camera_id # Original camera_id for logging
                        )
                        future_processing_info.append({
                            'future': future,
                            'location': face_location,
                            'camera_id': camera_id # Original camera_id for drawing
                        })

                # Process face recognition results and draw on the specific camera's buffer
                for info in future_processing_info:
                    try:
                        name, _ = info['future'].result(timeout=0.1) # Slightly longer timeout
                        x, y, w, h = info['location']
                        # Get the specific frame from the buffer for drawing
                        target_frame_for_drawing = frame_buffer.frame_buffers.get(info['camera_id'])

                        if target_frame_for_drawing is not None:
                            cv2.rectangle(target_frame_for_drawing, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.rectangle(target_frame_for_drawing, (x, y + h - 25), (x + w, y + h), (0, 0, 255), cv2.FILLED)
                            cv2.putText(target_frame_for_drawing, name, (x + 6, y + h - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        else:
                            print(f"Warning: Could not find frame buffer for camera_id {info['camera_id']} to draw on.")
                    except TimeoutError:
                        # print(f"Face processing timed out for a face on camera {info['camera_id']}")
                        pass # Silently ignore if processing takes too long for this frame cycle
                    except Exception as e:
                        print(f"Error getting result from future for camera {info['camera_id']}: {e}")
                        continue

                # Add camera numbers to frames in their individual buffers
                for cam_idx_in_grid, original_cam_id in enumerate(active_camera_indices):
                    # cam_idx_in_grid will be 0, 1, ... corresponding to the position in the grid
                    # original_cam_id is the actual ID like 0, 1, 2...
                    if original_cam_id in frame_buffer.frame_buffers:
                        cv2.putText(frame_buffer.frame_buffers[original_cam_id], f"Cam {original_cam_id}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                frame_buffer.draw_to_grid()
                display_grid = frame_buffer.get_grid()
                if display_grid.size == 0 :
                    print("Grid buffer is empty. Cannot display.")
                    time.sleep(0.1)
                    continue

                cv2.imshow('Multi-Camera Face Recognition', display_grid)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting signal received...")
                    stop_event.set()
                    break
        
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught. Exiting...")
            stop_event.set()

        finally:
            print("Main loop ended. Cleaning up...")
            if not stop_event.is_set():
                stop_event.set() # Ensure stop event is set for all threads

            print("Waiting for camera threads to join...")
            for i, thread in enumerate(camera_threads):
                if thread.is_alive():
                    print(f"  Joining camera thread {active_camera_indices[i]}...")
                    thread.join(timeout=3.0) # Increased timeout
                    if thread.is_alive():
                        print(f"  Warning: Camera thread {active_camera_indices[i]} did not terminate gracefully.")
                else:
                     print(f"  Camera thread {active_camera_indices[i]} was already stopped.")

            print("Shutting down thread pool executor...")
            executor.shutdown(wait=True) # Wait for all pending recognition tasks

            cv2.destroyAllWindows()
            print("Resources released. Program terminated.")

if __name__ == "__main__":
    # Add a check for the cascade file at the very beginning
    if not os.path.exists(HAAR_CASCADE_FILE):
        print(f"CRITICAL ERROR: Haar Cascade file '{HAAR_CASCADE_FILE}' not found.")
        print("Please download 'haarcascade_frontalface_default.xml' (or similar) and ensure the path is correct.")
        print("You can find it in the OpenCV data repository: https://github.com/opencv/opencv/tree/master/data/haarcascades")
        exit(1) # Exit if critical file is missing

    main()