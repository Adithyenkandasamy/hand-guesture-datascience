import numpy as np
import os
import cv2
import time
from collections import deque
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyautogui
import json

# Configuration parameters
IMG_SIZE = 100
NUM_FRAMES = 18  # Default from training code
NUM_LANDMARKS = 21  # Number of hand landmarks
NUM_CHANNELS = 3
NUM_CLASSES = 5
PREDICTION_THRESHOLD = 0.80  # Increased confidence threshold
PREDICTION_INTERVAL = 0.5  # Increased time between predictions
CONFIDENCE_HISTORY_SIZE = 10  # Increased history size for better averaging
MIN_CONSISTENT_PREDICTIONS = 5  # Increased required consistent predictions

# Gesture classes and corresponding actions
GESTURE_CLASSES = {
    0: "Thumbs Up",    # Volume +
    1: "Thumbs Down",  # Volume -
    2: "Left Swipe",   # Rewind
    3: "Right Swipe",  # Forward
    4: "Stop"          # Pause/Play
}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def load_model_metadata(model_folder):
    """Load model configuration parameters"""
    metadata_path = os.path.join(model_folder, "/home/jinwoo/Desktop/hand-guesture-datascience/models/model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        global IMG_SIZE, NUM_FRAMES, NUM_CLASSES, NUM_LANDMARKS
        IMG_SIZE = metadata.get("img_size", IMG_SIZE)
        NUM_FRAMES = metadata.get("num_frames", NUM_FRAMES)
        NUM_CLASSES = metadata.get("num_classes", NUM_CLASSES)
        NUM_LANDMARKS = metadata.get("num_landmarks", NUM_LANDMARKS)
        
        # Update gesture classes if provided
        if "gesture_classes" in metadata:
            gesture_classes = metadata["gesture_classes"]
            # Convert keys from string to int (JSON converts keys to strings)
            for k, v in gesture_classes.items():
                GESTURE_CLASSES[int(k)] = v
        
        print(f"Using model configuration: {NUM_FRAMES} frames at {IMG_SIZE}x{IMG_SIZE}, {NUM_CLASSES} classes")
        return metadata.get("use_landmarks", True)
    
    return True  # Default to using landmarks if metadata not found

def normalize_frames(frames):
    """Normalize pixel values to range [0, 1]"""
    return frames / 255.0

def normalize_landmarks(landmarks):
    """Flatten the landmarks coordinates"""
    return landmarks.flatten()

def extract_hand_landmarks(image, hands_detector):
    """Extract hand landmarks from an image using MediaPipe"""
    # Convert to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = hands_detector.process(rgb_image)
    
    # Initialize landmarks array with zeros
    landmarks = np.zeros((NUM_LANDMARKS, 3))  # x, y, z for each landmark
    
    # If hand landmarks detected, extract coordinates
    if results.multi_hand_landmarks:
        # Use only the first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract landmark coordinates (x, y, z)
        for i, landmark in enumerate(hand_landmarks.landmark):
            if i < NUM_LANDMARKS:
                landmarks[i] = [landmark.x, landmark.y, landmark.z]
    
    return landmarks, results.multi_hand_landmarks is not None

def get_hand_bounding_box(landmarks, image_shape):
    """Get hand bounding box from landmarks"""
    if np.all(landmarks == 0):
        return None
    
    height, width = image_shape[:2]
    
    # Convert normalized coordinates to pixel values
    x_coords = landmarks[:, 0] * width
    y_coords = landmarks[:, 1] * height
    
    # Get non-zero coordinates
    valid_x = x_coords[x_coords > 0]
    valid_y = y_coords[y_coords > 0]
    
    if len(valid_x) == 0 or len(valid_y) == 0:
        return None
    
    # Calculate bounding box
    x_min, y_min = int(np.min(valid_x)), int(np.min(valid_y))
    x_max, y_max = int(np.max(valid_x)), int(np.max(valid_y))
    
    # Add padding
    padding = 30
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    return (x_min, y_min, x_max, y_max)

def preprocess_frame(frame, target_size=(IMG_SIZE, IMG_SIZE)):
    """Preprocess a single frame: resize and convert color"""
    try:
        if frame is None:
            return np.zeros((target_size[0], target_size[1], NUM_CHANNELS), dtype=np.float32)
        
        # Resize to target size
        resized = cv2.resize(frame, target_size)
        
        # Convert to RGB (model was trained on RGB)
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            # If grayscale, convert to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        return rgb.astype(np.float32)
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return np.zeros((target_size[0], target_size[1], NUM_CHANNELS), dtype=np.float32)

def perform_action(pred_class):
    """Perform action based on predicted gesture class"""
    actions = {
        0: ('up', "Volume UP"),      # Thumbs Up
        1: ('down', "Volume DOWN"),  # Thumbs Down
        2: ('left', "Rewind 10s"),   # Left Swipe
        3: ('right', "Forward 10s"), # Right Swipe
        4: ('space', "Pause/Play")   # Stop
    }
    key, action = actions.get(pred_class, (None, None))
    if key:
        pyautogui.press(key)
        return action
    return None

def run_gesture_recognition(model, use_landmarks=True, camera_id=0):
    """Main function to run gesture recognition with webcam input"""
    # Initialize video capture
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initialize buffers
    frame_buffer = deque(maxlen=NUM_FRAMES)
    landmarks_buffer = deque(maxlen=NUM_FRAMES)
    prediction_history = deque(maxlen=CONFIDENCE_HISTORY_SIZE)
    
    # Tracking variables
    last_prediction_time = 0
    current_gesture = None
    current_confidence = 0
    action_performed = None
    frame_count = 0
    consistent_predictions = 0
    last_prediction_class = None
    
    print(f"Starting gesture recognition{'with landmarks' if use_landmarks else 'image-only'} mode...")
    print(f"Press 'q' to quit")
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands_detector:
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Mirror image for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Extract hand landmarks
            landmarks, hand_detected = extract_hand_landmarks(frame, hands_detector)
            
            # Draw landmarks on frame
            annotated_frame = frame.copy()
            if hand_detected:
                # Get hand bounding box
                bbox = get_hand_bounding_box(landmarks, frame.shape)
                
                # Draw landmarks
                for i in range(NUM_LANDMARKS):
                    if landmarks[i].any():  # If landmark is detected
                        x, y = int(landmarks[i, 0] * frame.shape[1]), int(landmarks[i, 1] * frame.shape[0])
                        cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                
                # Draw bounding box
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Crop hand region for processing
                    hand_crop = frame[y_min:y_max, x_min:x_max]
                    
                    # Preprocess frame
                    processed_frame = preprocess_frame(hand_crop)
                    
                    # Add to buffers
                    frame_buffer.append(processed_frame)
                    landmarks_buffer.append(landmarks)
                    
                    # Make prediction if buffer is full and enough time has passed
                    current_time = time.time()
                    if (len(frame_buffer) == NUM_FRAMES and 
                        len(landmarks_buffer) == NUM_FRAMES and
                        current_time - last_prediction_time > PREDICTION_INTERVAL):
                        
                        last_prediction_time = current_time
                        
                        try:
                            # Prepare input
                            if use_landmarks:
                                # Hybrid model (images + landmarks)
                                img_input = np.zeros((1, NUM_FRAMES, IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
                                landmarks_input = np.zeros((1, NUM_FRAMES, NUM_LANDMARKS * 3))
                                
                                # Normalize inputs
                                normalized_frames = normalize_frames(np.array(frame_buffer))
                                img_input[0] = normalized_frames
                                
                                for j, lm in enumerate(landmarks_buffer):
                                    landmarks_input[0, j] = normalize_landmarks(lm)
                                
                                # Make prediction
                                prediction = model.predict([img_input, landmarks_input], verbose=0)[0]
                            else:
                                # Image-only model
                                img_input = np.zeros((1, NUM_FRAMES, IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
                                normalized_frames = normalize_frames(np.array(frame_buffer))
                                img_input[0] = normalized_frames
                                
                                # Make prediction
                                prediction = model.predict(img_input, verbose=0)[0]
                            
                            # Get predicted class and confidence
                            pred_class = np.argmax(prediction)
                            confidence = prediction[pred_class]
                            
                            # Only log prediction if it meets the threshold
                            if confidence >= PREDICTION_THRESHOLD:
                                print(f"Prediction: {GESTURE_CLASSES.get(pred_class, 'Unknown')} ({confidence:.4f})")
                            
                            # Add to prediction history if it meets the threshold
                            if confidence >= PREDICTION_THRESHOLD:
                                prediction_history.append((pred_class, confidence))
                            
                            # Analyze prediction history
                            if len(prediction_history) >= MIN_CONSISTENT_PREDICTIONS:
                                class_counts = {}
                                class_confidences = {}
                                
                                for cls, conf in prediction_history:
                                    if cls not in class_counts:
                                        class_counts[cls] = 0
                                        class_confidences[cls] = []
                                    class_counts[cls] += 1
                                    class_confidences[cls].append(conf)
                                
                                # Find highest confidence class
                                highest_avg_confidence = 0
                                best_class = None
                                
                                for cls, confs in class_confidences.items():
                                    avg_conf = sum(confs) / len(confs)
                                    if avg_conf > highest_avg_confidence:
                                        highest_avg_confidence = avg_conf
                                        best_class = cls
                                
                                # Check if prediction meets threshold
                                if best_class is not None and highest_avg_confidence >= PREDICTION_THRESHOLD:
                                    # Check if same as previous prediction
                                    if best_class == last_prediction_class:
                                        consistent_predictions += 1
                                    else:
                                        consistent_predictions = 1
                                    
                                    last_prediction_class = best_class
                                    
                                    # Execute action if consistent predictions
                                    if consistent_predictions >= MIN_CONSISTENT_PREDICTIONS:
                                        current_gesture = GESTURE_CLASSES.get(best_class, "Unknown")
                                        current_confidence = highest_avg_confidence
                                        action_performed = perform_action(best_class)
                                        
                                        # Reset consistency counter after performing action
                                        consistent_predictions = 0
                                        
                                        # Clear prediction history after successful action
                                        prediction_history.clear()
                        except Exception as e:
                            print(f"Error during prediction: {e}")
                            import traceback
                            traceback.print_exc()
            else:
                # Clear buffers when hand is lost
                frame_buffer.clear()
                landmarks_buffer.clear()
                consistent_predictions = 0
            
            # Display information on frame
            cv2.putText(annotated_frame, f"FPS: {1.0/(time.time() - frame_count) if frame_count > 0 else 0:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(annotated_frame, f"Hand detected: {'Yes' if hand_detected else 'No'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(annotated_frame, f"Buffer: {len(frame_buffer)}/{NUM_FRAMES}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display current gesture and action
            if current_gesture:
                cv2.putText(annotated_frame, f"Gesture: {current_gesture}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(annotated_frame, f"Confidence: {current_confidence:.2f}", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if action_performed:
                cv2.putText(annotated_frame, f"Action: {action_performed}", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Hand Gesture Recognition", annotated_frame)
            
            # Update frame count for FPS calculation
            frame_count = time.time()
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Default model path - update with your actual path
    default_model_path = "/home/jinwoo/Desktop/hand-guesture-datascience/models"
    
    # Ask for model path if default not found
    if not os.path.exists(default_model_path):
        model_path = input("Enter the path to your model directory: ")
    else:
        model_path = default_model_path
    
    # Look for model file in directory
    model_files = [f for f in os.listdir(model_path) if f.endswith('.h5')]
    
    if not model_files:
        print(f"No model files found in {model_path}")
        return
    
    # Select model file
    print("Available models:")
    for i, f in enumerate(model_files):
        print(f"{i+1}. {f}")
    
    if len(model_files) > 1:
        selection = input(f"Select model (1-{len(model_files)}): ")
        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(model_files):
                idx = 0
        except:
            idx = 0
    else:
        idx = 0
    
    model_file = model_files[idx]
    model_file_path = os.path.join(model_path, model_file)
    
    # Try to load model metadata
    use_landmarks = load_model_metadata(model_path)
    
    try:
        # Load model
        print(f"Loading model: {model_file_path}")
        model = load_model(model_file_path)
        model.summary()
        
        # Run gesture recognition
        run_gesture_recognition(model, use_landmarks=use_landmarks)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()