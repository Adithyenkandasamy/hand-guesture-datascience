import numpy as np
import os
import cv2
import time
from collections import deque
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyautogui

# Configuration - these will be updated after inspecting the model
IMG_SIZE = 100  # Back to original size
NUM_FRAMES = 18  # Back to original size
NUM_CLASSES = 5
PREDICTION_THRESHOLD = 0.65
SKIP_FRAMES = 3  # Process every 3rd frame for detection
PREDICTION_INTERVAL = 0.8  # Seconds between predictions

GESTURE_CLASSES = {
    0: "Thumbs Up",    # Volume +
    1: "Thumbs Down",  # Volume -
    2: "Left Swipe",   # Rewind
    3: "Right Swipe",  # Forward
    4: "Stop"          # Pause/Play
}

# MediaPipe setup with reduced complexity
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_frames(frames):
    return frames / 255.0

def detect_and_crop_hand(frame, hands):
    # Convert to RGB only once
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    annotated = frame.copy()
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        # Simple drawing without styles to save processing
        mp_drawing.draw_landmarks(
            annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        # Calculate bounding box
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)

        # Add padding
        padding = 20
        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

        # Enforce minimum crop size
        if x_max - x_min < 50:
            center_x = (x_min + x_max) // 2
            x_min = max(0, center_x - 25)
            x_max = min(w, center_x + 25)
        if y_max - y_min < 50:
            center_y = (y_min + y_max) // 2
            y_min = max(0, center_y - 25)
            y_max = min(h, center_y + 25)

        cropped = frame[y_min:y_max, x_min:x_max]
        return cropped, annotated, True

    return frame, annotated, False

def preprocess_frame(frame, size=(IMG_SIZE, IMG_SIZE)):
    try:
        # Direct OpenCV resize is faster than PIL
        frame = cv2.resize(frame, size)
        # Convert to float32 for model input
        return frame.astype(np.float32)
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return np.zeros((size[0], size[1], 3), dtype=np.float32)

def perform_youtube_action(pred_class):
    actions = {
        0: ('up', "Volume UP"),
        1: ('down', "Volume DOWN"),
        2: ('left', "Rewind 10s"),
        3: ('right', "Forward 10s"),
        4: ('space', "Pause/Play")
    }
    key, action = actions.get(pred_class, (None, None))
    if key:
        pyautogui.press(key)
        return action
    return None

def inspect_model(model_path):
    """Inspect the model to understand its input shape requirements"""
    model = load_model(model_path)
    print("\nModel Summary:")
    model.summary()
    
    # Get input shape
    input_shape = model.input_shape
    print(f"\nInput shape: {input_shape}")
    
    # If it's a sequential model, check first layer
    if hasattr(model, 'layers') and len(model.layers) > 0:
        first_layer = model.layers[0]
        print(f"First layer: {first_layer.name}, Input shape: {first_layer.input_shape}")
    
    # Return the expected frame count and image size
    if input_shape and len(input_shape) > 2:
        # Handle different possible input shapes
        if len(input_shape) == 5:  # (None, frames, height, width, channels)
            return input_shape[1], input_shape[2], input_shape[3]
        elif len(input_shape) == 4:  # (None, height, width, channels*frames) or similar
            return None, input_shape[1], input_shape[2]
    
    # Default if shape can't be determined
    return NUM_FRAMES, IMG_SIZE, IMG_SIZE

def prepare_input_batch(frames, model_input_shape):
    """Prepare the input based on model's expected shape"""
    # Convert frames to numpy array
    frames_array = np.array(frames, dtype=np.float32)
    
    # Normalize
    frames_array = normalize_frames(frames_array)
    
    # Check the expected input shape
    if len(model_input_shape) == 5:  # (None, frames, height, width, channels)
        # Model expects 3D+time input
        return np.array([frames_array])
    elif len(model_input_shape) == 4:  # (None, height, width, channels*frames) or similar
        # Model might expect flattened frames
        batch_size, frame_count, height, width, channels = frames_array.shape
        # Reshape to flatten the frames dimension
        flattened = frames_array.reshape(batch_size, height, width, channels * frame_count)
        return flattened
    
    # Default - just return the normalized frames
    return np.array([frames_array])

def capture_and_predict(model, camera_source=0):
    # Get model input shape
    input_shape = model.input_shape
    print(f"Model input shape: {input_shape}")
    
    # Try to optimize camera capture
    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower resolution
    
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    # Use deque for efficient frame buffer management
    frame_buffer = deque(maxlen=NUM_FRAMES)
    
    # Performance tracking
    prev_time = time.time()
    frame_count = 0
    last_prediction_time = 0
    
    # Accuracy tracking
    recent_predictions = deque(maxlen=10)
    accuracy_history = deque(maxlen=20)  # Keep only recent accuracy values
    current_gesture = None
    action_performed = None

    # Reduce hand detection complexity
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0  # Use simplest model
    ) as hands:

        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame capture failed.")
                break
                
            # Flip the frame horizontally to create mirror effect
            frame = cv2.flip(frame, 1)

            # Process only every nth frame for hand detection
            process_this_frame = frame_count % SKIP_FRAMES == 0
            
            if process_this_frame:
                cropped, annotated, hand_found = detect_and_crop_hand(frame, hands)
            else:
                annotated = frame.copy()
                hand_found = len(frame_buffer) > 0  # Continue using previous detection
                
            frame_count += 1
            
            # Display status
            curr_time = time.time()
            fps = 1 / max(curr_time - prev_time, 0.001)  # Avoid division by zero
            prev_time = curr_time

            if hand_found and process_this_frame:
                processed = preprocess_frame(cropped, size=(IMG_SIZE, IMG_SIZE))
                frame_buffer.append(processed)

                # Only predict at certain intervals
                if (len(frame_buffer) == NUM_FRAMES and 
                    curr_time - last_prediction_time > PREDICTION_INTERVAL):
                    
                    last_prediction_time = curr_time
                    
                    try:
                        # Prepare input based on model's requirements
                        input_data = prepare_input_batch(frame_buffer, input_shape)
                        
                        # Run prediction
                        prediction = model.predict(input_data, verbose=0)[0]
                        pred_class = np.argmax(prediction)
                        confidence = prediction[pred_class]
                        
                        # Track prediction if confident
                        if confidence > PREDICTION_THRESHOLD:
                            gesture = GESTURE_CLASSES[pred_class]
                            current_gesture = gesture
                            action_performed = perform_youtube_action(pred_class)
                            
                            # Add to recent predictions
                            recent_predictions.append((pred_class, confidence))
                            
                            # Calculate accuracy
                            if len(recent_predictions) >= 3:
                                # How many predictions match the current one
                                matching = sum(1 for p, _ in recent_predictions if p == pred_class)
                                accuracy = (matching / len(recent_predictions)) * 100
                                accuracy_history.append(accuracy)
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        # If there's an input shape mismatch, print debug info
                        print(f"Buffer shape: {np.array(frame_buffer).shape}")
                        print(f"Expected input shape: {input_shape}")
            elif process_this_frame:
                # Clear buffer when hand is lost
                frame_buffer.clear()
                current_gesture = None
            
            # Create information display
            info_overlay = np.zeros((150, 250, 3), dtype=np.uint8)
            
            # FPS
            cv2.putText(info_overlay, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Gesture
            if current_gesture:
                cv2.putText(info_overlay, f"Gesture: {current_gesture}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                cv2.putText(info_overlay, "No gesture", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            # Action
            if action_performed:
                cv2.putText(info_overlay, f"Action: {action_performed}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Accuracy
            if accuracy_history:
                avg_accuracy = sum(accuracy_history) / len(accuracy_history)
                cv2.putText(info_overlay, f"Accuracy: {avg_accuracy:.1f}%", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
            
            # Display buffer status
            cv2.putText(info_overlay, f"Frames: {len(frame_buffer)}/{NUM_FRAMES}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add overlay to main frame
            x_offset, y_offset = 10, 10
            overlay_h, overlay_w = info_overlay.shape[:2]
            
            # Make sure overlay fits within the frame
            if y_offset + overlay_h <= annotated.shape[0] and x_offset + overlay_w <= annotated.shape[1]:
                # Create a semi-transparent background
                annotated[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = \
                    cv2.addWeighted(
                        annotated[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w],
                        0.5,  # Alpha for original content
                        info_overlay,
                        0.5,  # Alpha for overlay
                        0
                    )
            
            # Display the frame
            cv2.imshow("Gesture Control", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def main():
    model_path = '/home/jinwoo/Desktop/hand-guesture-datascience/gesture_model_20250404-205437/final_gesture_model.h5'
    if not os.path.exists(model_path):
        print("Model file not found:", model_path)
        return

    try:
        # First, inspect the model to understand its requirements
        print("Inspecting model...")
        model = load_model(model_path)
        
        # Update global parameters based on model inspection
        global NUM_FRAMES, IMG_SIZE
        input_shape = model.input_shape
        
        # If input shape is a 5D tensor (batch, frames, height, width, channels)
        if len(input_shape) == 5:
            NUM_FRAMES = input_shape[1]
            IMG_SIZE = input_shape[2]  # Assuming height = width
        
        print(f"Using NUM_FRAMES={NUM_FRAMES}, IMG_SIZE={IMG_SIZE}")
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        
        # Start capturing and predicting
        capture_and_predict(model)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()