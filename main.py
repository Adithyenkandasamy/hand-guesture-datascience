import numpy as np
import os
import cv2
import time
from collections import deque
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyautogui

# Configuration focused on highest prediction value
IMG_SIZE = 100
NUM_FRAMES = 6
NUM_CLASSES = 5
PREDICTION_THRESHOLD = 0.5  # Lower threshold to be more responsive
CONFIDENCE_HISTORY_SIZE = 3  # Track fewer predictions for quicker response
MIN_CONSISTENT_PREDICTIONS = 2  # Require fewer consistent predictions
SKIP_FRAMES = 1  # Process every frame for better detection
PREDICTION_INTERVAL = 0.1  # Quicker predictions

GESTURE_CLASSES = {
    0: "Thumbs Up",    # Volume +
    1: "Thumbs Down",  # Volume -
    2: "Left Swipe",   # Rewind
    3: "Right Swipe",  # Forward
    4: "Stop"          # Pause/Play
}

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_frames(frames):
    return frames / 255.0

def detect_and_crop_hand(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    annotated = frame.copy()
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        # Calculate bounding box with larger padding for better coverage
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)

        # Add generous padding for better coverage
        padding = 40
        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

        # Ensure minimum size
        if x_max - x_min < 100:
            center_x = (x_min + x_max) // 2
            x_min = max(0, center_x - 50)
            x_max = min(w, center_x + 50)
        if y_max - y_min < 100:
            center_y = (y_min + y_max) // 2
            y_min = max(0, center_y - 50)
            y_max = min(h, center_y + 50)

        cropped = frame[y_min:y_max, x_min:x_max]
        return cropped, annotated, True

    return frame, annotated, False

def preprocess_frame(frame, size=(IMG_SIZE, IMG_SIZE)):
    try:
        # Try to improve image quality for better recognition
        # Apply slight Gaussian blur to reduce noise
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # Enhance contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        # Resize to expected size
        resized = cv2.resize(enhanced, size)
        return resized.astype(np.float32)
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

def prepare_input_batch(frames, model_input_shape):
    frames_array = np.array(frames, dtype=np.float32)
    frames_array = normalize_frames(frames_array)
    
    # Print frames shape to debug input requirements
    print(f"Frames shape before reshaping: {frames_array.shape}")
    
    if len(model_input_shape) == 5:  # (None, frames, height, width, channels)
        # For 3D CNN models (TimeDistributed)
        batch = np.expand_dims(frames_array, axis=0)
        print(f"Input batch shape for 3D model: {batch.shape}")
        return batch
    elif len(model_input_shape) == 4:  # (None, height, width, channels*frames)
        # For 2D CNN models with stacked frames
        height, width, channels = frames_array[0].shape
        flattened = np.reshape(frames_array, (1, height, width, channels * len(frames)))
        print(f"Input batch shape for 2D model: {flattened.shape}")
        return flattened
    else:
        # Generic fallback
        batch = np.expand_dims(frames_array, axis=0)
        print(f"Input batch shape (generic): {batch.shape}")
        return batch

def capture_and_predict(model, camera_source=0):
    input_shape = model.input_shape
    print(f"Model input shape: {input_shape}")
    
    # Print output shape to understand the expected predictions
    output_shape = model.output_shape
    print(f"Model output shape: {output_shape}")
    
    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    frame_buffer = deque(maxlen=NUM_FRAMES)
    
    # Performance tracking
    prev_time = time.time()
    frame_count = 0
    last_prediction_time = 0
    
    # Accuracy tracking focused on highest predictions
    prediction_history = deque(maxlen=CONFIDENCE_HISTORY_SIZE)
    current_gesture = None
    current_confidence = 0
    action_performed = None
    consistent_predictions = 0
    last_prediction_class = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        model_complexity=1  # Use medium complexity for better accuracy
    ) as hands:

        print("Press 'q' to quit.")
        print("Waiting for hand gestures...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame capture failed.")
                break
                
            frame = cv2.flip(frame, 1)  # Mirror effect
            
            process_this_frame = frame_count % SKIP_FRAMES == 0
            
            if process_this_frame:
                cropped, annotated, hand_found = detect_and_crop_hand(frame, hands)
            else:
                annotated = frame.copy()
                hand_found = len(frame_buffer) > 0
                
            frame_count += 1
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time

            if hand_found and process_this_frame:
                processed = preprocess_frame(cropped)
                frame_buffer.append(processed)

                # Only predict when we have enough frames and enough time has passed
                if (len(frame_buffer) == NUM_FRAMES and 
                    curr_time - last_prediction_time > PREDICTION_INTERVAL):
                    
                    last_prediction_time = curr_time
                    
                    try:
                        # Get prediction
                        input_data = prepare_input_batch(list(frame_buffer), input_shape)
                        prediction = model.predict(input_data, verbose=0)[0]
                        
                        # Print raw predictions for debugging
                        print("Raw predictions:", prediction)
                        
                        pred_class = np.argmax(prediction)
                        confidence = prediction[pred_class]
                        
                        # Store prediction result (class, confidence)
                        prediction_history.append((pred_class, confidence))
                        
                        # Count occurrences of each class in recent predictions
                        class_counts = {}
                        class_confidences = {}
                        
                        for cls, conf in prediction_history:
                            if cls not in class_counts:
                                class_counts[cls] = 0
                                class_confidences[cls] = []
                            class_counts[cls] += 1
                            class_confidences[cls].append(conf)
                        
                        # Find the class with the highest average confidence
                        highest_avg_confidence = 0
                        best_class = None
                        
                        for cls, confs in class_confidences.items():
                            avg_conf = sum(confs) / len(confs)
                            if avg_conf > highest_avg_confidence:
                                highest_avg_confidence = avg_conf
                                best_class = cls
                        
                        # Only accept predictions with sufficient confidence
                        if best_class is not None and highest_avg_confidence >= PREDICTION_THRESHOLD:
                            # Check if this is the same class as before
                            if best_class == last_prediction_class:
                                consistent_predictions += 1
                            else:
                                consistent_predictions = 1
                                
                            last_prediction_class = best_class
                            
                            # Execute action if we have consistent predictions
                            if consistent_predictions >= MIN_CONSISTENT_PREDICTIONS:
                                current_gesture = GESTURE_CLASSES[best_class]
                                current_confidence = highest_avg_confidence
                                action_performed = perform_youtube_action(best_class)
                                
                                # Print detailed prediction information
                                print(f"\nGesture detected: {current_gesture}")
                                print(f"Confidence: {current_confidence:.4f}")
                                print(f"Class counts in history: {class_counts}")
                                print(f"Action performed: {action_performed}")
                                print(f"Consistent predictions: {consistent_predictions}\n")
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        import traceback
                        traceback.print_exc()
            elif process_this_frame and not hand_found:
                # Clear buffer when hand is lost
                frame_buffer.clear()
                consistent_predictions = 0  # Reset consistency counter
            
            # Create information display
            info_overlay = np.zeros((200, 320, 3), dtype=np.uint8)
            
            # FPS
            cv2.putText(info_overlay, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Hand detection status
            cv2.putText(info_overlay, f"Hand: {'YES' if hand_found else 'NO'}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if hand_found else (0, 0, 255), 1)
            
            # Frame buffer status
            cv2.putText(info_overlay, f"Frames: {len(frame_buffer)}/{NUM_FRAMES}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Current gesture and confidence
            if current_gesture:
                cv2.putText(info_overlay, f"Gesture: {current_gesture}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(info_overlay, f"Confidence: {current_confidence:.2f}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                cv2.putText(info_overlay, "Awaiting gesture...", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            
            # Action performed
            if action_performed:
                action_text = f"Action: {action_performed}"
                text_size = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = annotated.shape[1] // 2 - text_size[0] // 2
                cv2.putText(annotated, action_text, 
                           (text_x, annotated.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
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
            cv2.imshow("YouTube Gesture Control", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def main():
    default_path = '/home/jinwoo/Desktop/hand-guesture-datascience/gesture_model_20250404-205437/final_gesture_model.h5'
    
    if not os.path.exists(default_path):
        print(f"Default model path not found: {default_path}")
        user_path = input("Please enter the path to your gesture model .h5 file: ")
        model_path = user_path if user_path else default_path
    else:
        model_path = default_path
    
    if not os.path.exists(model_path):
        print("Model file not found:", model_path)
        return

    try:
        print("Loading model...")
        model = load_model(model_path)
        
        # Get model input shape
        input_shape = model.input_shape
        print(f"Model input shape: {input_shape}")
        
        # Update global parameters based on model inspection
        global NUM_FRAMES, IMG_SIZE
        if len(input_shape) == 5:  # 3D+time model
            NUM_FRAMES = input_shape[1]
            IMG_SIZE = input_shape[2]  # Assuming height = width
            print(f"3D+time model detected: Using {NUM_FRAMES} frames at {IMG_SIZE}x{IMG_SIZE}")
        else:
            print(f"Using default: {NUM_FRAMES} frames at {IMG_SIZE}x{IMG_SIZE}")
        
        # Display model summary
        model.summary()
        
        # Display instructions
        print("\n=== YOUTUBE GESTURE CONTROL FOR UBUNTU ===")
        print("This version is optimized for YouTube in Chrome on Ubuntu.")
        print("Supported gestures:")
        for cls, gesture in GESTURE_CLASSES.items():
            action = perform_youtube_action(cls)
            print(f"  {gesture} - {action if action else 'N/A'}")
        print("==========================================")
        print("Make sure Chrome with YouTube is the active window.")
        print("Press 'q' to quit.")
        
        # Start capturing and predicting
        capture_and_predict(model)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()