import numpy as np
import os
import cv2
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
import pyautogui  # For controlling YouTube (keyboard actions)

# Configuration parameters
IMG_SIZE = 100  # Size to resize frames to (height and width)
NUM_FRAMES = 18  # Number of frames to use from each video
NUM_CLASSES = 5  # Number of gesture classes

# Gesture labels and corresponding YouTube actions
GESTURE_CLASSES = {
    0: "Thumbs Up",    # Increase volume (press +)
    1: "Thumbs Down",  # Decrease volume (press -)
    2: "Left Swipe",   # Rewind 10 seconds (press Left Arrow)
    3: "Right Swipe",  # Forward 10 seconds (press Right Arrow)
    4: "Stop"          # Pause/Play (press Spacebar)
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Helper functions
def normalize_frames(frames):
    """Normalize pixel values to range [0, 1]"""
    return frames / 255.0

def detect_and_crop_hand(frame, hands):
    """
    Detect hand in frame and crop to hand area with padding
    Returns the cropped hand image or the original frame if no hand is detected
    """
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Create a copy for drawing
    annotated_frame = frame.copy()
    
    h, w, c = frame.shape
    
    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        # Get the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw hand landmarks on the frame
        mp_drawing.draw_landmarks(
            annotated_frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Extract bounding box coordinates
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        # Add padding to the bounding box
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Ensure minimum size of the bounding box
        if x_max - x_min < 50:
            center = (x_min + x_max) // 2
            x_min = max(0, center - 25)
            x_max = min(w, center + 25)
        
        if y_max - y_min < 50:
            center = (y_min + y_max) // 2
            y_min = max(0, center - 25)
            y_max = min(h, center + 25)
        
        # Crop the hand region
        hand_crop = frame[y_min:y_max, x_min:x_max]
        
        # Return both cropped hand and annotated frame
        return hand_crop, annotated_frame, True
    
    return frame, annotated_frame, False

def preprocess_frame(frame, target_size=(IMG_SIZE, IMG_SIZE)):
    """Preprocess a single frame: resize to target size"""
    # Convert to PIL Image if not already
    if not isinstance(frame, Image.Image):
        try:
            frame = Image.fromarray(frame)
        except:
            # Handle case where frame might be None or invalid
            return np.zeros((target_size[0], target_size[1], 3))
    
    # Resize to target size
    frame = frame.resize(target_size)
    
    # Convert to numpy array
    frame = np.asarray(frame).astype(np.float32)
    
    return frame

# Camera capture and real-time prediction
def capture_and_predict(model, camera_source=0, prediction_threshold=0.65):
    """Capture video from webcam, detect hands, and make real-time predictions"""
    cap = cv2.VideoCapture(camera_source)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Buffer to store processed frames
    frame_buffer = []
    
    print("Starting real-time gesture recognition. Press 'q' to quit.")
    print("Make sure the YouTube video is playing and focused in your browser.")
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Detect and crop hand
            hand_crop, annotated_frame, hand_detected = detect_and_crop_hand(frame, hands)
            
            # Preprocess the frame
            if hand_detected:
                processed_frame = preprocess_frame(hand_crop, (IMG_SIZE, IMG_SIZE))
                
                # Add to buffer and keep only the required number of frames
                frame_buffer.append(processed_frame)
                if len(frame_buffer) > NUM_FRAMES:
                    frame_buffer.pop(0)
                
                # Make prediction when buffer is full
                if len(frame_buffer) == NUM_FRAMES:
                    # Prepare input for model
                    input_data = np.array([normalize_frames(np.array(frame_buffer))])
                    
                    # Make prediction
                    prediction = model.predict(input_data, verbose=0)[0]
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[predicted_class]
                    
                    # Display prediction if confidence is above threshold
                    if confidence > prediction_threshold:
                        gesture = GESTURE_CLASSES[predicted_class]
                        cv2.putText(annotated_frame, f"{gesture} ({confidence:.2f})", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Perform YouTube action based on predicted gesture
                        if predicted_class == 0:  # Thumbs Up
                            pyautogui.press('+')  # Increase volume
                            print("Action: Volume UP (+)")
                        elif predicted_class == 1:  # Thumbs Down
                            pyautogui.press('-')  # Decrease volume
                            print("Action: Volume DOWN (-)")
                        elif predicted_class == 2:  # Left Swipe
                            pyautogui.press('left')  # Rewind 10 seconds
                            print("Action: Rewind 10s (Left Arrow)")
                        elif predicted_class == 3:  # Right Swipe
                            pyautogui.press('right')  # Forward 10 seconds
                            print("Action: Forward 10s (Right Arrow)")
                        elif predicted_class == 4:  # Stop
                            pyautogui.press('space')  # Pause/Play
                            print("Action: Pause/Play (Spacebar)")
            else:
                annotated_frame = frame
                # Clear buffer if no hand detected for some frames
                if len(frame_buffer) > 0:
                    frame_buffer.pop(0)
                
                # Display message when no hand is detected
                cv2.putText(annotated_frame, "No hand detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the frame with annotations
            cv2.imshow('Hand Gesture Recognition', annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Load the trained model
    model_path = 'your_model.h5'  # Replace with the actual path to your saved model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}.  Make sure you have trained the model and that the path is correct.")
        return

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error: Failed to load model from {model_path}.  Error details: {e}")
        return
    
    # Run the webcam prediction
    capture_and_predict(model)

if __name__ == "__main__":
    main()
