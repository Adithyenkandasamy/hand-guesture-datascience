import numpy as np
import os
import cv2
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, MaxPooling2D, Conv3D, MaxPooling3D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers, regularizers

# Set random seeds for reproducibility
np.random.seed(30)
import random as rn
rn.seed(30)
tf.random.set_seed(30)

# Configuration parameters
IMG_SIZE = 100  # Size to resize frames to (height and width)
NUM_FRAMES = 18  # Number of frames to use from each video
NUM_CLASSES = 5  # Number of gesture classes
BATCH_SIZE = 32  # Batch size for training

# Gesture labels
GESTURE_CLASSES = {
    0: "Thumbs Up",    # Increase volume
    1: "Thumbs Down",  # Decrease volume
    2: "Left Swipe",   # Rewind 10 seconds
    3: "Right Swipe",  # Forward 10 seconds
    4: "Stop"          # Pause content
}

# Frame indices to use (choose specific frames from the 30-frame videos)
FRAME_INDICES = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, 28, 29]

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

def create_data_generator(source_path, file_list, batch_size, img_size, frame_indices, normalize_func):
    """Generator function to create batches of data for training/validation with hand detection"""
    num_frames = len(frame_indices)
    
    # Initialize MediaPipe Hands for preprocessing
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3) as hands:
        
        while True:
            # Shuffle the file list
            t = np.random.permutation(file_list)
            num_batches = int(np.ceil(len(t) / batch_size))
            
            # Process each batch
            for batch in range(num_batches):
                # Calculate actual batch size for last batch
                current_batch_size = min(batch_size, len(t) - batch * batch_size)
                
                # Initialize arrays for batch data and labels
                batch_data = np.zeros((current_batch_size, num_frames, img_size, img_size, 3))
                batch_labels = np.zeros((current_batch_size, NUM_CLASSES))
                
                # Process each video in the batch
                for folder_idx in range(current_batch_size):
                    file_idx = batch * batch_size + folder_idx
                    folder_name = t[file_idx].strip().split(';')[0]
                    class_idx = int(t[file_idx].strip().split(';')[2])
                    
                    # List all frames in the folder
                    imgs = sorted(os.listdir(os.path.join(source_path, folder_name)))
                    
                    # Process each selected frame
                    for i, frame_idx in enumerate(frame_indices):
                        if frame_idx < len(imgs):
                            # Load the frame
                            img_path = os.path.join(source_path, folder_name, imgs[frame_idx])
                            image = cv2.imread(img_path)
                            
                            if image is not None:
                                # Detect and crop hand
                                hand_crop, _, hand_detected = detect_and_crop_hand(image, hands)
                                
                                # Preprocess the frame (either hand crop or original)
                                processed_frame = preprocess_frame(hand_crop, (img_size, img_size))
                                
                                # Normalize and store each channel
                                batch_data[folder_idx, i] = normalize_func(processed_frame)
                    
                    # One-hot encode the label
                    batch_labels[folder_idx, class_idx] = 1
                
                yield batch_data, batch_labels

def build_enhanced_3d_cnn_model(input_shape, num_classes):
    """Build and return an enhanced 3D CNN model for gesture recognition"""
    model = Sequential()
    
    # First 3D Convolutional Block
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same'))
    
    # Second 3D Convolutional Block
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same'))
    
    # Third 3D Convolutional Block
    model.add(Conv3D(128, kernel_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
    
    # Fourth 3D Convolutional Block
    model.add(Conv3D(256, kernel_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
    
    # Fifth 3D Convolutional Block (added for more complexity)
    model.add(Conv3D(512, kernel_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
    
    # Flatten the output
    model.add(Flatten())
    
    # Fully connected layers with regularization
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def setup_callbacks(model_dir):
    """Set up callbacks for model training"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Model checkpoint to save the best model
    filepath = os.path.join(model_dir, 'model-{epoch:02d}-{val_loss:.4f}-{val_categorical_accuracy:.4f}.h5')
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_categorical_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    # Learning rate reduction on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # More aggressive reduction
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )
    
    return [checkpoint, reduce_lr, early_stop]

def plot_training_history(history, save_path=None):
    """Plot training and validation metrics"""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend(loc='lower right')
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend(loc='upper right')
    
    # Adjust layout and display
    plt.tight_layout()
    
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

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
                        
                        # Display action associated with gesture
                        action_map = {
                            0: "Action: Volume UP",
                            1: "Action: Volume DOWN",
                            2: "Action: Rewind 10s",
                            3: "Action: Forward 10s",
                            4: "Action: Pause"
                        }
                        cv2.putText(annotated_frame, action_map[predicted_class], (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
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

# Function to perform data augmentation
def augment_training_data(train_generator, steps_per_epoch, batch_size):
    """Apply augmentation to the training data"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Create augmentation generator
    augmentation = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2,
        channel_shift_range=0.1,
        horizontal_flip=False  # Don't flip horizontally as it changes gesture meaning
    )
    
    # Function to apply augmentation to a batch
    def augment_batch(X_batch, y_batch):
        X_aug = np.zeros_like(X_batch)
        
        # Apply augmentation to each frame sequence
        for i in range(X_batch.shape[0]):
            # Get sequence for this example
            sequence = X_batch[i]
            
            # Apply same augmentation to all frames in sequence
            for j in range(sequence.shape[0]):
                # Add batch dimension for ImageDataGenerator
                frame = sequence[j:j+1]
                # Apply augmentation
                aug_frame = next(augmentation.flow(frame, batch_size=1))[0]
                # Store augmented frame
                X_aug[i, j] = aug_frame
        
        return X_aug, y_batch
    
    # Generator that yields augmented batches
    while True:
        # Get batch from original generator
        X_batch, y_batch = next(train_generator)
        
        # Apply augmentation
        X_aug, y_aug = augment_batch(X_batch, y_batch)
        
        yield X_aug, y_aug

# Main execution function with hand detection enhancement
def main():
    # Define paths
    dataset_path = '/home/datasets/Project_data'
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    
    # Load and shuffle the train and validation files
    train_files = np.random.permutation(open(os.path.join(dataset_path, 'train.csv')).readlines())
    val_files = np.random.permutation(open(os.path.join(dataset_path, 'val.csv')).readlines())
    
    # Create model directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = f"hand_gesture_model_{timestamp}"
    
    # Print dataset information
    print(f"Total training samples: {len(train_files)}")
    print(f"Total validation samples: {len(val_files)}")
    
    # Check class distribution
    train_labels = [int(file.strip().split(';')[2]) for file in train_files]
    val_labels = [int(file.strip().split(';')[2]) for file in val_files]
    
    print("\nTraining set class distribution:")
    for i in range(NUM_CLASSES):
        count = train_labels.count(i)
        print(f"  Class {i} ({GESTURE_CLASSES[i]}): {count} samples ({count/len(train_labels)*100:.1f}%)")
    
    print("\nValidation set class distribution:")
    for i in range(NUM_CLASSES):
        count = val_labels.count(i)
        print(f"  Class {i} ({GESTURE_CLASSES[i]}): {count} samples ({count/len(val_labels)*100:.1f}%)")
    
    # Create data generators with hand detection
    print("\nCreating data generators with hand detection...")
    train_generator = create_data_generator(
        train_path, train_files, BATCH_SIZE, IMG_SIZE, FRAME_INDICES, normalize_frames
    )
    
    val_generator = create_data_generator(
        val_path, val_files, BATCH_SIZE, IMG_SIZE, FRAME_INDICES, normalize_frames
    )
    
    # Apply data augmentation to training generator
    print("Setting up data augmentation...")
    augmented_train_generator = augment_training_data(
        train_generator, 
        int(np.ceil(len(train_files) / BATCH_SIZE)), 
        BATCH_SIZE
    )
    
    # Calculate steps per epoch
    steps_per_epoch = int(np.ceil(len(train_files) / BATCH_SIZE))
    validation_steps = int(np.ceil(len(val_files) / BATCH_SIZE))
    
    # Build enhanced model
    print("Building enhanced 3D CNN model...")
    input_shape = (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    model = build_enhanced_3d_cnn_model(input_shape, NUM_CLASSES)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Set up callbacks
    callbacks = setup_callbacks(model_dir)
    
    # Train model with augmented data
    print("\nTraining enhanced model with hand detection and augmentation...")
    history = model.fit(
        augmented_train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=50,  # Increased epochs
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, os.path.join(model_dir, 'training_history.png'))
    
    # Save the final model
    model.save(os.path.join(model_dir, 'final_hand_gesture_model.h5'))
    
    print(f"\nEnhanced model training complete. Model saved to {model_dir}/final_hand_gesture_model.h5")
    
    # Test with webcam
    print("\nLaunching real-time hand gesture recognition...")
    capture_and_predict(model)

# Function to deploy the trained model on a TV with hand-specific controls
def deploy_on_tv(model_path):
    """
    Function to deploy the trained model on a smart TV system
    
    This is a simplified representation of what would be needed for actual deployment.
    In a real system, this would interface with the TV's operating system and APIs.
    """
    print("Loading hand gesture model for TV deployment...")
    model = load_model(model_path)
    
    # This would be replaced with actual TV system integration code
    print("Model loaded successfully. Integrating with TV system...")
    print("Setting up enhanced hand gesture control for the following commands:")
    for idx, gesture in GESTURE_CLASSES.items():
        actions = ["Increase Volume", "Decrease Volume", "Rewind 10s", "Forward 10s", "Pause/Play"]
        print(f"  - {gesture}: {actions[idx]}")
    
    print("\nConnecting to TV camera...")
    print("Starting enhanced hand gesture recognition system...")
    
    # For simulation, we'll use the webcam function with hand detection
    capture_and_predict(model)

# Interface for live TV control using gesture recognition
def tv_control_interface(model_path):
    """
    Simplified interface for controlling a TV with hand gestures
    
    Parameters:
    model_path (str): Path to the trained gesture recognition model
    """
    # Load the model
    print("Loading gesture recognition model...")
    model = load_model(model_path)
    
    # Create TV simulation window
    tv_window = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # TV state variables
    volume = 50  # Initial volume (0-100)
    playing = True  # Playing state
    position = 0  # Current position in seconds
    content_length = 5400  # Example: 90 minutes content
    last_action_time = datetime.datetime.now()
    last_action = None
    
    # Function to update TV simulation
    def update_tv_display(window, volume, playing, position, last_action):
        window.fill(0)  # Clear window
        
        # Draw TV frame
        cv2.rectangle(window, (20, 20), (620, 460), (50, 50, 50), 2)
        
        # Display volume
        cv2.rectangle(window, (30, 400), (30 + int(volume * 5.8), 420), (0, 255, 0), -1)
        cv2.putText(window, f"Volume: {volume}%", (30, 395),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display playback status
        status = "Playing" if playing else "Paused"
        cv2.putText(window, status, (30, 350),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display position
        minutes = position // 60
        seconds = position % 60
        total_minutes = content_length // 60
        total_seconds = content_length % 60
        cv2.putText(window, f"Position: {minutes:02d}:{seconds:02d} / {total_minutes:02d}:{total_seconds:02d}", 
                   (30, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display last action if recent
        if last_action and (datetime.datetime.now() - last_action_time).total_seconds() < 3:
            cv2.putText(window, f"Last action: {last_action}", 
                       (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return window
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Buffer to store processed frames
    frame_buffer = []
    
    print("Starting TV control interface. Press 'q' to quit.")
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while True:
            # Update position if playing
            if playing:
                position = min(position + 1, content_length)  # increment 1 second
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Detect and crop hand
            hand_crop, annotated_frame, hand_detected = detect_and_crop_hand(frame, hands)
            
            # Process frame for prediction
            if hand_detected:
                processed_frame = preprocess_frame(hand_crop, (IMG_SIZE, IMG_SIZE))
                
                # Add to buffer
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
                    
                    # Apply action if confidence is high enough
                    if confidence > 0.65:
                        gesture = GESTURE_CLASSES[predicted_class]
                        
                        # Apply corresponding action
                        if predicted_class == 0:  # Thumbs Up - Increase volume
                            volume = min(volume + 5, 100)
                            last_action = "Volume UP"
                        elif predicted_class == 1:  # Thumbs Down - Decrease volume
                            volume = max(volume - 5, 0)
                            last_action = "Volume DOWN"
                        elif predicted_class == 2:  # Left Swipe - Rewind
                            position = max(position - 10, 0)
                            last_action = "Rewind 10s"
                        elif predicted_class == 3:  # Right Swipe - Forward
                            position = min(position + 10, content_length)
                            last_action = "Forward 10s"
                        elif predicted_class == 4:  # Stop - Pause/Play
                            playing = not playing
                            last_action = "Toggle Play/Pause"
                        
                        last_action_time = datetime.datetime.now()
                        
                        # Display gesture and confidence
                        cv2.putText(annotated_frame, f"{gesture} ({confidence:.2f})", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Clear buffer gradually if no hand detected
                if len(frame_buffer) > 0 and len(frame_buffer) % 3 == 0:
                    frame_buffer.pop(0)
            
            # Update and show TV simulation
            tv_display = update_tv_display(tv_window.copy(), volume, playing, position, last_action)
            cv2.imshow('TV Simulation', tv_display)
            
            # Show camera feed with hand detection
            cv2.imshow('Hand Gesture Control', annotated_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    choice = input("Select an option:\n1. Train enhanced hand gesture model\n2. Test existing model with webcam\n3. Launch TV control interface\nYour choice (1/2/3): ")
    
    if choice == '1':
        main()
    elif choice == '2':
        model_path = input("Enter the path to your trained model (.h5 file): ")
        model = load_model(model_path)
        capture_and_predict(model)
    elif choice == '3':
        model_path = input("Enter the path to your trained model (.h5 file): ")
        tv_control_interface(model_path)
    else:
        print("Invalid choice. Exiting.")