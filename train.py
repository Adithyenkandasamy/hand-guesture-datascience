import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import mediapipe as mp
import json
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import gc  # For manual garbage collection

# Enable memory growth to avoid memory issues
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Configuration parameters
IMG_SIZE = 100  # Size to resize frames to (height and width)
NUM_FRAMES = 18  # Number of frames to use from each video
NUM_CHANNELS = 3  # RGB channels
NUM_CLASSES = 5  # Number of gesture classes
NUM_LANDMARKS = 21  # Number of hand landmarks from MediaPipe

# Reduce batch size to avoid memory issues
BATCH_SIZE = 8

# Gesture labels
GESTURE_CLASSES = {
    0: "Thumbs Up",
    1: "Thumbs Down",
    2: "Left Swipe",
    3: "Right Swipe",
    4: "Stop"
}

# Initialize MediaPipe - only initialize once to save memory
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for better performance with sequential frames
    max_num_hands=1,
    min_detection_confidence=0.5
)

def load_csv_data(csv_path, base_folder=None):
    """Load and parse a CSV file with video folder names and class labels."""
    data = []
    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()
            
            # Skip header if present
            if lines and (lines[0].lower().startswith('image') or lines[0].lower().startswith('imagename')):
                lines = lines[1:]
            
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                parts = line.split(';')
                if len(parts) >= 3:
                    folder_name = parts[0]
                    
                    # Verify folder exists before adding if base_folder is provided
                    if base_folder:
                        folder_path = os.path.join(base_folder, folder_name)
                        if not os.path.exists(folder_path):
                            print(f"Warning: Folder does not exist: {folder_path}")
                            continue
                        
                    try:
                        class_idx = int(parts[2])
                        data.append((folder_name, class_idx))
                    except ValueError:
                        print(f"Invalid class index in line: {line}")
    except Exception as e:
        print(f"Error loading CSV data: {e}")
    
    return data

def extract_hand_landmarks(image):
    """Extract hand landmarks from an image using MediaPipe"""
    # Convert to RGB for MediaPipe if needed
    if image.shape[2] == 3 and image.dtype == np.uint8:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image  # Avoid unnecessary conversions
    
    # Process the image
    results = hands.process(image_rgb)
    
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

def get_hand_bounding_box(landmarks, image_shape, padding=20):
    """Get hand bounding box from landmarks with safety checks"""
    if np.all(landmarks == 0):
        return None
    
    height, width = image_shape[:2]
    
    # Convert normalized coordinates to pixel values
    x_coords = landmarks[:, 0] * width
    y_coords = landmarks[:, 1] * height
    
    # Get non-zero coordinates
    valid_x = x_coords[np.nonzero(x_coords)]
    valid_y = y_coords[np.nonzero(y_coords)]
    
    if len(valid_x) == 0 or len(valid_y) == 0:
        return None
    
    # Calculate bounding box
    x_min, y_min = int(np.min(valid_x)), int(np.min(valid_y))
    x_max, y_max = int(np.max(valid_x)), int(np.max(valid_y))
    
    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    # Ensure we have a valid bounding box
    if x_min >= x_max or y_min >= y_max:
        return None
        
    return (x_min, y_min, x_max, y_max)

def preprocess_frame(frame, target_size=(IMG_SIZE, IMG_SIZE)):
    """Preprocess a single frame: crop if necessary and resize with error handling"""
    try:
        # Convert to PIL Image if not already
        if not isinstance(frame, Image.Image):
            # Handle different image formats
            if isinstance(frame, np.ndarray):
                if frame.size == 0:  # Check for empty array
                    return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
                frame = Image.fromarray(frame)

        # Ensure the image has the correct mode
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')

        # Resize to target size
        frame = frame.resize(target_size)

        # Convert to numpy array
        frame = np.asarray(frame).astype(np.float32)
        
        return frame
        
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        # Return a blank frame in case of error
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)

def normalize_frames(frames):
    """Normalize pixel values to range [0, 1]"""
    return frames / 255.0

def normalize_landmarks(landmarks):
    """Normalize landmark coordinates to range [0, 1]"""
    # Already normalized by MediaPipe, just flatten
    return landmarks.flatten()

def load_video_sequence(base_path, folder_name, max_frames=NUM_FRAMES, save_landmarks=True):
    """Load a sequence of frames from a video folder and extract hand landmarks with error handling"""
    folder_path = os.path.join(base_path, folder_name)
    
    # Initialize empty arrays of the right shape
    empty_frames = np.zeros((max_frames, IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype=np.float32)
    empty_landmarks = np.zeros((max_frames, NUM_LANDMARKS, 3), dtype=np.float32)
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return empty_frames, empty_landmarks
    
    # Get list of image files
    try:
        img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    except Exception as e:
        print(f"Error listing directory: {e}")
        return empty_frames, empty_landmarks
    
    if len(img_files) == 0:
        print(f"No image files found in {folder_path}")
        return empty_frames, empty_landmarks
    
    # Load frames and extract landmarks
    frames = []
    landmarks_sequence = []
    
    for i, img_file in enumerate(img_files):
        if i >= max_frames:
            break
        
        try:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load image: {img_path}")
                # Add blank frame and landmarks
                frames.append(np.zeros((IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype=np.float32))
                landmarks_sequence.append(np.zeros((NUM_LANDMARKS, 3), dtype=np.float32))
                continue
            
            # Extract hand landmarks
            landmarks, hand_detected = extract_hand_landmarks(img)
            
            # If hand detected, crop image to hand area
            if hand_detected:
                bbox = get_hand_bounding_box(landmarks, img.shape)
                if bbox is not None:
                    x_min, y_min, x_max, y_max = bbox
                    # Safety check for valid crop dimensions
                    if x_min < x_max and y_min < y_max:
                        img = img[y_min:y_max, x_min:x_max]
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Preprocess frame
            processed_frame = preprocess_frame(img)
            frames.append(processed_frame)
            
            # Add landmarks to sequence
            landmarks_sequence.append(landmarks)
            
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            # Add blank frame and landmarks
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype=np.float32))
            landmarks_sequence.append(np.zeros((NUM_LANDMARKS, 3), dtype=np.float32))
    
    # Pad with zeros if we don't have enough frames
    while len(frames) < max_frames:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype=np.float32))
        landmarks_sequence.append(np.zeros((NUM_LANDMARKS, 3), dtype=np.float32))
    
    # Ensure we have the right number of frames
    frames = frames[:max_frames]
    landmarks_sequence = landmarks_sequence[:max_frames]
    
    # Convert to numpy arrays
    frames_array = np.array(frames)
    landmarks_array = np.array(landmarks_sequence)
    
    # Save landmarks if requested
    if save_landmarks:
        try:
            landmarks_folder = os.path.join(os.path.dirname(base_path), "landmarks")
            os.makedirs(landmarks_folder, exist_ok=True)
            
            landmarks_file = os.path.join(landmarks_folder, f"{folder_name}.npy")
            np.save(landmarks_file, landmarks_array)
        except Exception as e:
            print(f"Error saving landmarks: {e}")
    
    return frames_array, landmarks_array

def data_generator(data_list, base_path, batch_size=BATCH_SIZE, use_landmarks=True):
    """Generator to yield batches of video sequences with landmarks and memory management"""
    num_samples = len(data_list)
    
    while True:
        # Shuffle the data
        indices = np.random.permutation(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            # Clear some memory between batches
            gc.collect()
            
            batch_indices = indices[start_idx:min(start_idx + batch_size, num_samples)]
            batch_size_actual = len(batch_indices)
            
            batch_x = np.zeros((batch_size_actual, NUM_FRAMES, IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
            batch_landmarks = np.zeros((batch_size_actual, NUM_FRAMES, NUM_LANDMARKS * 3))
            batch_y = np.zeros(batch_size_actual, dtype=np.int32)
            
            for i, idx in enumerate(batch_indices):
                try:
                    folder_name, class_idx = data_list[idx]
                    
                    # Load video frames and landmarks
                    frames, landmarks_seq = load_video_sequence(base_path, folder_name)
                    
                    # Normalize frames
                    normalized_frames = normalize_frames(frames)
                    batch_x[i] = normalized_frames
                    
                    # Process landmarks
                    for j, landmarks in enumerate(landmarks_seq):
                        batch_landmarks[i, j] = normalize_landmarks(landmarks)
                    
                    # Store the class index directly
                    batch_y[i] = class_idx
                    
                except Exception as e:
                    print(f"Error in generator for sample {idx}: {e}")
                    # Keep zeros for this sample
            
            if use_landmarks:
                yield (batch_x, batch_landmarks), batch_y
            else:
                yield batch_x, batch_y

def build_hybrid_model(reduced_complexity=True):
    """
    Build a hybrid model that uses both image features and hand landmarks
    with option for reduced complexity to avoid memory issues
    """
    # Image input branch
    img_input = Input(shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
    
    # CNN for image features - can be reduced in complexity
    if reduced_complexity:
        # Simpler CNN architecture
        cnn = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(img_input)
        cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)
        cnn = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(cnn)
        cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)
        cnn = TimeDistributed(Dropout(0.25))(cnn)
        cnn = TimeDistributed(Flatten())(cnn)
    else:
        # Original more complex CNN
        cnn = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(img_input)
        cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)
        cnn = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(cnn)
        cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)
        cnn = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(cnn)
        cnn = TimeDistributed(MaxPooling2D((2, 2)))(cnn)
        cnn = TimeDistributed(Dropout(0.25))(cnn)
        cnn = TimeDistributed(Flatten())(cnn)
    
    # Landmarks input branch
    landmarks_input = Input(shape=(NUM_FRAMES, NUM_LANDMARKS * 3))
    
    # Dense layers for landmarks - also simplified
    if reduced_complexity:
        landmarks_features = TimeDistributed(Dense(64, activation='relu'))(landmarks_input)
        landmarks_features = TimeDistributed(Dropout(0.25))(landmarks_features)
    else:
        landmarks_features = TimeDistributed(Dense(128, activation='relu'))(landmarks_input)
        landmarks_features = TimeDistributed(Dense(256, activation='relu'))(landmarks_features)
        landmarks_features = TimeDistributed(Dropout(0.25))(landmarks_features)
    
    # Merge two branches
    merged = Concatenate()([cnn, landmarks_features])
    
    # LSTM layers for sequence learning - simplified if needed
    if reduced_complexity:
        lstm = LSTM(128, return_sequences=True)(merged)
        lstm = Dropout(0.3)(lstm)
        lstm = LSTM(64)(lstm)
    else:
        lstm = LSTM(256, return_sequences=True)(merged)
        lstm = Dropout(0.3)(lstm)
        lstm = LSTM(128)(lstm)
    
    lstm = Dropout(0.3)(lstm)
    
    # Output layer
    output = Dense(NUM_CLASSES, activation='softmax')(lstm)
    
    # Create model
    model = Model(inputs=[img_input, landmarks_input], outputs=output)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_image_only_model(reduced_complexity=True):
    """
    Build a ConvLSTM model using only image data (no landmarks)
    with option for reduced complexity
    """
    if reduced_complexity:
        model = Sequential([
            TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'), 
                          input_shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, NUM_CHANNELS)),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Dropout(0.25)),
            TimeDistributed(Flatten()),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(NUM_CLASSES, activation='softmax')
        ])
    else:
        model = Sequential([
            TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), 
                          input_shape=(NUM_FRAMES, IMG_SIZE, IMG_SIZE, NUM_CHANNELS)),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Dropout(0.25)),
            TimeDistributed(Flatten()),
            LSTM(256, return_sequences=True),
            Dropout(0.3),
            LSTM(128),
            Dropout(0.3),
            Dense(NUM_CLASSES, activation='softmax')
        ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    return fig

def evaluate_hybrid_model(model, eval_data, base_path):
    """Evaluate the hybrid model performance on validation data"""
    # Predictions for all validation samples
    true_classes = []
    pred_classes = []
    
    for folder_name, class_idx in eval_data:
        try:
            # Load video sequence
            frames, landmarks_seq = load_video_sequence(base_path, folder_name, save_landmarks=False)
            
            # Prepare input
            img_input = np.zeros((1, NUM_FRAMES, IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
            landmarks_input = np.zeros((1, NUM_FRAMES, NUM_LANDMARKS * 3))
            
            normalized_frames = normalize_frames(frames)
            img_input[0] = normalized_frames
            
            for j, landmarks in enumerate(landmarks_seq):
                landmarks_input[0, j] = normalize_landmarks(landmarks)
            
            # Make prediction
            prediction = model.predict([img_input, landmarks_input], verbose=0)[0]
            predicted_class = np.argmax(prediction)
            
            # Store true and predicted classes
            true_classes.append(class_idx)
            pred_classes.append(predicted_class)
            
        except Exception as e:
            print(f"Error evaluating sample {folder_name}: {e}")
    
    # Calculate confusion matrix only if we have predictions
    if true_classes and pred_classes:
        cm = confusion_matrix(true_classes, pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=list(GESTURE_CLASSES.values()),
                  yticklabels=list(GESTURE_CLASSES.values()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, pred_classes, 
                                  target_names=list(GESTURE_CLASSES.values())))
        
        return cm
    else:
        print("Not enough data for evaluation")
        return None

def save_model_metadata(model_path, use_landmarks=True):
    """Save model configuration parameters for use during inference"""
    metadata = {
        "img_size": IMG_SIZE,
        "num_frames": NUM_FRAMES,
        "num_channels": NUM_CHANNELS,
        "num_classes": NUM_CLASSES,
        "gesture_classes": GESTURE_CLASSES,
        "use_landmarks": use_landmarks,
        "num_landmarks": NUM_LANDMARKS
    }
    
    # Save as JSON file
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Model metadata saved to {model_path}")
    except Exception as e:
        print(f"Error saving metadata: {e}")

def main():
    print("Hand Gesture Recognition - Model Training with Hand Landmarks")
    print("----------------------------------------------------------")
    
    # Define paths
    dataset_path = "/home/jinwoo/Desktop/hand-guesture-datascience/"
    train_csv = "/home/jinwoo/Desktop/hand-guesture-datascience/train.csv"
    val_csv = "/home/jinwoo/Desktop/hand-guesture-datascience/val.csv"
    model_output_path = "/home/jinwoo/Desktop/hand-guesture-datascience/models"
    
    # Configuration options
    use_landmarks = True
    reduced_complexity = True  # Set to True to use simpler model architecture
    incremental_training = True  # Set to True to train in smaller chunks
    
    # Create output directory if not exists
    os.makedirs(model_output_path, exist_ok=True)
    
    # Load training and validation data
    print("\nLoading dataset information...")
    train_base_path = os.path.join(dataset_path, "train")
    val_base_path = os.path.join(dataset_path, "val")
    
    train_data = load_csv_data(train_csv, train_base_path)
    val_data = load_csv_data(val_csv, val_base_path)
    
    print(f"Found {len(train_data)} training samples")
    print(f"Found {len(val_data)} validation samples")
    
    # Create data generators
    print("\nPreparing data generators...")
    
    # Choose appropriate batch size based on available memory
    batch_size = BATCH_SIZE
    print(f"Using batch size: {batch_size}")
    
    # Build model based on user choice
    print("\nBuilding model...")
    if use_landmarks:
        print("Using hybrid model with both image data and hand landmarks")
        print(f"Model complexity: {'Reduced' if reduced_complexity else 'Full'}")
        model = build_hybrid_model(reduced_complexity=reduced_complexity)
        train_gen = data_generator(train_data, train_base_path, batch_size, use_landmarks=True)
        val_gen = data_generator(val_data, val_base_path, batch_size, use_landmarks=True)
    else:
        print("Using image-only model")
        print(f"Model complexity: {'Reduced' if reduced_complexity else 'Full'}")
        model = build_image_only_model(reduced_complexity=reduced_complexity)
        train_gen = data_generator(train_data, train_base_path, batch_size, use_landmarks=False)
        val_gen = data_generator(val_data, val_base_path, batch_size, use_landmarks=False)
    
    # Define model summary
    model.summary()
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_output_path, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Train the model
    print("\nTraining model...")
    
    # Calculate steps
    steps_per_epoch = max(1, len(train_data) // batch_size)
    validation_steps = max(1, len(val_data) // batch_size)
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    if incremental_training:
        # Train in smaller chunks to avoid memory issues
        epochs_per_increment = 2
        total_epochs = 10
        complete_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        
        for i in range(0, total_epochs, epochs_per_increment):
            print(f"\nTraining increment {i//epochs_per_increment + 1}/{total_epochs//epochs_per_increment}")
            
            history = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_gen,
                validation_steps=validation_steps,
                epochs=epochs_per_increment,
                callbacks=[checkpoint, early_stopping, reduce_lr],
                verbose=1
            )
            
            # Accumulate history
            for key in complete_history:
                if key in history.history:
                    complete_history[key].extend(history.history[key])
            
            # Save checkpoint after each increment
            model.save(os.path.join(model_output_path, f'model_checkpoint_{i+epochs_per_increment}.h5'))
            
            # Clear memory
            tf.keras.backend.clear_session()
            gc.collect()
            
        # Convert complete history to History object
        class HistoryWrapper:
            def __init__(self, history_dict):
                self.history = history_dict
                
        history = HistoryWrapper(complete_history)
    else:
        # Standard training
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=10,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=1
        )
    
    # Save model configuration
    save_model_metadata(os.path.join(model_output_path, 'model_metadata.json'), use_landmarks)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    print("\nEvaluating model...")
    evaluate_hybrid_model(model, val_data, val_base_path)
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()