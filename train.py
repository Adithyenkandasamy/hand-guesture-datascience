import numpy as np
import os
import cv2
import datetime
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
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

# Helper functions
def normalize_frames(frames):
    """Normalize pixel values to range [0, 1]"""
    return frames / 255.0

def preprocess_frame(frame, target_size=(IMG_SIZE, IMG_SIZE)):
    """Preprocess a single frame: crop if necessary and resize"""
    # Convert to PIL Image if not already
    if not isinstance(frame, Image.Image):
        frame = Image.fromarray(frame)

    # Crop if the width is 160 (specific to your dataset)
    if frame.size[0] == 160:
        frame = frame.crop((20, 0, 140, 120))  # left=20, top=0, right=140, bottom=120

    # Resize to target size
    frame = frame.resize(target_size)

    # Convert to numpy array
    frame = np.asarray(frame).astype(np.float32)

    return frame

def create_data_generator(source_path, file_list, batch_size, img_size, frame_indices, normalize_func):
    """Generator function to create batches of data for training/validation"""
    num_frames = len(frame_indices)
    num_batches = int(np.ceil(len(file_list) / batch_size))

    while True:
        # Shuffle the file list
        t = np.random.permutation(file_list)
        
        # Process each batch
        for batch in range(num_batches):
            # Calculate actual batch size for last batch
            current_batch_size = min(batch_size, len(file_list) - batch * batch_size)
            
            # Initialize arrays for batch data and labels
            batch_data = np.zeros((current_batch_size, num_frames, img_size, img_size, 3))
            batch_labels = np.zeros((current_batch_size, NUM_CLASSES))
            
            # Process each video in the batch
            for folder_idx in range(current_batch_size):
                file_idx = batch * batch_size + folder_idx
                line = t[file_idx].strip()
                
                # Skip header row if it exists
                if line.lower().startswith('image') or line.lower().startswith('imagename'):
                    continue
                
                parts = line.split(';')
                if len(parts) < 3:
                    continue
                
                folder_name = parts[0]
                
                # Try to convert label to integer
                try:
                    class_idx = int(parts[2])
                except ValueError:
                    continue

                # List all frames in the folder
                imgs = sorted(os.listdir(os.path.join(source_path, folder_name)))

                # Process each selected frame
                for i, frame_idx in enumerate(frame_indices):
                    if frame_idx < len(imgs):
                        # Load the frame
                        img_path = os.path.join(source_path, folder_name, imgs[frame_idx])
                        image = Image.open(img_path)

                        # Preprocess the frame
                        processed_frame = preprocess_frame(image, (img_size, img_size))

                        # Normalize and store each channel
                        for c in range(3):  # RGB channels
                            batch_data[folder_idx, i, :, :, c] = normalize_func(processed_frame[:, :, c])

                # One-hot encode the label
                batch_labels[folder_idx, class_idx] = 1

            yield batch_data, batch_labels

def build_3d_cnn_model(input_shape, num_classes):
    """Build and return a 3D CNN model for gesture recognition"""
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

    # Flatten the output
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
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
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
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

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix from model predictions"""
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a figure
    plt.figure(figsize=(10, 8))

    # Plot the confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')

    # Save if a path is provided
    if save_path:
        plt.savefig(save_path)

    plt.show()

# Camera capture and real-time prediction
def capture_and_predict(model, camera_source=0, prediction_threshold=0.7):
    """Capture video from webcam and make real-time predictions"""
    cap = cv2.VideoCapture(camera_source)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Buffer to store frames
    frame_buffer = []

    print("Starting real-time gesture recognition. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame, (IMG_SIZE, IMG_SIZE))

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
                cv2.putText(frame, f"{gesture} ({confidence:.2f})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display action associated with gesture (still keeping this for webcam demo)
                action_map = {
                    0: "Action: Volume UP",
                    1: "Action: Volume DOWN",
                    2: "Action: Rewind 10s",
                    3: "Action: Forward 10s",
                    4: "Action: Pause"
                }
                cv2.putText(frame, action_map[predicted_class], (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Gesture Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Main execution function
def main():
    # Define paths
    dataset_path = '/home/jinwoo/Desktop/hand-guesture-datascience/'
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    # Load and shuffle the train and validation files
    train_files = np.random.permutation(open(os.path.join(dataset_path, 'train.csv')).readlines())
    val_files = np.random.permutation(open(os.path.join(dataset_path, 'val.csv')).readlines())

    # Create model directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = f"gesture_model_{timestamp}"

    # Print dataset information
    print(f"Total training samples: {len(train_files)}")
    print(f"Total validation samples: {len(val_files)}")

    # Check class distribution
    train_labels = []
    for file in train_files:
        try:
            label = int(file.strip().split(';')[2])
            train_labels.append(label)
        except ValueError:
            continue  # Skip non-numeric labels

    val_labels = []
    for file in val_files:
        try:
            label = int(file.strip().split(';')[2])
            val_labels.append(label)
        except ValueError:
            continue  # Skip non-numeric labels

    print("\nTraining set class distribution:")
    for i in range(NUM_CLASSES):
        count = train_labels.count(i)
        print(f"  Class {i} ({GESTURE_CLASSES[i]}): {count} samples ({count/len(train_labels)*100:.1f}%)")

    print("\nValidation set class distribution:")
    for i in range(NUM_CLASSES):
        count = val_labels.count(i)
        print(f"  Class {i} ({GESTURE_CLASSES[i]}): {count} samples ({count/len(val_labels)*100:.1f}%)")

    # Create data generators
    train_generator = create_data_generator(
        train_path, train_files, BATCH_SIZE, IMG_SIZE, FRAME_INDICES, normalize_frames
    )

    val_generator = create_data_generator(
        val_path, val_files, BATCH_SIZE, IMG_SIZE, FRAME_INDICES, normalize_frames
    )

    # Calculate steps per epoch
    steps_per_epoch = int(np.ceil(len(train_files) / BATCH_SIZE))
    validation_steps = int(np.ceil(len(val_files) / BATCH_SIZE))

    # Build model
    input_shape = (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    model = build_3d_cnn_model(input_shape, NUM_CLASSES)

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    # Print model summary
    model.summary()

    # Set up callbacks
    callbacks = setup_callbacks(model_dir)

    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,  # You can adjust this
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plot_training_history(history, os.path.join(model_dir, 'training_history.png'))

    # Save the final model
    model.save(os.path.join(model_dir, 'final_gesture_model.h5'))

    print(f"\nModel training complete. Model saved to {model_dir}/final_gesture_model.h5")

    # Optional: Test with webcam
    print("\nWould you like to test the model with your webcam? (y/n)")
    response = input().lower()
    if response == 'y':
        capture_and_predict(model)

if __name__ == "__main__":
    main()
