import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# Configuration
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 30

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_image(image_path):
    """Load and preprocess a single image."""
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            
        # Resize image
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        img = img / 255.0
        
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))

def create_model():
    """Create a CNN model for gesture recognition."""
    logger.info("Creating model architecture...")
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same',
                    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(MaxPooling2D((2, 2)))
    
    # Second convolutional block
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    
    # Third convolutional block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    
    # Global average pooling to reduce parameters
    model.add(GlobalAveragePooling2D())
    
    # Dense layers for classification
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    
    return model

def train_gesture_model(csv_path, base_image_dir, model_save_path):
    """Train a gesture recognition model using CSV file with image paths."""
    start_time = time.time()
    logger.info(f"Starting model training process...")
    logger.info(f"CSV file: {csv_path}")
    logger.info(f"Base image directory: {base_image_dir}")
    
    try:
        # Load CSV file without headers
        logger.info("Loading CSV file...")
        df = pd.read_csv(csv_path, header=None)
        logger.info(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")
        
        # Display first few rows to understand structure
        logger.info(f"First 5 rows of CSV file:\n{df.head()}")
        
        # Assuming first column is image path and second column is gesture label
        if len(df.columns) < 2:
            logger.error("CSV should have at least 2 columns: image_path and gesture")
            return None
            
        # Rename columns for clarity
        df.columns = ['image_path'] + [f'col_{i}' for i in range(1, len(df.columns))]
        df['gesture'] = df['col_1']  # Second column is gesture
        
        # Map gesture names to numerical values
        gesture_map = {
            'Thumbs_Up': 0, 
            'Thumbs_Down': 1,
            'Left_Swipe': 2,
            'Right_Swipe': 3,
            'Stop': 4
        }
        
        # Create full image paths
        logger.info("Processing image paths...")
        df['full_path'] = df['image_path'].apply(
            lambda x: os.path.join(base_image_dir, x) if not os.path.isabs(x) else x
        )
        
        # Check if images exist
        df['image_exists'] = df['full_path'].apply(lambda x: os.path.exists(x))
        missing_count = len(df) - df['image_exists'].sum()
        logger.info(f"Found {missing_count} missing images out of {len(df)}")
        
        if missing_count > 0:
            # Show some examples of missing images
            missing_examples = df[~df['image_exists']]['full_path'].head(5).tolist()
            logger.warning(f"Examples of missing images: {missing_examples}")
            
            # Filter out missing images
            df = df[df['image_exists']]
            logger.info(f"Continuing with {len(df)} valid images")
            
            if len(df) == 0:
                logger.error("No valid images found. Check your image paths and base directory.")
                return None
        
        # Map gesture labels to numerical values
        logger.info("Mapping gesture labels...")
        df['label'] = df['gesture'].map(gesture_map)
        
        # Check if we have unmapped gestures
        unmapped = df[df['label'].isna()]
        if not unmapped.empty:
            unique_unmapped = unmapped['gesture'].unique()
            logger.warning(f"Found {len(unmapped)} rows with unmapped gesture labels: {unique_unmapped}")
            logger.warning(f"Valid gestures are: {list(gesture_map.keys())}")
            
            # Try to match case-insensitive
            for unknown in unique_unmapped:
                for known, value in gesture_map.items():
                    if isinstance(unknown, str) and known.lower() in unknown.lower():
                        logger.info(f"Mapping '{unknown}' to '{known}'")
                        df.loc[df['gesture'] == unknown, 'label'] = value
            
            # Filter out remaining unmapped
            df = df.dropna(subset=['label'])
            logger.info(f"Continuing with {len(df)} rows after removing unmapped gestures")
        
        # Split data into train and validation sets
        logger.info("Splitting data into train and validation sets...")
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        logger.info(f"Training set: {len(train_df)} samples")
        logger.info(f"Validation set: {len(val_df)} samples")
        
        # Count samples per class
        class_counts = train_df['label'].value_counts().sort_index()
        logger.info(f"Training samples per class: {class_counts.to_dict()}")
        
        # Load images
        logger.info("Loading training images...")
        X_train = np.array([load_image(path) for path in train_df['full_path']])
        y_train = train_df['label'].values.astype(int)
        
        logger.info("Loading validation images...")
        X_val = np.array([load_image(path) for path in val_df['full_path']])
        y_val = val_df['label'].values.astype(int)
        
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}")
        
        # Create model
        model = create_model()
        model.summary(print_fn=logger.info)
        
        # Data augmentation (using the newer TensorFlow API)
        logger.info("Setting up data augmentation...")
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomTranslation(0.2, 0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomFlip(mode="horizontal"),
        ])
        
        # Apply augmentation during training
        def augment(x_batch, y_batch):
            x_batch = data_augmentation(x_batch)
            return x_batch, y_batch
            
        # Create TF dataset for training
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
        train_dataset = train_dataset.batch(BATCH_SIZE)
        train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Create TF dataset for validation
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(BATCH_SIZE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Callbacks
        logger.info("Setting up training callbacks...")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', verbose=1)
        ]
        
        # Train model
        logger.info("Starting model training...")
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        logger.info(f"Training complete. Saving model to {model_save_path}...")
        model.save(model_save_path)
        
        # Report training results
        val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
        logger.info(f"Final validation accuracy: {val_acc:.4f}")
        logger.info(f"Final validation loss: {val_loss:.4f}")
        
        # Total training time
        training_time = time.time() - start_time
        logger.info(f"Total training time: {training_time:.2f} seconds")
        
        return model, history
        
    except Exception as e:
        logger.exception(f"Error during model training: {e}")
        return None, None

if __name__ == "__main__":
    print("=" * 50)
    print("Gesture Recognition Model Trainer")
    print("=" * 50)
    
    # Get inputs
    csv_path = '/home/jinwoo/Desktop/hand-guesture-datascience/train.csv'
    base_image_dir = '/home/jinwoo/Desktop/hand-guesture-datascience/train'
    model_save_path = "gesture_model.h5"
    
    # Validate inputs
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        exit(1)
        
    if not os.path.exists(base_image_dir):
        print(f"Error: Base image directory not found at {base_image_dir}")
        exit(1)
    
    # Train model
    model, history = train_gesture_model(csv_path, base_image_dir, model_save_path)
    
    if model is None:
        print("\nTraining failed. See logs for details.")
    else:
        print("\nTraining completed successfully!")
        print(f"Model saved to {model_save_path}")