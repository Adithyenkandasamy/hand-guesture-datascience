#!/usr/bin/env python3
"""
Gesture Recognition YouTube Controller (Linux Compatible)

Uses a pre-trained CNN model to recognize hand gestures from a webcam feed and map them to 
YouTube playback controls.
"""

import cv2
import numpy as np
import json
import time
import os
import subprocess
from tensorflow.keras.models import load_model
import pyautogui


class GestureController:
    """Handles gesture recognition and YouTube control."""

    # Gesture-to-keyboard action mapping
    GESTURE_ACTIONS = {
        "Thumbs Up": {"key": "up", "description": "Increase Volume"},
        "Thumbs Down": {"key": "down", "description": "Decrease Volume"},
        "Left Swipe": {"key": "left", "description": "Seek Back 10s"},
        "Right Swipe": {"key": "right", "description": "Seek Forward 10s"},
        "Stop": {"key": "space", "description": "Pause/Play"},
    }

    def __init__(self, model_path, labels_path, cooldown_time=1.0):
        """
        Initialize the GestureController with model and labels.

        Args:
            model_path (str): Path to the trained Keras model (.h5)
            labels_path (str): Path to the JSON file mapping class indices to gesture names
            cooldown_time (float): Time to wait before triggering the same action again (seconds)
        """
        self.model = self._load_model(model_path)
        self.labels_dict, self.class_indices = self._load_labels(labels_path)
        self.cooldown_time = cooldown_time
        self.last_action_time = 0
        self.last_gesture = None
        self.cap = None

    def _load_model(self, model_path):
        """Load the pre-trained Keras model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading model from {model_path}...")
        return load_model(model_path)

    def _load_labels(self, labels_path):
        """Load the labels mapping from JSON file."""
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        print(f"Loading labels from {labels_path}...")
        with open(labels_path, "r") as f:
            labels_map = json.load(f)

        unique_gestures = sorted(set(labels_map.values()))
        class_indices = {i: gesture for i, gesture in enumerate(unique_gestures)}

        print("Class index to gesture mapping:", class_indices)
        return labels_map, class_indices

    def _preprocess_frame(self, frame):
        """Preprocess the frame before feeding it to the model."""
        resized = cv2.resize(frame, (128, 128))
        normalized = resized / 255.0
        batched = np.expand_dims(normalized, axis=0)
        return batched

    def _predict_gesture(self, frame):
        """Predict the gesture from a frame."""
        processed_frame = self._preprocess_frame(frame)
        predictions = self.model.predict(processed_frame, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]

        if predicted_class_idx in self.class_indices:
            return self.class_indices[predicted_class_idx], confidence
        return "Unknown", confidence

    def _perform_action(self, gesture):
        """Perform the action associated with the detected gesture."""
        if gesture not in self.GESTURE_ACTIONS:
            return False

        current_time = time.time()
        if (current_time - self.last_action_time < self.cooldown_time) and (
            gesture == self.last_gesture
        ):
            return False

        action = self.GESTURE_ACTIONS[gesture]
        print(f"Performing action: {gesture} -> {action['description']} (Key: {action['key']})")
        pyautogui.press(action["key"])

        self.last_action_time = current_time
        self.last_gesture = gesture
        return True

    def _focus_chrome(self):
        """Ensures Chrome is the active window using xdotool (Linux only)."""
        try:
            subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--class", "chrome", "windowactivate"],
                check=True,
            )
        except subprocess.CalledProcessError:
            print("Could not focus Chrome. Make sure it's open.")

    def start_webcam(self, camera_id=0):
        """Start the webcam capture."""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with ID {camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def run(self):
        """Run the gesture recognition loop."""
        if self.cap is None:
            self.start_webcam()

        print("Gesture Recognition Started. Press 'q' to quit.")
        print("\nAvailable Gestures:")
        for gesture, action in self.GESTURE_ACTIONS.items():
            print(f"- {gesture}: {action['description']} (Key: {action['key']})")

        try:
            frame_count = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame. Exiting...")
                    break

                frame = cv2.flip(frame, 1)

                frame_count += 1
                if frame_count % 3 != 0:
                    cv2.imshow("Gesture Control", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                display_frame = frame.copy()
                gesture, confidence = self._predict_gesture(frame)

                print(f"Raw prediction: {gesture} with confidence {confidence:.4f}")

                action_performed = self._perform_action(gesture)

                cv2.putText(
                    display_frame,
                    f"Gesture: {gesture} ({confidence:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if action_performed else (0, 0, 255),
                    2,
                )

                cv2.imshow("Gesture Control", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Gesture recognition stopped.")


def main():
    """Main function to run the gesture controller."""
    import argparse

    parser = argparse.ArgumentParser(description="Gesture-based YouTube Controller")
    parser.add_argument("--model", type=str, default="gesture_model.h5", help="Path to model (.h5)")
    parser.add_argument("--labels", type=str, default="image_labels.json", help="Path to labels JSON")
    parser.add_argument("--cooldown", type=float, default=1.0, help="Cooldown time (seconds)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    try:
        os.environ["QT_QPA_PLATFORM"] = "xcb"

        controller = GestureController(
            model_path=args.model, labels_path=args.labels, cooldown_time=args.cooldown
        )
        controller.start_webcam(camera_id=args.camera)
        controller.run()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
