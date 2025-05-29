import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt



class DynamicGestureTracker:
    def __init__(self, sequence_length=30):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Track one hand for simplicity
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Sequence parameters
        self.sequence_length = sequence_length
        self.current_sequence = deque(maxlen=sequence_length)

        # Feature extraction
        self.feature_size = 21 * 3  # 21 landmarks * (x, y, z)

        # Models
        self.lstm_model = None
        self.rf_model = None

        # Data storage
        self.training_data = []
        self.training_labels = []

        # Gesture classes
        self.gesture_classes = [
            'wave_right', 'wave_left', 'circle_clockwise', 'circle_counter',
            'swipe_up', 'swipe_down', 'swipe_left', 'swipe_right',
            'zoom_in', 'zoom_out', 'idle'
        ]

    def extract_landmarks(self, results):
        """Extract normalized hand landmarks"""
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            # Extract x, y, z coordinates for all 21 landmarks
            coords = []
            for landmark in landmarks.landmark:
                coords.extend([landmark.x, landmark.y, landmark.z])
            return np.array(coords)
        else:
            # Return zeros if no hand detected
            return np.zeros(self.feature_size)

    def calculate_motion_features(self, sequence):
        """Calculate motion-based features from sequence"""
        if len(sequence) < 2:
            return np.zeros(10)  # Return default features

        sequence = np.array(sequence)

        # Calculate velocities (first derivatives)
        velocities = np.diff(sequence, axis=0)

        # Calculate accelerations (second derivatives)
        accelerations = np.diff(velocities, axis=0)

        # Statistical features
        features = []

        # Velocity statistics
        if len(velocities) > 0:
            features.extend([
                np.mean(np.linalg.norm(velocities, axis=1)),  # Mean velocity magnitude
                np.std(np.linalg.norm(velocities, axis=1)),  # Velocity variation
                np.max(np.linalg.norm(velocities, axis=1)),  # Max velocity
            ])
        else:
            features.extend([0, 0, 0])

        # Acceleration statistics
        if len(accelerations) > 0:
            features.extend([
                np.mean(np.linalg.norm(accelerations, axis=1)),  # Mean acceleration
                np.std(np.linalg.norm(accelerations, axis=1)),  # Acceleration variation
            ])
        else:
            features.extend([0, 0])

        # Trajectory features
        if len(sequence) > 0:
            # Hand position variance (how much the hand moves)
            position_variance = np.var(sequence[:, :2], axis=0)  # x, y variance
            features.extend([np.mean(position_variance), np.std(position_variance)])

            # Path length (total distance traveled)
            distances = np.linalg.norm(np.diff(sequence[:, :2], axis=0), axis=1)
            path_length = np.sum(distances)
            features.append(path_length)

            # Direction changes
            if len(distances) > 1:
                direction_changes = np.sum(np.abs(np.diff(np.arctan2(
                    np.diff(sequence[:, 1]), np.diff(sequence[:, 0])))))
                features.append(direction_changes)
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0, 0])

        return np.array(features)

    def collect_training_data(self, gesture_name, duration=5):
        """Collect training data for a specific gesture"""
        cap = cv2.VideoCapture(0)

        print(f"Collecting data for gesture: {gesture_name}")
        print(f"Get ready... Starting in 3 seconds")

        # Countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Starting in {i}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(1000)

        print(f"Recording {gesture_name} for {duration} seconds...")

        sequences = []
        current_seq = []
        frame_count = 0
        max_frames = duration * 30  # Assuming 30 FPS

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Extract landmarks
            landmarks = self.extract_landmarks(results)
            current_seq.append(landmarks)

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Show recording status
            cv2.putText(frame, f"Recording: {gesture_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}/{max_frames}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Data Collection', frame)
            cv2.waitKey(1)

            # Create sequences of specified length
            if len(current_seq) == self.sequence_length:
                sequences.append(current_seq.copy())
                current_seq = current_seq[1:]  # Sliding window

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        # Add sequences to training data
        for seq in sequences:
            self.training_data.append(seq)
            self.training_labels.append(gesture_name)

        print(f"Collected {len(sequences)} sequences for {gesture_name}")

    def train_lstm_model(self):
        """Train LSTM model for dynamic gesture recognition"""
        if len(self.training_data) == 0:
            print("No training data available!")
            return

        # Prepare data
        X = np.array(self.training_data)

        # Create label encoder
        unique_labels = list(set(self.training_labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_to_idx[label] for label in self.training_labels])
        y_categorical = to_categorical(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42)

        # Build LSTM model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.feature_size)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(len(unique_labels), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train model
        print("Training LSTM model...")
        history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                            validation_data=(X_test, y_test), verbose=1)

        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"LSTM Model Accuracy: {accuracy:.4f}")

        self.lstm_model = model
        self.label_to_idx = label_to_idx
        self.idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        return history

    def train_random_forest(self):
        """Train Random Forest model using motion features"""
        if len(self.training_data) == 0:
            print("No training data available!")
            return

        # Extract motion features for each sequence
        X_features = []
        for sequence in self.training_data:
            motion_features = self.calculate_motion_features(sequence)
            X_features.append(motion_features)

        X = np.array(X_features)
        y = np.array(self.training_labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Evaluate
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Random Forest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        self.rf_model = rf

    def predict_gesture(self, sequence, model_type='lstm'):
        """Predict gesture from sequence"""
        if model_type == 'lstm' and self.lstm_model is not None:
            if len(sequence) == self.sequence_length:
                X = np.array([sequence])
                prediction = self.lstm_model.predict(X, verbose=0)
                predicted_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                return self.idx_to_label[predicted_idx], confidence

        elif model_type == 'rf' and self.rf_model is not None:
            motion_features = self.calculate_motion_features(sequence)
            prediction = self.rf_model.predict([motion_features])
            probabilities = self.rf_model.predict_proba([motion_features])
            confidence = np.max(probabilities)
            return prediction[0], confidence

        return "unknown", 0.0

    def real_time_recognition(self, model_type='lstm'):
        """Run real-time gesture recognition"""
        cap = cv2.VideoCapture(0)
        current_sequence = []

        print("Real-time Dynamic Gesture Recognition")
        print("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Extract landmarks
            landmarks = self.extract_landmarks(results)
            current_sequence.append(landmarks)

            # Maintain sequence length
            if len(current_sequence) > self.sequence_length:
                current_sequence.pop(0)

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Predict gesture
            if len(current_sequence) == self.sequence_length:
                gesture, confidence = self.predict_gesture(current_sequence, model_type)

                # Display prediction
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Dynamic Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def save_model(self, filepath):
        """Save trained models"""
        model_data = {
            'lstm_model': self.lstm_model,
            'rf_model': self.rf_model,
            'label_to_idx': getattr(self, 'label_to_idx', {}),
            'idx_to_label': getattr(self, 'idx_to_label', {}),
            'training_data': self.training_data,
            'training_labels': self.training_labels
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Models saved to {filepath}")

    def load_model(self, filepath):
        """Load trained models"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.lstm_model = model_data.get('lstm_model')
        self.rf_model = model_data.get('rf_model')
        self.label_to_idx = model_data.get('label_to_idx', {})
        self.idx_to_label = model_data.get('idx_to_label', {})
        self.training_data = model_data.get('training_data', [])
        self.training_labels = model_data.get('training_labels', [])

        print(f"Models loaded from {filepath}")


# Example usage and training pipeline
if __name__ == "__main__":
    # Initialize tracker
    tracker = DynamicGestureTracker(sequence_length=30)

    # Training phase
    print("=== Training Phase ===")
    gestures_to_collect = ['wave_right', 'wave_left', 'circle_clockwise', 'swipe_up', 'idle']

    for gesture in gestures_to_collect:
        input(f"Press Enter to start collecting data for '{gesture}'...")
        tracker.collect_training_data(gesture, duration=10)

    # Train models
    print("\n=== Training Models ===")
    tracker.train_lstm_model()
    tracker.train_random_forest()

    # Save models
    tracker.save_model('dynamic_gesture_model.pkl')

    # Real-time recognition
    print("\n=== Real-time Recognition ===")
    input("Press Enter to start real-time recognition...")
    tracker.real_time_recognition(model_type='lstm')