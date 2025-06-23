import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from gtts import gTTS
import os
from PIL import Image
from io import BytesIO

# Set up Streamlit page
st.set_page_config(page_title="ASL Word Spell & Speak", layout="wide")
st.title("ASL Word Spelling with Text-to-Speech")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ASL classes
ASL_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]


# Initialize session state
def initialize_session_state():
    session_defaults = {
        'current_word': [],
        'word_history': [],
        'stop_detection': False,
        'audio_file': None,
        'speaking': False,
        'last_spoken_word': None
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# Load the model
@st.cache_resource
def load_asl_model():
    return load_model('ASLmodelF.h5')


model = load_asl_model()


def extract_hand_features(hand_landmarks):
    """Convert hand landmarks to 42D feature vector"""
    return np.array([coord for landmark in hand_landmarks.landmark
                     for coord in [landmark.x, landmark.y]])[:42]


def text_to_speech(text, filename='output.mp3'):
    """Convert text to speech using gTTS"""
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename


# Sidebar controls
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.05)
    show_landmarks = st.checkbox("Show Hand Landmarks", value=True)
    show_confidence = st.checkbox("Show Confidence", value=True)
    hold_duration = st.slider("Letter Hold Duration (frames)", 1, 30, 10, 1)

    st.header("Word Controls")
    add_space = st.button("Add Space")
    clear_word = st.button("Clear Word")
    speak_word = st.button("Speak Word")
    stop_button = st.button("Stop Detection")

# Handle button actions
if add_space:
    st.session_state.current_word.append(' ')
    st.session_state.word_history.append(''.join(st.session_state.current_word))

if clear_word:
    st.session_state.current_word = []
    st.session_state.word_history.append('-- Word Cleared --')

if speak_word and st.session_state.current_word:
    current_word = ''.join(st.session_state.current_word)
    if current_word != st.session_state.last_spoken_word:
        if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
            os.remove(st.session_state.audio_file)
        st.session_state.audio_file = text_to_speech(current_word)
        st.session_state.last_spoken_word = current_word
        st.session_state.speaking = True

if stop_button:
    st.session_state.stop_detection = True

# Create layout columns
col1, col2, col3 = st.columns([2, 1, 1])

# Initialize video capture
cap = cv2.VideoCapture(0)
frame_placeholder = col1.empty()
results_placeholder = col2.empty()
word_placeholder = col3.empty()

# Detection variables
last_letter = None
letter_counter = 0

# Main detection loop
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

while cap.isOpened() and not st.session_state.stop_detection:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    prediction_text = "No hand detected"
    confidence_text = ""
    prediction = None
    confidence = 0

    if results.multi_hand_landmarks and not st.session_state.speaking:
        for hand_landmarks in results.multi_hand_landmarks:
            if show_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_hand_features(hand_landmarks)

            if features.size == 42:
                input_data = np.expand_dims(features, axis=0)
                prediction = model.predict(input_data, verbose=0)[0]
                pred_class = np.argmax(prediction)
                confidence = prediction[pred_class]

                current_letter = ASL_CLASSES[pred_class]
                prediction_text = current_letter
                confidence_text = f"{confidence:.1%}"

                if confidence > confidence_threshold:
                    color = (0, 255, 0)  # Green

                    # Letter hold duration logic
                    if current_letter == last_letter:
                        letter_counter += 1
                    else:
                        last_letter = current_letter
                        letter_counter = 0

                    # Add letter after hold duration
                    if letter_counter == hold_duration:
                        if current_letter not in ['nothing', 'del', 'space']:
                            st.session_state.current_word.append(current_letter)
                            st.session_state.word_history.append(''.join(st.session_state.current_word))
                        elif current_letter == 'del' and st.session_state.current_word:
                            st.session_state.current_word.pop()
                            st.session_state.word_history.append(''.join(st.session_state.current_word))
                        elif current_letter == 'space':
                            st.session_state.current_word.append(' ')
                            st.session_state.word_history.append(''.join(st.session_state.current_word))
                        letter_counter = 0
                else:
                    color = (0, 0, 255)  # Red
                    prediction_text += " (Low Confidence)"

                cv2.putText(frame, prediction_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                if show_confidence:
                    cv2.putText(frame, confidence_text, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display video feed
    frame_placeholder.image(frame, channels="BGR", use_column_width=True)

    # Display results
    if prediction is not None:
        results_text = f"""
        ## Current Detection
        - **Letter**: {prediction_text}
        - **Confidence**: {confidence_text}
        - **Threshold**: {confidence_threshold:.0%}
        - **Hold Counter**: {letter_counter}/{hold_duration}
        """
        if confidence > confidence_threshold:
            results_placeholder.success(results_text)
        else:
            results_placeholder.warning(results_text)
    else:
        results_placeholder.info("## Current Detection\n" + prediction_text)

    # Display word building
    current_word_display = ''.join(
        st.session_state.current_word) if st.session_state.current_word else "[No letters yet]"
    word_text = f"""
    ## Current Word
    **{current_word_display}**

    ### Word History
    """
    for word in st.session_state.word_history[-5:][::-1]:  # Show last 5 entries
        word_text += f"- {word}\n"

    word_placeholder.markdown(word_text)

    # Handle audio playback
    if st.session_state.speaking and st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
        with open(st.session_state.audio_file, 'rb') as f:
            audio_bytes = f.read()
        col3.audio(audio_bytes, format='audio/mp3')
        os.remove(st.session_state.audio_file)
        st.session_state.audio_file = None
        st.session_state.speaking = False

# Release resources
cap.release()
hands.close()

# Clean up any remaining audio files
if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
    os.remove(st.session_state.audio_file)