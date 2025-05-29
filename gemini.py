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
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)


# Load the model
@st.cache_resource
def load_asl_model():
    return load_model('ASLmodelF.h5')


model = load_asl_model()

# ASL classes
ASL_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]


def extract_hand_features(hand_landmarks):
    """Convert hand landmarks to 42D feature vector"""
    return np.array([coord for landmark in hand_landmarks.landmark
                     for coord in [landmark.x, landmark.y]])[:42]


def text_to_speech(text, filename='output.mp3'):
    """Convert text to speech using gTTS"""
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename


# Initialize session state for word building
if 'current_word' not in st.session_state:
    st.session_state.current_word = []
if 'word_history' not in st.session_state:
    st.session_state.word_history = []

# Create Streamlit columns
col1, col2, col3 = st.columns([2, 1, 1])

# Add controls to sidebar
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.05, key='conf_threshold')
    show_landmarks = st.checkbox("Show Hand Landmarks", value=True, key='show_landmarks')
    show_confidence = st.checkbox("Show Confidence", value=True, key='show_confidence')
    hold_duration = st.slider("Letter Hold Duration (frames)", 1, 30, 10, 1, key='hold_duration')

    st.header("Word Controls")
    add_space = st.button("Add Space", key='add_space')
    clear_word = st.button("Clear Word", key='clear_word')
    speak_word = st.button("Speak Word", key='speak_word')
    stop_button = st.button("Stop Detection", key='stop_button')

# Initialize video capture
cap = cv2.VideoCapture(0)
frame_placeholder = col1.empty()
results_placeholder = col2.empty()
word_placeholder = col3.empty()

# Detection variables
last_letter = None
letter_counter = 0

# Main detection loop
while True:
    ret, frame = cap.read()
    if not ret or stop_button:
        if not ret:
            st.error("Failed to capture video")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    prediction_text = "No hand detected"
    confidence_text = ""
    prediction = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if show_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
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
                        letter_counter = 0
                else:
                    color = (0, 0, 255)  # Red
                    prediction_text += " (Low Confidence)"

                cv2.putText(frame, prediction_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                if show_confidence:
                    cv2.putText(frame, confidence_text, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Handle word controls
    if add_space:
        st.session_state.current_word.append(' ')
        st.session_state.word_history.append(''.join(st.session_state.current_word))

    if clear_word:
        st.session_state.current_word = []
        st.session_state.word_history.append('-- Word Cleared --')

    if speak_word and st.session_state.current_word:
        word = ''.join(st.session_state.current_word)
        audio_file = text_to_speech(word)
        st.audio(audio_file, format='audio/mp3')
        os.remove(audio_file)  # Clean up temporary file

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

# Release resources
cap.release()
hands.close()