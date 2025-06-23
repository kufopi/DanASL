import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from gtts import gTTS
import os
from PIL import Image
import threading
import tempfile
import time

# Set up Streamlit config
st.set_page_config(page_title="ASL Word Spell & Speak", layout="wide")

st.title("ASL Word Spelling with Text-to-Speech")
ASL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['del', 'nothing', 'space']

# Load model once
@st.cache_resource
def load_asl_model():
    return load_model('ASLmodelF.h5')

model = load_asl_model()

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Session state setup
if 'current_word' not in st.session_state:
    st.session_state.current_word = []
    st.session_state.word_history = []
    st.session_state.audio_file = None
    st.session_state.detecting = False
    st.session_state.last_spoken = None

# Sidebar UI
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.05)
    hold_duration = st.slider("Letter Hold Duration", 1, 30, 10)
    show_landmarks = st.checkbox("Show Hand Landmarks", True)
    show_confidence = st.checkbox("Show Confidence", True)

    st.divider()
    st.header("Word Controls")
    if st.button("Add Space"):
        st.session_state.current_word.append(" ")
    if st.button("Clear Word"):
        st.session_state.current_word = []
    if st.button("Speak Word") and st.session_state.current_word:
        word = ''.join(st.session_state.current_word)
        if word != st.session_state.last_spoken:
            tts = gTTS(text=word, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                tts.save(f.name)
                st.session_state.audio_file = f.name
            st.session_state.last_spoken = word

# Video display column
frame_slot = st.empty()
result_slot = st.empty()
word_slot = st.empty()
audio_slot = st.empty()

def extract_features(hand_landmarks):
    return np.array([coord for lm in hand_landmarks.landmark for coord in [lm.x, lm.y]])[:42]

def run_detection():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    last_letter = None
    counter = 0

    while st.session_state.detecting:
        ret, frame = cap.read()
        if not ret:
            frame_slot.error("Failed to capture video.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        letter, confidence = "No Hand", 0.0

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                if show_landmarks:
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                features = extract_features(hand)
                if features.size == 42:
                    pred = model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
                    idx = np.argmax(pred)
                    confidence = pred[idx]
                    letter = ASL_CLASSES[idx]

                    if confidence > confidence_threshold:
                        if letter == last_letter:
                            counter += 1
                        else:
                            last_letter = letter
                            counter = 0

                        if counter >= hold_duration:
                            if letter == 'del' and st.session_state.current_word:
                                st.session_state.current_word.pop()
                            elif letter == 'space':
                                st.session_state.current_word.append(' ')
                            elif letter not in ['del', 'nothing']:
                                st.session_state.current_word.append(letter)
                            st.session_state.word_history.append(''.join(st.session_state.current_word))
                            counter = 0

        # Visual Feedback
        cv2.putText(frame, f"{letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        if show_confidence:
            cv2.putText(frame, f"{confidence:.1%}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame_slot.image(frame, channels="BGR", use_column_width=True)

        results_text = f"""
        ### Current Detection:
        - Letter: **{letter}**
        - Confidence: **{confidence:.1%}**
        - Threshold: **{confidence_threshold:.1%}**
        """
        result_slot.markdown(results_text)

        word_display = f"""### Current Word:  
        **{''.join(st.session_state.current_word) or '[Empty]'}**  
        """
        if st.session_state.word_history:
            word_display += "#### History:\n" + '\n'.join(f"- {w}" for w in st.session_state.word_history[-5:])
        word_slot.markdown(word_display)

        time.sleep(0.05)

    cap.release()
    hands.close()

# Start/Stop Detection
if not st.session_state.detecting:
    if st.button("Start Detection"):
        st.session_state.detecting = True
        threading.Thread(target=run_detection, daemon=True).start()
else:
    if st.button("Stop Detection"):
        st.session_state.detecting = False

# Audio Playback
if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
    with open(st.session_state.audio_file, 'rb') as f:
        audio_bytes = f.read()
    audio_slot.audio(audio_bytes, format='audio/mp3')
    os.remove(st.session_state.audio_file)
    st.session_state.audio_file = None
