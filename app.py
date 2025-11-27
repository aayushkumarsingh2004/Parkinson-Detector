
import streamlit as st
import joblib
import numpy as np
import librosa
import pandas as pd
import os

# Set page title
st.set_page_config(page_title="Parkinson's Detector", page_icon="🧠")

st.title("🧠 Parkinson's Detection System")
st.write("DTU B.Tech Project - Group 86")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # These filenames must match exactly what you upload to GitHub
    voice_path = 'voice_model_rf.pkl'
    gait_path = 'gait_model_rf.pkl'
    
    try:
        v_model = joblib.load(voice_path)
        g_model = joblib.load(gait_path)
        return v_model, g_model
    except:
        return None, None

voice_model, gait_model = load_models()

if voice_model and gait_model:
    st.success("✅ Models Loaded Successfully")
else:
    st.warning("⚠️ Models not found. Please ensure .pkl files are in the GitHub repository.")

# --- TABS ---
tab1, tab2 = st.tabs(["🎤 Voice Test", "🦵 Gait Test"])

# === VOICE SECTION ===
with tab1:
    st.header("Voice Analysis")
    st.write("Upload a WAV file of the patient saying 'Ahhh' or reading a sentence.")
    uploaded_file = st.file_uploader("Upload Audio (WAV)", type=['wav'])
    
    if uploaded_file and st.button("Analyze Voice"):
        if voice_model:
            try:
                # Save temp file
                with open("temp.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process Audio
                y, sr = librosa.load("temp.wav")
                y, _ = librosa.effects.trim(y, top_db=20)
                
                # Extract Features
                mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0: pitch_values.append(pitch)
                
                jitter = np.mean(np.abs(np.diff(pitch_values))) / np.mean(pitch_values) if len(pitch_values) > 0 else 0
                
                # Predict
                features = [jitter] + list(mfccs)
                pred = voice_model.predict([features])[0]
                prob = voice_model.predict_proba([features])[0][1] # Probability of PD
                
                st.markdown("---")
                if pred == 1:
                    st.error(f"🔴 Result: High Risk of Parkinson's")
                    st.write(f"**Confidence Score:** {prob*100:.1f}%")
                else:
                    st.success(f"🟢 Result: Healthy")
                    st.write(f"**Confidence Score:** {(1-prob)*100:.1f}%")
                    
            except Exception as e:
                st.error(f"Error processing audio: {e}")

# === GAIT SECTION (UPDATED WITH CONFIDENCE SCORE) ===
with tab2:
    st.header("Gait Analysis")
    st.write("Upload the 'Accelerometer.csv' file from the Sensor Logger App.")
    uploaded_csv = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_csv and st.button("Analyze Gait"):
        if gait_model:
            try:
                # Load CSV
                df = pd.read_csv(uploaded_csv)
                
                # Intelligent Column Search
                signal = None
                possible_cols = ['y', 'Y', 'Ankle_Y', 'ay', 'Ay', 'acceleration_y']
                for col in possible_cols:
                    if col in df.columns:
                        signal = df[col].values
                        break
                
                if signal is None:
                     # Fallback: Assume it's the 3rd column (Index 2)
                     signal = df.iloc[:, 2].values 
                
                # Extract Features
                feat_std = np.std(signal)
                feat_range = np.max(signal) - np.min(signal)
                feat_jerk = np.mean(np.abs(np.diff(signal)))
                
                # Predict
                features = [[feat_std, feat_range, feat_jerk]]
                pred = gait_model.predict(features)[0]
                
                # THIS IS THE NEW PART (Probability Calculation)
                prob = gait_model.predict_proba(features)[0][1] 
                
                st.markdown("---")
                if pred == 1:
                    st.error(f"🔴 Result: High Risk Gait Pattern")
                    st.write(f"**Confidence Score:** {prob*100:.1f}%")
                    st.warning("Irregularities detected in step force and rhythm.")
                else:
                    st.success(f"🟢 Result: Healthy Gait Pattern")
                    st.write(f"**Confidence Score:** {(1-prob)*100:.1f}%")
                    st.info("Movement is rhythmic and stable.")
                    
            except Exception as e:
                st.error(f"Error reading CSV. Make sure it is the raw 'Accelerometer.csv' file. Details: {e}")
