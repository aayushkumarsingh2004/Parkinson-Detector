
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
    st.write("Upload a WAV file.")
    uploaded_file = st.file_uploader("Upload Audio (WAV)", type=['wav'])
    
    if uploaded_file and st.button("Analyze Voice"):
        if voice_model:
            try:
                with open("temp.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                y, sr = librosa.load("temp.wav")
                y, _ = librosa.effects.trim(y, top_db=20)
                mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
                
                # Simple jitter calc
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0: pitch_values.append(pitch)
                
                jitter = np.mean(np.abs(np.diff(pitch_values))) / np.mean(pitch_values) if len(pitch_values) > 0 else 0
                
                features = [jitter] + list(mfccs)
                pred = voice_model.predict([features])[0]
                prob = voice_model.predict_proba([features])[0][1]
                
                st.markdown("---")
                if pred == 1:
                    st.error(f"🔴 Result: High Risk of Parkinson's")
                    st.write(f"**Confidence Score:** {prob*100:.1f}%")
                else:
                    st.success(f"🟢 Result: Healthy")
                    st.write(f"**Confidence Score:** {(1-prob)*100:.1f}%")
                    
            except Exception as e:
                st.error(f"Error: {e}")

# === GAIT SECTION (FIXED) ===
with tab2:
    st.header("Gait Analysis")
    st.write("Upload 'Accelerometer.csv' from Sensor Logger.")
    uploaded_csv = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_csv and st.button("Analyze Gait"):
        if gait_model:
            try:
                df = pd.read_csv(uploaded_csv)
                
                # Find Column
                signal = None
                possible_cols = ['y', 'Y', 'Ankle_Y', 'ay', 'Ay', 'acceleration_y']
                for col in possible_cols:
                    if col in df.columns:
                        signal = df[col].values
                        break
                
                if signal is None:
                     signal = df.iloc[:, 2].values 
                
                # SCALING FIX
                if np.max(np.abs(signal)) < 20:
                    signal = signal * 1000 
                
                feat_std = np.std(signal)
                feat_range = np.max(signal) - np.min(signal)
                feat_jerk = np.mean(np.abs(np.diff(signal)))
                
                pred = gait_model.predict([[feat_std, feat_range, feat_jerk]])[0]
                prob = gait_model.predict_proba([[feat_std, feat_range, feat_jerk]])[0][1]
                
                st.markdown("---")
                # HERE IS THE FIX: Added the colon (:) at the end
                if pred == 1:
                    st.error(f"🔴 Result: High Risk Gait Pattern")
                    st.write(f"**Confidence Score:** {prob*100:.1f}%")
                    st.warning("Irregularities detected.")
                else:
                    st.success(f"🟢 Result: Healthy Gait Pattern")
                    st.write(f"**Confidence Score:** {(1-prob)*100:.1f}%")
                    st.info("Movement is stable.")
                    
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
