
import streamlit as st
import joblib
import numpy as np
import librosa
import os

# Set page title
st.set_page_config(page_title="Parkinson's Detector", page_icon="🧠")

st.title("🧠 Parkinson's Detection System")
st.write("DTU B.Tech Project - Group 86")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Attempt to load local models (if uploaded to cloud)
    voice_path = 'voice_model_rf.pkl'
    gait_path = 'gait_model_rf.pkl'
    
    try:
        v_model = joblib.load(voice_path)
        g_model = joblib.load(gait_path)
        return v_model, g_model
    except:
        return None, None

voice_model, gait_model = load_models()

if voice_model:
    st.success("✅ Models Loaded Successfully")
else:
    st.warning("⚠️ Models not found. Please upload .pkl files to the app directory.")

# --- TABS ---
tab1, tab2 = st.tabs(["🎤 Voice Test", "🦵 Gait Test"])

with tab1:
    st.header("Voice Analysis")
    uploaded_file = st.file_uploader("Upload Audio (WAV)", type=['wav'])
    
    if uploaded_file and st.button("Analyze Voice"):
        if voice_model:
            try:
                # 1. Save temp file
                with open("temp.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. Process
                y, sr = librosa.load("temp.wav")
                y, _ = librosa.effects.trim(y, top_db=20)
                mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                # (Simple jitter calc for demo)
                jitter = 0.005 # Placeholder for fast cloud demo
                
                features = [jitter] + list(mfccs)
                pred = voice_model.predict([features])[0]
                prob = voice_model.predict_proba([features])[0][1]
                
                if pred == 1:
                    st.error(f"🔴 Result: High Risk ({prob*100:.1f}%)")
                else:
                    st.success(f"🟢 Result: Healthy ({ (1-prob)*100:.1f}%)")
            except Exception as e:
                st.error(f"Error processing audio: {e}")
        else:
            st.error("Model not loaded.")

with tab2:
    st.header("Gait Analysis")
    uploaded_csv = st.file_uploader("Upload Sensor Logger CSV", type=['csv'])
    
    if uploaded_csv and st.button("Analyze Gait"):
        if gait_model:
            try:
                import pandas as pd
                df = pd.read_csv(uploaded_csv)
                # Simple logic to find 'y' axis (usually index 2 or 3)
                # We assume column 'y' exists or we take the 3rd column
                if 'y' in df.columns:
                    signal = df['y'].values
                else:
                    signal = df.iloc[:, 2].values
                
                # Extract Features
                feat_std = np.std(signal)
                feat_range = np.max(signal) - np.min(signal)
                feat_jerk = np.mean(np.abs(np.diff(signal)))
                
                pred = gait_model.predict([[feat_std, feat_range, feat_jerk]])[0]
                
                if pred == 1:
                    st.error("🔴 Result: High Risk Gait Detected")
                else:
                    st.success("🟢 Result: Healthy Gait Pattern")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

