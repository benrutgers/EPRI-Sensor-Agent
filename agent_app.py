import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt # We *still* need this for the colormap
from matplotlib import cm # NEW: Import the colormap library
from scipy.signal import butter, filtfilt
import google.generativeai as genai
from PIL import Image
import torch
import math
import os
import io

# --- (!!!) FIX 1: ROBUST IMPORT (!!!) ---
# This makes our app "version-proof"
try:
    from transformers import ViTImageProcessor as ViTProcessor
except ImportError:
    print("ViTImageProcessor not found, falling back to ViTFeatureExtractor")
    from transformers import ViTFeatureExtractor as ViTProcessor
    
from transformers import ViTForImageClassification

# --- 1. Page Setup ---
st.set_page_config(page_title="NEC & EPRI DAS Agent", page_icon="🔬", layout="wide")

# --- 2. Load Our "Engine" (The AI Model) ---
@st.cache_resource
def load_ai_model(hf_repo_name, hf_token):
    print("Loading AI model from Hugging Face...")
    try:
        # We now use our "version-proof" class 'ViTProcessor'
        processor = ViTProcessor.from_pretrained(hf_repo_name, token=hf_token)
        model = ViTForImageClassification.from_pretrained(hf_repo_name, token=hf_token)
        print("Model loaded successfully.")
        return processor, model
    except Exception as e:
        st.error(f"Error loading AI model: {e}. Check your Hugging Face repo name and token.")
        return None, None

# --- 3. Load Our "Settings" (from the config.ini we created) ---
FS = 5000.0
HP_CUTOFF = 20.0
HP_ORDER = 4
VPCT = 99.0
CHUNK_SECONDS = 0.2
DPI = 100
CHUNK_SAMPLES = int(FS * CHUNK_SECONDS) # 1000 samples

# --- 4. Our Professional Signal Processing Functions ---
def highpass_filter(S, fs, cutoff, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, S, axis=0)

# --- (!!!) FIX 2: PROFESSIONAL IMAGE CREATION (!!!) ---
# We are no longer using plt.savefig(). This is a direct,
# data-driven conversion from a NumPy array to an RGB image.
def create_heatmap_image(S_chunk):
    # 1. Calculate the robust color limits (from your script)
    r = np.percentile(np.abs(S_chunk), VPCT)
    if r == 0: r = 1.0
    vmin = -r
    vmax = r

    # 2. Normalize the chunk from -r to +r into a 0-1 scale
    # np.clip ensures no values are outside this range
    normalized_chunk = (np.clip(S_chunk, vmin, vmax) - vmin) / (vmax - vmin)

    # 3. Apply the 'seismic' colormap *directly* to the data
    # This converts our (1000, 800) array to a (1000, 800, 4) RGBA array
    cmap = cm.get_cmap('seismic')
    rgba_image = cmap(normalized_chunk)
    
    # 4. Convert to a standard 8-bit (0-255) RGB image
    # We multiply by 255 and take the first 3 channels (RGB),
    # discarding the 4th (Alpha).
    rgb_array = (rgba_image[:, :, :3] * 255).astype(np.uint8)
    
    # 5. Convert this *perfect* RGB array into a PIL Image
    return Image.fromarray(rgb_array)

# --- 5. Our AI "Brain" (The Gemini LLM) Function ---
@st.cache_data
def get_ai_report(_gemini_api_key, total_chunks, vandalism_count, hf_token):
    try:
        genai.configure(api_key=_gemini_api_key) # Use the passed-in key
        model = genai.GenerativeModel('models/gemini-flash-latest')
        
        if vandalism_count > 0:
            alert_level = "CRITICAL"
            analysis = f"VANDALISM DETECTED. The AI model identified {vandalism_count} distinct 0.2-second chunks that match the 'vandalism' signature."
            recommendation = "This is a high-confidence detection. Recommend immediate visual inspection of the fiber line for security breach or damage."
        else:
            alert_level = "NORMAL"
            analysis = f"Analysis complete. All {total_chunks} 0.2-second chunks match the 'ambient' signature."
            recommendation = "No anomalies detected. The line is operating under normal conditions."

        prompt = f"""
        You are an expert NEC & EPRI DAS (Distributed Acoustic Sensing) system monitor.
        Your task is to write a brief, professional fault analysis report.

        FILE ANALYSIS:
        - Total 0.2-second chunks analyzed: {total_chunks}
        - "Vandalism" chunks detected: {vandalism_count}
        - Alert Level: {alert_level}

        Write a 3-paragraph report:
        1.  **Executive Summary:** State the Alert Level and the overall finding.
        2.  **Detailed Analysis:** Explain *what* was found (or not found).
        3.  **Recommendation:** Provide a clear, actionable recommendation for the grid operator.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error in Gemini report: {e}")
        return f"Error generating report: {e}"

# --- 6. The Streamlit App GUI ---
st.set_page_config(page_title="NEC & EPRI DAS Agent", page_icon="🔬", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/benrutgers/epri-dashboard/main/nec_logo.jpg", width=200) 
    st.subheader("Configuration")
    st.write("""
    This app uses a custom-trained AI (ViT) model to analyze .npy files 
    from the EPRI DAS interrogator and detect vandalism events.
    """)
    st.divider()
    
    st.info("App requires two secrets to be set by the owner: 'GOOGLE_AI_API_KEY' (for Gemini) and 'HF_TOKEN' (for Hugging Face).")
    
    GEMINI_API_KEY = st.secrets.get("GOOGLE_AI_API_KEY")
    HF_TOKEN = st.secrets.get("HF_TOKEN")
    
    # IMPORTANT: Make sure this is your correct repo name
    HF_REPO_NAME = "benrutgers/epri-das-classifier" 

# --- Main App Body ---
st.title("🔬 NEC & EPRI | DAS Fault Detection Agent")

if not HF_TOKEN:
    st.error("Hugging Face token not set in Secrets. Cannot load AI model.")
    st.stop()
    
processor, model = load_ai_model(HF_REPO_NAME, HF_TOKEN)
if not model:
    st.stop()

st.subheader("1. Upload a DAS Sensor File")
uploaded_file = st.file_uploader("Upload a .npy file from the DAS interrogator", type=["npy"])

if uploaded_file is not None:
    st.success(f"Successfully loaded file: {uploaded_file.name}")
    
    with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
        
        S = np.load(uploaded_file).astype(np.float32)
        nt, nx = S.shape
        
        S_filtered = highpass_filter(S, fs=FS, cutoff=HP_CUTOFF, order=HP_ORDER)
        S_final = S_filtered - S_filtered.mean(axis=0, keepdims=True)
        
        total_samples = S_final.shape[0]
        num_chunks = math.floor(total_samples / CHUNK_SAMPLES)
        
        st.write(f"File has {nt} time steps. Slicing into {num_chunks} 0.2-second chunks for analysis...")
        
        vandalism_count = 0
        ambient_count = 0
        results = []
        
        progress_bar = st.progress(0, text="Analyzing chunks...")
        
        for i in range(num_chunks):
            start_index = i * CHUNK_SAMPLES
            end_index = start_index + CHUNK_SAMPLES
            S_chunk = S_final[start_index:end_index, :]
            
            # 1. Create the heatmap image *in memory* (NEW, ROBUST METHOD)
            heatmap_image_rgb = create_heatmap_image(S_chunk)
            
            # 2. Process and predict (passing a *list* of one)
            inputs = processor(images=[heatmap_image_rgb], return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = model.config.id2label[predicted_class_idx]
            
            results.append(predicted_class)
            if predicted_class == "vandalism":
                vandalism_count += 1
            else:
                ambient_count += 1
            
            progress_bar.progress((i + 1) / num_chunks, text=f"Analyzing chunk {i+1}/{num_chunks}")
        
        progress_bar.empty()
        st.success(f"Analysis Complete.")
    
    st.subheader("2. Analysis Results")
    col1, col2 = st.columns(2)
    col1.metric("Total Chunks Analyzed", f"{num_chunks}")
    if vandalism_count > 0:
        col2.metric("VANDALISM EVENTS DETECTED", f"{vandalism_count}", delta_color="inverse")
    else:
        col2.metric("VANDALISM EVENTS DETECTED", "0")
    
    st.subheader("3. AI Agent Report")
    if not GEMINI_API_KEY:
        st.error("Google AI API Key not set in Secrets. Cannot generate report.")
    else:
        with st.spinner("AI 'Brain' (Gemini) is writing the report..."):
            report_text = get_ai_report(GEMINI_API_KEY, num_chunks, vandalism_count, HF_TOKEN)
            st.markdown(report_text)
