import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import google.generativeai as genai
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import math
import os
import io

# --- 1. Page Setup ---
st.set_page_config(page_title="NEC & EPRI DAS Agent", page_icon="🔬", layout="wide")

# --- 2. Load Our "Engine" (The AI Model) ---
# We will load our fine-tuned ViT model from Hugging Face.
# We use @st.cache_resource to load it *once* and keep it in memory.
@st.cache_resource
def load_ai_model(hf_repo_name, hf_token):
    print("Loading AI model from Hugging Face...")
    try:
        processor = ViTImageProcessor.from_pretrained(hf_repo_name, token=hf_token)
        model = ViTForImageClassification.from_pretrained(hf_repo_name, token=hf_token)
        print("Model loaded successfully.")
        return processor, model
    except Exception as e:
        st.error(f"Error loading AI model: {e}. Check your Hugging Face repo name and token.")
        return None, None

# --- 3. Load Our "Settings" (from the config.ini we created) ---
# We are hard-coding them here because the app doesn't have the config.ini
# These are the "recipe" settings we discovered.
FS = 5000.0
HP_CUTOFF = 20.0
HP_ORDER = 4
VPCT = 99.0
CHUNK_SECONDS = 0.2
DPI = 100
CHUNK_SAMPLES = int(FS * CHUNK_SECONDS) # 1000 samples

# --- 4. Our Professional Signal Processing Functions ---
# These are the same functions from our "fuel factory" script
def highpass_filter(S, fs, cutoff, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, S, axis=0)

def create_heatmap_image(S_chunk):
    # This creates the heatmap image *in memory* without saving a file
    plt.figure(figsize=(8, 6))
    r = np.percentile(np.abs(S_chunk), VPCT)
    if r == 0: r = 1.0
    plt.imshow(S_chunk.T, aspect='auto', cmap='seismic', vmin=-r, vmax=r)
    plt.axis('off')
    plt.margins(0,0)
    
    # Save the plot to a temporary in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf) # Return a PIL Image object

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
    # We will load the NEC logo from your *other* app's GitHub repo
    st.image("https://raw.githubusercontent.com/benrutgers/epri-dashboard/main/nec_logo.jpg", width=200) 
    st.subheader("Configuration")
    st.write("""
    This app uses a custom-trained AI (ViT) model to analyze .npy files 
    from the EPRI DAS interrogator and detect vandalism events.
    """)
    st.divider()
    
    # --- We need TWO secret keys now ---
    st.info("App requires two secrets to be set by the owner: 'GOOGLE_AI_API_KEY' (for Gemini) and 'HF_TOKEN' (for Hugging Face).")
    
    # Get secrets from Streamlit
    GEMINI_API_KEY = st.secrets.get("GOOGLE_AI_API_KEY")
    HF_TOKEN = st.secrets.get("HF_TOKEN")
    
    # The name of your Hugging Face model
    # IMPORTANT: Change this to your username and repo name!
    HF_REPO_NAME = "benrutgers/epri-das-classifier" # <-- Make sure this is correct!

# --- Main App Body ---
st.title("🔬 NEC & EPRI | DAS Fault Detection Agent")

# Load the AI "Eyes"
if not HF_TOKEN:
    st.error("Hugging Face token not set in Secrets. Cannot load AI model.")
    st.stop()
    
processor, model = load_ai_model(HF_REPO_NAME, HF_TOKEN)
if not model:
    st.stop()

# --- File Uploader ---
st.subheader("1. Upload a DAS Sensor File")
uploaded_file = st.file_uploader("Upload a .npy file from the DAS interrogator", type=["npy"])

if uploaded_file is not None:
    st.success(f"Successfully loaded file: {uploaded_file.name}")
    
    # --- Run the entire pipeline ---
    with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
        
        # Load the .npy file from the uploader
        S = np.load(uploaded_file).astype(np.float32)
        nt, nx = S.shape
        
        # Apply our professional filters
        S_filtered = highpass_filter(S, fs=FS, cutoff=HP_CUTOFF, order=HP_ORDER)
        S_final = S_filtered - S_filtered.mean(axis=0, keepdims=True)
        
        total_samples = S_final.shape[0]
        num_chunks = math.floor(total_samples / CHUNK_SAMPLES)
        
        st.write(f"File has {nt} time steps. Slicing into {num_chunks} 0.2-second chunks for analysis...")
        
        vandalism_count = 0
        ambient_count = 0
        results = []
        
        # This is our main "Agent" loop
        progress_bar = st.progress(0, text="Analyzing chunks...")
        
        for i in range(num_chunks):
            start_index = i * CHUNK_SAMPLES
            end_index = start_index + CHUNK_SAMPLES
            S_chunk = S_final[start_index:end_index, :]
            
            # 1. Create the heatmap image in memory
            heatmap_image = create_heatmap_image(S_chunk)
            
            # 2. Process and predict with our AI "Eyes"
            inputs = processor(images=heatmap_image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = model.config.id2label[predicted_class_idx]
            
            # 3. Store the result
            results.append(predicted_class)
            if predicted_class == "vandalism":
                vandalism_count += 1
            else:
                ambient_count += 1
            
            # Update progress bar
            progress_bar.progress((i + 1) / num_chunks, text=f"Analyzing chunk {i+1}/{num_chunks}")
        
        progress_bar.empty()
        st.success(f"Analysis Complete.")
    
    # --- Show the Results ---
    st.subheader("2. Analysis Results")
    col1, col2 = st.columns(2)
    col1.metric("Total Chunks Analyzed", f"{num_chunks}")
    if vandalism_count > 0:
        col2.metric("VANDALISM EVENTS DETECTED", f"{vandalism_count}", delta_color="inverse")
    else:
        col2.metric("VANDALISM EVENTS DETECTED", "0")
    
    # --- Generate the AI Report ---
    st.subheader("3. AI Agent Report")
    if not GEMINI_API_KEY:
        st.error("Google AI API Key not set in Secrets. Cannot generate report.")
    else:
        with st.spinner("AI 'Brain' (Gemini) is writing the report..."):
            report_text = get_ai_report(GEMINI_API_KEY, num_chunks, vandalism_count, HF_TOKEN)

            st.markdown(report_text)
