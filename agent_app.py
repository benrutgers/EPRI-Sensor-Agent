import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib
# (FIX 1A: THE CRASH)
# This tells matplotlib "You are on a server. Do NOT use a GUI."
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import butter, filtfilt
import google.generativeai as genai
from PIL import Image
import torch
import math
import os
import io
import datetime 
from fpdf import FPDF, XPos, YPos 
import gc # (!!!) THIS IS THE REAL FIX (Part 1) (!!!)

# (FIX: ROBUST IMPORT)
try:
    from transformers import ViTImageProcessor as ViTProcessor
except ImportError:
    print("ViTImageProcessor not found, falling back to ViTFeatureExtractor")
    from transformers import ViTFeatureExtractor as ViTProcessor
    
from transformers import ViTForImageClassification

# --- 1. Page Setup ---
st.set_page_config(page_title="NEC & EPRI DAS Agent", page_icon="⚡", layout="wide") 

# --- 2. Load Our "Engine" (The AI Model) ---
@st.cache_resource
def load_ai_model(hf_repo_name, hf_token):
    print("Loading AI model from Hugging Face...")
    try:
        processor = ViTProcessor.from_pretrained(hf_repo_name, token=hf_token)
        model = ViTForImageClassification.from_pretrained(hf_repo_name, token=hf_token)
        print("Model loaded successfully.")
        return processor, model
    except Exception as e:
        st.error(f"Error loading AI model: {e}. Check your Hugging Face repo name and token.")
        return None, None

# --- 3. Load Our "Settings" ---
FS = 5000.0
HP_CUTOFF = 20.0
HP_ORDER = 4
VPCT = 99.0
CHUNK_SECONDS = 0.2
DPI = 100
CHUNK_SAMPLES = int(FS * CHUNK_SECONDS)

# --- 4. Professional Signal Processing Functions ---
def highpass_filter(S, fs, cutoff, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, S, axis=0)

# (FIX 1B: THE CRASH)
# This is the robust, "memory-leak-proof" image creation function
def create_heatmap_image(S_chunk):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    r = np.percentile(np.abs(S_chunk), VPCT)
    if r == 0: r = 1.0
    vmin, vmax = -r, r
    normalized_chunk = (np.clip(S_chunk.T, vmin, vmax) - vmin) / (vmax - vmin)
    
    cmap = matplotlib.colormaps['seismic'] 
    
    rgba_image_data = cmap(normalized_chunk)
    rgb_array = (rgba_image_data[:, :, :3] * 255).astype(np.uint8)
    
    ax.imshow(rgb_array, aspect='auto')
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, bbox_inches='tight', pad_inches=0)
    
    plt.close(fig) 
    
    buf.seek(0)
    return Image.open(buf).convert("RGB") 

# --- PDF DOWNLOAD FUNCTION ---
def create_pdf_report(report_text, current_time, vandalism_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.image("nec_logo.jpg", x=10, y=8, w=60) 
    pdf.ln(25) 
    
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "NEC & EPRI Grid Operations Meteorological Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, f"Issuance Time: {current_time}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(5)
    
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, f"System: NEC LS3300 DAS System", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
    pdf.cell(0, 10, f"Alert Level: {'CRITICAL' if vandalism_count > 0 else 'NORMAL'}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
    pdf.ln(5)
    
    pdf.set_font("Helvetica", '', 11)
    pdf.multi_cell(0, 5, report_text)
    
    return bytes(pdf.output())

# --- 5. Our AI "Brain" (The Gemini LLM) Function ---
@st.cache_data
def get_ai_report(_gemini_api_key, total_chunks, vandalism_count, current_time):
    try:
        genai.configure(api_key=_gemini_api_key) 
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
        Your task is to write a *brief, 2-paragraph* fault analysis report.
        
        IMPORTANT:
        1.  Your response MUST be plain text only (no markdown, no "##", no "**").
        2.  Do NOT invent a date or system name. They will be in the PDF header.
        3.  Your response should begin *directly* with the "Detailed Analysis" paragraph.

        CONTEXT:
        - System: NEC LS3300 DAS System
        - Report Issuance Time: {current_time}
        - Total 0.2-second chunks analyzed: {total_chunks}
        - "Vandalism" chunks detected: {vandalism_count}
        - Alert Level: {alert_level}

        Write the 2-paragraph report:
        1.  **Detailed Analysis:** Explain *what* was found (or not found).
        2.  **Recommendation:** Provide a clear, actionable recommendation.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error in Gemini report: {e}")
        return f"Error generating report: {e}"

# --- 6. The Streamlit App GUI ---
st.set_page_config(page_title="NEC & EPRI DAS Agent", page_icon="⚡", layout="wide") 

# --- Sidebar ---
with st.sidebar:
    st.image("nec_logo.jpg", width=200) 
    st.subheader("Configuration")
    st.write("""
    This app uses a custom-trained AI (ViT) model to analyze .npy files 
    from the EPRI DAS interrogator and detect vandalism events.
    """)
    st.divider()
    
    st.info("App requires two secrets to be set by the owner: 'GOOGLE_AI_API_KEY' (for Gemini) and 'HF_TOKEN' (for Hugging Face).")
    
    GEMINI_API_KEY = st.secrets.get("GOOGLE_AI_API_KEY")
    HF_TOKEN = st.secrets.get("HF_TOKEN")
    
    HF_REPO_NAME = "benrutgers/epri-das-classifier" 

# --- Main App Body ---
st.title("⚡ NEC & EPRI | DAS Fault Detection Agent") 

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
        
        progress_bar = st.progress(0, text="Analyzing chunks...")
        
        for i in range(num_chunks):
            start_index = i * CHUNK_SAMPLES
            end_index = start_index + CHUNK_SAMPLES
            S_chunk = S_final[start_index:end_index, :]
            
            heatmap_image_rgb = create_heatmap_image(S_chunk)
            
            inputs = processor(images=[heatmap_image_rgb], return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item() # This was the typo
            predicted_class = model.config.id2label[predicted_class_idx]
            
            if predicted_class == "vandalism":
                vandalism_count += 1
            else:
                ambient_count += 1
            
            progress_bar.progress((i + 1) / num_chunks, text=f"Analyzing chunk {i+1}/{num_chunks}")
            
            # (!!!) THIS IS THE REAL FIX (Part 2) (!!!)
            # We are now manually "taking out the trash" after *every* loop
            # to prevent the memory leak and crash.
            del S_chunk, heatmap_image_rgb, inputs, outputs, logits
            gc.collect() # Force the "janitor" to run
        
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
            
            current_time_utc = datetime.datetime.now(datetime.timezone.utc)
            eastern_time = current_time_utc.astimezone(datetime.timezone(datetime.timedelta(hours=-5)))
            current_time_str = eastern_time.strftime("%Y-%m-%d %H:%M:%S %Z")

            report_text = get_ai_report(GEMINI_API_KEY, num_chunks, vandalism_count, current_time_str)
            
            st.text_area("Generated Report (Plain Text)", report_text, height=175)

            pdf_data = create_pdf_report(report_text, current_time_str, vandalism_count)
            st.download_button(
                label="⬇️ Download Full PDF Report",
                data=pdf_data,
                file_name="NEC_EPRI_DAS_Report.pdf",
                mime="application/pdf"
            )
