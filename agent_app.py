import streamlit as st

import numpy as np
import pandas as pd
import librosa
import matplotlib

# Use non-interactive backend for matplotlib
matplotlib.use("Agg")

from scipy.signal import butter, filtfilt
import google.generativeai as genai
from PIL import Image
import torch
import math
import os
import io
import datetime
from fpdf import FPDF, XPos, YPos
import gc  # for manual garbage collection

# Pre-create colormap once
CMAP = matplotlib.colormaps["seismic"]

# --- Robust transformers import ---
try:
    from transformers import ViTImageProcessor as ViTProcessor
except ImportError:
    print("ViTImageProcessor not found, falling back to ViTFeatureExtractor")
    from transformers import ViTFeatureExtractor as ViTProcessor

from transformers import ViTForImageClassification

# --- 1. Page Setup (call ONCE) ---
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


# --- 3. Global Settings (unchanged) ---
FS = 5000.00
HP_CUTOFF = 20.00
HP_ORDER = 4
VPCT = 99.00
CHUNK_SECONDS = 0.20
DPI = 100
CHUNK_SAMPLES = int(FS * CHUNK_SECONDS)


# --- 4. Signal Processing ---
def highpass_filter(S, fs, cutoff, order):
    nyq = 0.50 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, S, axis=0)


# --- 5. Heatmap Image Creation (no matplotlib figures) ---
def create_heatmap_image(S_chunk):
    """
    Convert a DAS chunk (time x channels) into a heatmap-like RGB image.

    IMPORTANT: This matches the original preprocessing logic:
      - Per-chunk percentile VPCT over |S_chunk|
      - Clip to [-r, r]
      - Normalize to [0, 1]
      - Apply 'seismic' colormap and convert to uint8 RGB
    """

    # 1) Percentile-based clipping PER CHUNK (same as original app)
    r = np.percentile(np.abs(S_chunk), VPCT)
    if r == 0:
        r = 1.00
    vmin, vmax = -r, r

    # 2) Normalize to [0, 1]; transpose so shape is (channels, time) -> (H, W)
    normalized_chunk = (np.clip(S_chunk.T, vmin, vmax) - vmin) / (vmax - vmin)

    # 3) Apply the 'seismic' colormap EXACTLY like original code
    rgba = CMAP(normalized_chunk)                 # float32 in [0, 1]
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8) # to uint8 RGB

    # 4) Convert to a PIL image for ViT
    img = Image.fromarray(rgb, mode="RGB")
    return img


# --- 6. Helper to sanitize text for FPDF (Latin-1 only) ---
def _sanitize_for_pdf(text: str) -> str:
    """
    Ensure the text only contains characters that the default FPDF
    Latin-1 fonts can handle. Unsupported chars are replaced with '?'.
    """
    if not isinstance(text, str):
        text = str(text)
    return text.encode("latin-1", "replace").decode("latin-1")


# --- 7. PDF DOWNLOAD FUNCTION ---
def create_pdf_report(report_text, current_time, vandalism_count):
    pdf = FPDF()
    pdf.add_page()

    # Logo
    pdf.image("nec_logo.jpg", x=10, y=8, w=60)
    pdf.ln(25)

    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(
        0,
        10,
        "NEC & EPRI Power Grid Monitoring Report",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align='C'
    )

    pdf.set_font("Helvetica", '', 12)
    pdf.cell(
        0,
        10,
        f"Issuance Time: {current_time}",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align='C'
    )
    pdf.ln(5)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(
        0,
        10,
        "System: NEC LS3300 DAS System",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align='L'
    )
    pdf.cell(
        0,
        10,
        f"Alert Level: {'CRITICAL' if vandalism_count > 0 else 'NORMAL'}",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align='L'
    )
    pdf.ln(5)

    pdf.set_font("Helvetica", '', 11)

    # Sanitize report text so FPDF doesn't choke on Unicode
    safe_text = _sanitize_for_pdf(report_text)
    pdf.multi_cell(0, 5, safe_text)

    # fpdf2 2.8.5: output(dest="S") returns a bytearray already in PDF format
    raw = pdf.output(dest="S")
    if isinstance(raw, (bytes, bytearray)):
        pdf_bytes = bytes(raw)
    else:
        # Older behavior (string): encode explicitly
        pdf_bytes = raw.encode("latin-1", "replace")

    return pdf_bytes


# --- 8. Gemini LLM report ---
@st.cache_data
def get_ai_report(_gemini_api_key, total_chunks, vandalism_count, current_time):
    try:
        genai.configure(api_key=_gemini_api_key)
        model = genai.GenerativeModel('models/gemini-flash-latest')

        if vandalism_count > 0:
            alert_level = "CRITICAL"
        else:
            alert_level = "NORMAL"

        prompt = f"""
        You are an expert NEC & EPRI DAS (Distributed Acoustic Sensing) system monitor.
        Your task is to write a brief, 2-paragraph fault analysis report.

        IMPORTANT:
        1.  Your response MUST be plain text only (no markdown, no "##", no "**").
        2.  Do NOT invent a date or system name. They will be in the PDF header.
        3.  Your response should begin directly with the "Detailed Analysis" paragraph.

        CONTEXT:
        - System: NEC LS3300 DAS System
        - Report Issuance Time: {current_time}
        - Total 0.2-second chunks analyzed: {total_chunks}
        - "Vandalism" chunks detected: {vandalism_count}
        - Alert Level: {alert_level}

        Write the 2-paragraph report:
        1. Detailed Analysis: Explain what was found (or not found).
        2. Recommendation: Provide a clear, actionable recommendation.
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error in Gemini report: {e}")
        return f"Error generating report: {e}"


# --- 9. Sidebar ---
with st.sidebar:
    st.image("nec_logo.jpg", width=200)
    st.subheader("Configuration")
    st.write(
        """
        This app uses a custom-trained AI (ViT) model to analyze .npy files 
        from the EPRI DAS interrogator and detect vandalism events.
        """
    )
    st.divider()

    st.info(
        "App requires two secrets to be set by the owner: "
        "'GOOGLE_AI_API_KEY' (for Gemini) and 'HF_TOKEN' (for Hugging Face)."
    )

    GEMINI_API_KEY = st.secrets.get("GOOGLE_AI_API_KEY")
    HF_TOKEN = st.secrets.get("HF_TOKEN")

    HF_REPO_NAME = "benrutgers/epri-das-classifier"


# --- 10. Main App Body ---
st.title("⚡ NEC & EPRI | DAS Fault Detection Agent")

if not HF_TOKEN:
    st.error("Hugging Face token not set in Secrets. Cannot load AI model.")
    st.stop()

processor, model = load_ai_model(HF_REPO_NAME, HF_TOKEN)
if not model:
    st.stop()

# --- Session state setup for caching per-file results ---
if "file_info" not in st.session_state:
    st.session_state.file_info = None
    st.session_state.num_chunks = None
    st.session_state.vandalism_count = None
    st.session_state.report_text = None
    st.session_state.pdf_data = None
    st.session_state.current_time_str = None

st.subheader("1. Upload a DAS Sensor File")

uploaded_file = st.file_uploader(
    "Upload a .npy file from the DAS interrogator",
    type=["npy"],
    key="npy_uploader",
)

# Allow clearing / resetting if user removes file
if uploaded_file is None:
    st.session_state.file_info = None
    st.session_state.num_chunks = None
    st.session_state.vandalism_count = None
    st.session_state.report_text = None
    st.session_state.pdf_data = None
    st.session_state.current_time_str = None

if uploaded_file is not None:
    current_info = (uploaded_file.name, uploaded_file.size)
    is_new_file = (st.session_state.file_info != current_info)

    if is_new_file:
        # New file uploaded -> reset results and run analysis
        st.session_state.file_info = current_info
        st.session_state.num_chunks = None
        st.session_state.vandalism_count = None
        st.session_state.report_text = None
        st.session_state.pdf_data = None
        st.session_state.current_time_str = None

        st.success(f"Successfully loaded file: {uploaded_file.name}")

        with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):

            # Load and preprocess signal
            S = np.load(uploaded_file).astype(np.float32)
            nt, nx = S.shape

            S_filtered = highpass_filter(S, fs=FS, cutoff=HP_CUTOFF, order=HP_ORDER)
            S_final = S_filtered - S_filtered.mean(axis=0, keepdims=True)

            total_samples = S_final.shape[0]
            num_chunks = math.floor(total_samples / CHUNK_SAMPLES)

            st.write(
                f"File has {nt} time steps. "
                f"Slicing into {num_chunks} 0.2-second chunks for analysis..."
            )

            vandalism_count = 0
            ambient_count = 0

            progress_bar = st.progress(0, text="Analyzing chunks...")

            for i in range(num_chunks):
                start_index = i * CHUNK_SAMPLES
                end_index = start_index + CHUNK_SAMPLES
                S_chunk = S_final[start_index:end_index, :]

                # Per-chunk heatmap image (original behavior)
                heatmap_image_rgb = create_heatmap_image(S_chunk)

                inputs = processor(images=[heatmap_image_rgb], return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                predicted_class = model.config.id2label[predicted_class_idx]

                if predicted_class == "vandalism":
                    vandalism_count += 1
                else:
                    ambient_count += 1

                # Free per-chunk intermediates ASAP
                del inputs, outputs, logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                progress_bar.progress(
                    (i + 1) / num_chunks,
                    text=f"Analyzing chunk {i+1}/{num_chunks}",
                )

            progress_bar.empty()
            st.success("Analysis Complete.")

            # Save summary stats to session_state
            st.session_state.num_chunks = num_chunks
            st.session_state.vandalism_count = vandalism_count

            # Clean up big arrays
            try:
                del S, S_filtered, S_final
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Show analysis results if we have them ---
    if st.session_state.num_chunks is not None:
        st.subheader("2. Analysis Results")
        col1, col2 = st.columns(2)
        col1.metric("Total Chunks Analyzed", f"{st.session_state.num_chunks}")
        if st.session_state.vandalism_count and st.session_state.vandalism_count > 0:
            col2.metric(
                "VANDALISM EVENTS DETECTED",
                f"{st.session_state.vandalism_count}",
                delta_color="inverse",
            )
        else:
            col2.metric("VANDALISM EVENTS DETECTED", "0")

        # --- 3. AI Agent Report ---
        st.subheader("3. AI Agent Report")

        if not GEMINI_API_KEY:
            st.error("Google AI API Key not set in Secrets. Cannot generate report.")
        else:
            # Only call Gemini + build PDF once per file
            if st.session_state.report_text is None or st.session_state.pdf_data is None:
                with st.spinner("AI 'Brain' (Gemini) is writing the report..."):
                    current_time_utc = datetime.datetime.now(datetime.timezone.utc)
                    eastern_time = current_time_utc.astimezone(
                        datetime.timezone(datetime.timedelta(hours=-5))
                    )
                    current_time_str = eastern_time.strftime("%Y-%m-%d %H:%M:%S %Z")

                    report_text = get_ai_report(
                        GEMINI_API_KEY,
                        st.session_state.num_chunks,
                        st.session_state.vandalism_count,
                        current_time_str,
                    )

                    pdf_data = create_pdf_report(
                        report_text,
                        current_time_str,
                        st.session_state.vandalism_count,
                    )

                    st.session_state.current_time_str = current_time_str
                    st.session_state.report_text = report_text
                    st.session_state.pdf_data = pdf_data

            # Display cached report + download button
            st.text_area(
                "Generated Report (Plain Text)",
                st.session_state.report_text or "",
                height=200,
            )

            if st.session_state.pdf_data is not None:
                st.download_button(
                    label="⬇️ Download Full PDF Report",
                    data=st.session_state.pdf_data,
                    file_name="NEC_EPRI_DAS_Report.pdf",
                    mime="application/pdf",
                )
            else:
                st.warning("PDF data is not available.")
