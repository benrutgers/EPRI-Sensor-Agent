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

# ---------- Global CSS for a more polished / better header UI ----------
CUSTOM_CSS = """
<style>
/* Layout of the main content */
.block-container {
    padding-top: 0.75rem;
    padding-bottom: 2.0rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Top header styling: full-width bar across page with breathing room */
.nec-header {
    margin-top: 0.75rem;          /* space above the banner */
    margin-left: -3rem;           /* cancel container padding so bar spans full width */
    margin-right: -3rem;
    margin-bottom: 1.0rem;

    padding-top: 1.15rem;         /* more vertical padding */
    padding-bottom: 0.95rem;
    padding-left: 3.0rem;
    padding-right: 3.0rem;

    background: linear-gradient(90deg, #001f3f, #003366);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);

    /* Rounded on all corners so it feels like a card, not clipped */
    border-radius: 0.75rem;
}

.nec-header-left {
    display: flex;
    flex-direction: column;
    max-width: 100%;
}

/* Title and subtitle now fit comfortably inside the taller bar */
.nec-header-title {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    line-height: 1.35;
    margin-bottom: 0.2rem;
    word-wrap: break-word;
}
.nec-header-subtitle {
    font-size: 0.85rem;
    opacity: 0.90;
}

/* Section card */
.section-card {
    border-radius: 0.75rem;
    padding: 1.25rem 1.5rem;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 6px rgba(15,23,42,0.06);
    margin-bottom: 1.0rem;
}

/* Metrics styles */
.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    color: #6b7280;
    letter-spacing: 0.06em;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #111827;
}

/* Alert badge */
.alert-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-left: 0.5rem;
}
.alert-badge-critical {
    background-color: #fee2e2;
    color: #b91c1c;
}
.alert-badge-normal {
    background-color: #dcfce7;
    color: #166534;
}

/* Small help text */
.helper-text {
    font-size: 0.80rem;
    color: #6b7280;
}

/* Text area for report */
textarea {
    font-size: 0.82rem !important;
    line-height: 1.4 !important;
}

/* Responsive tweaks for smaller screens */
@media (max-width: 900px) {
    .block-container {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    .nec-header {
        margin-left: -1.5rem;
        margin-right: -1.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        padding-top: 1.0rem;
        padding-bottom: 0.9rem;
    }
    .nec-header-title {
        font-size: 1.0rem;
    }
    .nec-header-subtitle {
        font-size: 0.75rem;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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
    Detection behavior is unchanged: per-chunk percentile clipping + 'seismic' colormap.
    """
    r = np.percentile(np.abs(S_chunk), VPCT)
    if r == 0:
        r = 1.00
    vmin, vmax = -r, r

    normalized_chunk = (np.clip(S_chunk.T, vmin, vmax) - vmin) / (vmax - vmin)
    rgba = CMAP(normalized_chunk)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    return img

# --- 6. Helper to sanitize text for FPDF (Latin-1 only) ---
def _sanitize_for_pdf(text: str) -> str:
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
    safe_text = _sanitize_for_pdf(report_text)
    pdf.multi_cell(0, 5, safe_text)

    raw = pdf.output(dest="S")
    if isinstance(raw, (bytes, bytearray)):
        pdf_bytes = bytes(raw)
    else:
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
def render_sidebar():
    with st.sidebar:
        st.image("nec_logo.jpg", width=180)
        st.markdown("### Configuration")
        st.write(
            """
            This app uses a custom-trained ViT model to analyze DAS `.npy` files
            from the EPRI interrogator and classify **Vandalism** vs **Ambient** events.
            """
        )
        st.markdown("---")
        st.info(
            "Required secrets (set by app owner):\n\n"
            "- `GOOGLE_AI_API_KEY` – Gemini for natural-language reporting\n"
            "- `HF_TOKEN` – Hugging Face token for ViT model"
        )

        gemini_api_key = st.secrets.get("GOOGLE_AI_API_KEY")
        hf_token = st.secrets.get("HF_TOKEN")
        hf_repo_name = "benrutgers/epri-das-classifier"

    return gemini_api_key, hf_token, hf_repo_name

# --- 10. Header ---
def render_header():
    st.markdown(
        """
        <div class="nec-header">
            <div class="nec-header-left">
                <div class="nec-header-title">⚡ NEC & EPRI | DAS Fault Detection Agent</div>
                <div class="nec-header-subtitle">
                    Near-real-time vandalism and disturbance monitoring for power grid fiber sensing.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- 11. Alert badge helper ---
def render_alert_badge(vandalism_count: int):
    if vandalism_count and vandalism_count > 0:
        html = """
        <span class="alert-badge alert-badge-critical">
            CRITICAL
        </span>
        """
    else:
        html = """
        <span class="alert-badge alert-badge-normal">
            NORMAL
        </span>
        """
    st.markdown(html, unsafe_allow_html=True)

# ------------- MAIN APP -------------
def main():
    GEMINI_API_KEY, HF_TOKEN, HF_REPO_NAME = render_sidebar()
    render_header()

    if not HF_TOKEN:
        st.error("Hugging Face token not set in Secrets. Cannot load AI model.")
        return

    processor, model = load_ai_model(HF_REPO_NAME, HF_TOKEN)
    if not model:
        return

    # Session state for per-file caching
    if "file_info" not in st.session_state:
        st.session_state.file_info = None
        st.session_state.num_chunks = None
        st.session_state.vandalism_count = None
        st.session_state.report_text = None
        st.session_state.pdf_data = None
        st.session_state.current_time_str = None

    # --- Section 1: Upload ---
    with st.container():
        st.markdown(
            '<div class="section-card"><h4>1. Upload a DAS Sensor File</h4>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="helper-text">Upload a `.npy` file exported from the NEC LS3300 DAS interrogator. '
            'Typical files represent ~12 seconds of high-resolution acoustic data.</p>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Drag & drop or browse for a .npy file",
            type=["npy"],
            key="npy_uploader",
            label_visibility="collapsed",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is None:
        return

    # Determine if this is a new file
    current_info = (uploaded_file.name, uploaded_file.size)
    is_new_file = (st.session_state.file_info != current_info)

    # If new file -> reset cached outputs and run analysis
    if is_new_file:
        st.session_state.file_info = current_info
        st.session_state.num_chunks = None
        st.session_state.vandalism_count = None
        st.session_state.report_text = None
        st.session_state.pdf_data = None
        st.session_state.current_time_str = None

        with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
            S = np.load(uploaded_file).astype(np.float32)
            nt, nx = S.shape

            S_filtered = highpass_filter(S, fs=FS, cutoff=HP_CUTOFF, order=HP_ORDER)
            S_final = S_filtered - S_filtered.mean(axis=0, keepdims=True)

            total_samples = S_final.shape[0]
            num_chunks = math.floor(total_samples / CHUNK_SAMPLES)

            st.write(
                f"File has **{nt}** time steps across **{nx} channels**. "
                f"Slicing into **{num_chunks}** 0.2-second chunks for analysis..."
            )

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
                predicted_class_idx = logits.argmax(-1).item()
                predicted_class = model.config.id2label[predicted_class_idx]

                if predicted_class == "vandalism":
                    vandalism_count += 1
                else:
                    ambient_count += 1

                del inputs, outputs, logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                progress_bar.progress(
                    (i + 1) / num_chunks,
                    text=f"Analyzing chunk {i+1}/{num_chunks}",
                )

            progress_bar.empty()
            st.success("Analysis Complete.")

            st.session_state.num_chunks = num_chunks
            st.session_state.vandalism_count = vandalism_count

            try:
                del S, S_filtered, S_final
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # If we have results, show them
    if st.session_state.num_chunks is None:
        return

    # --- Section 2: Analysis Results ---
    with st.container():
        st.markdown(
            '<div class="section-card"><h4>2. Analysis Results</h4>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown('<div class="metric-label">Total chunks analyzed</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{st.session_state.num_chunks}</div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown('<div class="metric-label">Vandalism events detected</div>', unsafe_allow_html=True)
            vandalism_val = st.session_state.vandalism_count or 0
            st.markdown(
                f'<div class="metric-value">{vandalism_val}</div>',
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown('<div class="metric-label">Alert status</div>', unsafe_allow_html=True)
            render_alert_badge(st.session_state.vandalism_count)

        st.markdown(
            '<p class="helper-text">Each chunk represents a 0.2-second window of DAS activity along the monitored span.</p>',
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Section 3: AI Agent Report + PDF ---
    with st.container():
        st.markdown(
            '<div class="section-card"><h4>3. AI Agent Report</h4>',
            unsafe_allow_html=True,
        )

        if GEMINI_API_KEY is None:
            st.error("Google AI API Key not set in Secrets. Cannot generate report.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Call Gemini + build PDF once per file
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

            st.markdown(
                '<p class="helper-text">The narrative below is generated automatically from the classification statistics above. It is also embedded in the downloadable PDF report.</p>',
                unsafe_allow_html=True,
            )

            st.text_area(
                "Generated Report (Plain Text)",
                st.session_state.report_text or "",
                height=210,
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

            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
