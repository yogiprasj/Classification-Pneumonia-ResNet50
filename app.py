import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from PIL import Image
import plotly.graph_objects as go
import base64

# CONFIG PAGE
st.set_page_config(
    page_title="Pneumonia Detection Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #F5F5F5;
    --card:      #DFF1F1;
    --accent:    #BBD5DA;
    --danger:    #FF0000;
    --danger-soft: #FFE5E5;
    --success:   #1A7A4A;
    --success-soft: #E2F5EC;
    --text-dark: #1C2B35;
    --text-mid:  #3D5A66;
    --text-light:#6B8B97;
    --shadow:    0 4px 24px rgba(28,43,53,.08);
    --radius:    16px;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-dark);
}


[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0) !important;
    height: 0px !important;
}
[data-testid="stAppDeployButton"] {
    display: none !important;
}
[data-testid="stHeader"] > div:first-child > button {
    position: fixed !important;
    top: 25px !important;
    left: 15px !important;
    z-index: 999999;
    
    background-color: #1C2B35 !important; 
    color: #DFF1F1 !important;
    
    border: 2px solid #BBD5DA !important; 
    border-radius: 8px !important;
    width: 42px !important;
    height: 38px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    transition: all 0.3s ease;
}
[data-testid="stHeader"] > div:first-child > button:hover {
    background-color: #2e5060 !important;
    color: #ffffff !important;
    transform: scale(1.05); /* Sedikit membesar saat di-hover */
}
#MainMenu { display: none !important; }
footer { display: none !important; }

/* ── Main content padding ── */
[data-testid="stMainBlockContainer"] {
    padding: 2rem 2.5rem 1.5rem !important;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #1C2B35 0%, #243d4a 60%, #2e5060 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #DFF1F1 !important; }
[data-testid="stSidebar"] hr {
    border-color: rgba(187,213,218,.25) !important;
    margin: 1.25rem 0 !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.05rem !important;
    letter-spacing: .02em;
    color: #BBD5DA !important;
}

/* Info / Warning boxes in sidebar */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(187,213,218,.12) !important;
    border: 1px solid rgba(187,213,218,.3) !important;
    border-radius: 10px !important;
    color: #DFF1F1 !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p,
[data-testid="stSidebar"] [data-testid="stAlert"] div { color: #DFF1F1 !important; }

/* ── Dashboard header ── */
.dash-header {
    background: linear-gradient(120deg, #1C2B35 0%, #2e5060 50%, #3a6b7a 100%);
    border-radius: var(--radius);
    padding: 2rem 2.4rem;
    margin-bottom: 1.75rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(187,213,218,.12);
}
.dash-header::after {
    content: '';
    position: absolute;
    bottom: -60px; right: 80px;
    width: 240px; height: 240px;
    border-radius: 50%;
    background: rgba(187,213,218,.07);
}
.dash-header-icon {
    font-size: 2.8rem;
    line-height: 1;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,.3));
}
.dash-header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #DFF1F1;
    margin: 0;
    line-height: 1.15;
}
.dash-header-sub {
    font-size: .9rem;
    color: #BBD5DA;
    margin: .25rem 0 0;
    font-weight: 400;
    letter-spacing: .01em;
}

/* ── Card wrapper ── */
.card {
    background: var(--card);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    box-shadow: var(--shadow);
    border: 1px solid rgba(187,213,218,.6);
    margin-bottom: 1.25rem;
}
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: var(--text-dark);
    margin: 0 0 1rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,.7) !important;
    border-radius: 12px !important;
    padding: .25rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.85) !important;
    border: 2px dashed var(--accent) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #7ab5bf !important;
    background: white !important;
}
/* Fix file uploader text colours */
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] button {
    color: var(--text-dark) !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: var(--text-dark) !important;
}
/* Uploaded file name row */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] p,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] div {
    color: var(--text-dark) !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,.55);
    border-radius: 10px;
    padding: .65rem 1rem !important;
    border: 1px solid rgba(187,213,218,.5);
}
[data-testid="stMetricLabel"] { color: var(--text-mid) !important; font-size: .8rem !important; }
[data-testid="stMetricValue"] { color: var(--text-dark) !important; font-size: 1.5rem !important; font-weight: 700 !important; }

/* ── Diagnosis badge ── */
.diag-badge {
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    font-size: 1.4rem;
    font-weight: 700;
    font-family: 'DM Serif Display', serif;
    display: flex;
    align-items: center;
    gap: .7rem;
    margin-bottom: 1.1rem;
    letter-spacing: .01em;
}
.diag-pneumonia {
    background: var(--danger-soft);
    border: 2px solid #FF0000;
    color: #C00000;
}
.diag-normal {
    background: var(--success-soft);
    border: 2px solid #1A7A4A;
    color: var(--success);
}

/* ── Recommendation box ── */
.reco-box {
    background: rgba(255,255,255,.55);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    border-left: 4px solid var(--accent);
    margin-top: .6rem;
}
.reco-box.reco-danger { border-left-color: #FF0000; }
.reco-box.reco-success { border-left-color: #1A7A4A; }
.reco-box h4 {
    margin: 0 0 .65rem;
    font-size: .95rem;
    color: var(--text-dark);
    font-weight: 600;
}
.reco-item {
    display: flex;
    align-items: flex-start;
    gap: .55rem;
    margin-bottom: .55rem;
    font-size: .88rem;
    color: var(--text-mid);
    line-height: 1.5;
}
.reco-item span.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-top: .42rem;
    flex-shrink: 0;
}
.reco-danger .dot { background: #FF0000; }
.reco-success .dot { background: #1A7A4A; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,.55) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(187,213,218,.5) !important;
    margin-top: .85rem !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: var(--text-dark) !important;
    font-size: .9rem !important;
}
[data-testid="stExpander"] p,
[data-testid="stExpander"] div {
    color: var(--text-mid) !important;
    font-size: .88rem !important;
}

/* ── Divider ── */
hr { border-color: rgba(187,213,218,.45) !important; margin: 1.4rem 0 !important; }

/* ── Column card styling (replaces broken manual div wrappers) ── */
[data-testid="column"]:nth-child(1) > div:first-child,
[data-testid="column"]:nth-child(2) > div:first-child {
    background: var(--card) !important;
    border-radius: var(--radius) !important;
    padding: 1.6rem 1.8rem !important;
    box-shadow: var(--shadow) !important;
    border: 1px solid rgba(187,213,218,.6) !important;
    min-height: 200px;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    box-shadow: 0 3px 14px rgba(28,43,53,.12) !important;
}

/* ── Footer ── */
.site-footer {
    text-align: center;
    font-size: .82rem;
    color: var(--text-light);
    padding: 1.1rem 0 .5rem;
    letter-spacing: .02em;
}
.site-footer strong { color: var(--text-mid); }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL + THRESHOLD ────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    model = load_model("pneumonia_classification.keras")
    with open("config_threshold.json", "r") as f:
        config = json.load(f)
    threshold = config["threshold"]
    return model, threshold

model, threshold = load_all()


# ── LOAD LOGO ─────────────────────────────────────────────────────────────────
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_image("logo.png")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
        <div style="text-align:center; padding: 1.4rem 0 .6rem;">
            <img src="data:image/png;base64,{img_base64}" 
                 style="width:100px;"> <!-- Bagian border, radius, dan shadow dihapus -->
            <h3 style="margin:.85rem 0 .2rem; font-family:'DM Serif Display',serif; 
                        font-size:1.1rem; font-weight:400; color:#DFF1F1;">
                Muhammad Yogi Prasojo
            </h3>
            <p style="margin:.1rem 0; font-size:.82rem; color:#BBD5DA;">NPM: 11122016</p>
            <p style="margin:.1rem 0; font-size:.82rem; color:#BBD5DA;">Sistem Informasi</p>
            <p style="margin:.1rem 0; font-size:.82rem; color:#BBD5DA;">Universitas Gunadarma</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("---")

    st.markdown("## Cara Penggunaan")
    st.info(
        "1. Upload foto Chest X-Ray (Rontgen Paru)\n"
        "2. Format: JPG / PNG / JPEG\n"
        "3. Tunggu proses analisis AI\n"
        "4. Lihat hasil probabilitas dan diagnosa"
    )

    st.warning("⚠️ Sistem ini hanya sebagai pendukung keputusan, bukan diagnosis medis final.")


# ── PREPROCESS ────────────────────────────────────────────────────────────────
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ── GAUGE CHART ───────────────────────────────────────────────────────────────
def plot_gauge(prob):
    pct = prob * 100
    if pct < 40:
        bar_color = "#1A7A4A"
    elif pct < 65:
        bar_color = "#E6A817"
    else:
        bar_color = "#FF0000"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'font': {'size': 54, 'color': '#1C2B35', 'family': 'DM Sans'}, 'suffix': '%'},
        title={'text': "Probabilitas Pneumonia", 'font': {'size': 15, 'color': '#3D5A66', 'family': 'DM Sans'}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': '#BBD5DA',
                'tickfont': {'color': '#6B8B97', 'size': 11}
            },
            'bar': {'color': bar_color, 'thickness': 0.32},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40],  'color': 'rgba(26,122,74,.10)'},
                {'range': [40, 65], 'color': 'rgba(230,168,23,.10)'},
                {'range': [65, 100],'color': 'rgba(255,0,0,.09)'},
            ],
            'threshold': {
                'line': {'color': '#1C2B35', 'width': 2},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(t=40, b=10, l=20, r=20)
    )
    return fig


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <div class="dash-header-icon">🩺</div>
    <div>
        <p class="dash-header-title">Pneumonia Detection Dashboard</p>
        <p class="dash-header-sub">AI-based screening menggunakan ResNet50 + Threshold Optimization</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── TWO-COLUMN LAYOUT ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="medium")

# ── LEFT: UPLOAD ──────────────────────────────────────────────────────────────
with col1:
    st.markdown('<p class="card-title">📤 Upload X-Ray</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload gambar X-ray (jpg / png / jpeg)",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Input Image", use_container_width=True)


# ── RIGHT: RESULT ─────────────────────────────────────────────────────────────
with col2:
    st.markdown('<p class="card-title">📊 Hasil Analisis</p>', unsafe_allow_html=True)

    if uploaded_file:
        img_processed = preprocess(img)
        prob = model.predict(img_processed)[0][0]

        # Gauge
        st.plotly_chart(plot_gauge(prob), use_container_width=True)

        # Probability metric
        st.metric("Probability Score", f"{prob:.4f}")

        # ── DIAGNOSIS ──
        if prob > threshold:
            st.markdown(f"""
            <div class="diag-badge diag-pneumonia">
                🔴 PNEUMONIA &nbsp;|&nbsp; {prob:.2%}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="reco-box reco-danger">
                <h4>🩺 Rekomendasi</h4>
                <div class="reco-item"><span class="dot"></span>Hasil menunjukkan indikasi pneumonia berdasarkan analisis citra X-ray</div>
                <div class="reco-item"><span class="dot"></span>Disarankan segera konsultasi dengan tenaga medis profesional</div>
                <div class="reco-item"><span class="dot"></span>Lakukan pemeriksaan lanjutan seperti CT-Scan atau tes laboratorium</div>
                <div class="reco-item"><span class="dot"></span>Istirahat yang cukup dan hindari aktivitas berat</div>
                <div class="reco-item"><span class="dot"></span>Waspadai gejala seperti sesak napas, demam tinggi, atau batuk berkepanjangan</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="diag-badge diag-normal">
                🟢 NORMAL &nbsp;|&nbsp; {prob:.2%}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="reco-box reco-success">
                <h4>🧠 Rekomendasi</h4>
                <div class="reco-item"><span class="dot"></span>Tidak ditemukan indikasi pneumonia pada citra</div>
                <div class="reco-item"><span class="dot"></span>Tetap jaga kesehatan paru-paru dan hindari polusi</div>
                <div class="reco-item"><span class="dot"></span>Lakukan olahraga ringan secara rutin</div>
                <div class="reco-item"><span class="dot"></span>Jika muncul gejala pernapasan, segera lakukan pemeriksaan medis</div>
                <div class="reco-item"><span class="dot"></span>Pemeriksaan berkala tetap disarankan untuk deteksi dini</div>
            </div>
            """, unsafe_allow_html=True)

        # ── DETAIL SISTEM ──
        with st.expander("🔧 Detail Sistem"):
            st.write(f"**Threshold:** {threshold}")
            st.write("**Model:** ResNet50 (Transfer Learning)")
            st.write("**Input Size:** 224 × 224 px")

    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#6B8B97;">
            <div style="font-size:3rem; margin-bottom:.75rem;">🫁</div>
            <p style="font-size:.95rem; margin:0;">Upload X-ray pada panel kiri untuk memulai analisis.</p>
        </div>
        """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div class="site-footer">
    © 2026 &nbsp;·&nbsp; <strong>Skripsi Sistem Informasi Universitas Gunadarma</strong>
    &nbsp;·&nbsp; Muhammad Yogi Prasojo
</div>
""", unsafe_allow_html=True)