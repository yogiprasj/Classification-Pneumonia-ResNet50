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

# LOAD MODEL + THRESHOLD
@st.cache_resource
def load_all():
    model = load_model("pneumonia_classification.keras")

    with open("config_threshold.json", "r") as f:
        config = json.load(f)

    threshold = config["threshold"]
    return model, threshold


model, threshold = load_all()

# LOAD LOGO 
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


img_base64 = get_base64_image("logo.png")

# SIDEBAR
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" width="120">
            <h3 style="margin-bottom: 0;">Muhammad Yogi Prasojo</h3>
            <p style="margin: 0;">11122016</p>
            <p style="margin: 0;">Sistem Informasi</p>
            <p style="margin: 0;">Universitas Gunadarma</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("---")

    st.markdown("## 📖 Cara Penggunaan")
    st.info(
        "1. Upload foto Chest X-Ray (Rontgen Paru)\n"
        "2. Format: JPG / PNG / JPEG\n"
        "3. Tunggu proses analisis AI\n"
        "4. Lihat hasil probabilitas dan diagnosa"
    )

    st.warning("⚠️ Sistem ini hanya sebagai pendukung keputusan, bukan diagnosis medis final.")

# PREPROCESS
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# GAUGE CHART 
def plot_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'font': {'size': 60}},
        title={'text': "Probabilitas Pneumonia (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#FF4D4D", 'thickness': 0.35},
            'borderwidth': 0
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400
    )

    return fig

# HEADER
st.title("🩺 Pneumonia Detection Dashboard")
st.markdown("AI-based screening menggunakan ResNet50 + Threshold Optimization")
st.write("---")

# LAYOUT
col1, col2 = st.columns([1, 1])

# LEFT: UPLOAD
with col1:
    st.subheader("📤 Upload X-ray")

    uploaded_file = st.file_uploader(
        "Upload gambar X-ray (jpg/png/jpeg)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Input Image", use_container_width=True)

# RIGHT: RESULT
with col2:
    st.subheader("📊 Hasil Analisis")

    if uploaded_file:
        img_processed = preprocess(img)
        prob = model.predict(img_processed)[0][0]

        st.plotly_chart(plot_gauge(prob), use_container_width=True)
        st.metric("Probability", f"{prob:.4f}")

        # DECISION + REKOMENDASI
        if prob > threshold:
            st.error(f"PNEUMONIA ({prob:.2%})")

            st.markdown("### 🩺 Rekomendasi:")
            st.write("""
            - Hasil menunjukkan indikasi pneumonia berdasarkan analisis citra X-ray  
            - Disarankan segera konsultasi dengan tenaga medis profesional  
            - Lakukan pemeriksaan lanjutan seperti CT-Scan atau tes laboratorium  
            - Istirahat yang cukup dan hindari aktivitas berat  
            - Waspadai gejala seperti sesak napas, demam tinggi, atau batuk berkepanjangan  
            """)

        else:
            st.success(f"NORMAL ({prob:.2%})")

            st.markdown("### 🧠 Rekomendasi:")
            st.write("""
            - Tidak ditemukan indikasi pneumonia pada citra  
            - Tetap jaga kesehatan paru-paru dan hindari polusi  
            - Lakukan olahraga ringan secara rutin  
            - Jika muncul gejala pernapasan, segera lakukan pemeriksaan medis  
            - Pemeriksaan berkala tetap disarankan untuk deteksi dini  
            """)

        # DETAIL
        with st.expander("Detail Sistem"):
            st.write(f"Threshold: {threshold}")
            st.write("Model: ResNet50 (Transfer Learning)")
            st.write("Input Size: 224x224")

# FOOTER
st.write("---")
st.markdown(
    "<div style='text-align: center;'>© 2026 - Skripsi Sistem Informasi Universitas Gunadarma</div>",
    unsafe_allow_html=True
)