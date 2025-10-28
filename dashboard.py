# dashboard.py
import os
import base64
import numpy as np
import torch
import streamlit as st
import textwrap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================== KONFIGURASI MODEL ==================
MODEL_DIR = os.getenv("MODEL_DIR", "model_nusa_multi_label/model_nusa_multi_label")  # folder model
MAX_LEN   = 128
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


LABELS = ["Ujaran Kebencian", "Abusive", "Ujaran Kebencian Agama", "Rasist"]


# ================== FUNGSI UTILITAS ==================
def load_css(file_name: str):
    """Membaca file CSS lokal dan menginjeksikannya ke dalam <style> tag."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"File CSS '{file_name}' tidak ditemukan. Pastikan file tersebut ada.")

@st.cache_resource(show_spinner="Memuat model…")
def load_model_and_tokenizer(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval().to(DEVICE)

    # Tentukan jumlah & nama label
    num_labels = getattr(mdl.config, "num_labels", None)
    if num_labels is None and hasattr(mdl, "classifier"):
        num_labels = mdl.classifier.out_features
    if num_labels is None:
        raise RuntimeError("Tidak bisa menentukan jumlah label dari model.")

    if LABELS is None:
        names = [f"Label_{i}" for i in range(num_labels)]
    else:
        names = LABELS
        if len(names) != num_labels:
            st.warning(
                f"Jumlah LABELS ({len(names)}) ≠ num_labels model ({num_labels}). "
                "Memakai auto-label."
            )
            names = [f"Label_{i}" for i in range(num_labels)]
    return mdl, tok, names

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_multilabel(texts, threshold=0.5):
    model, tokenizer, label_names = load_model_and_tokenizer(MODEL_DIR)
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        logits = out.logits.detach().cpu().numpy()
        probs = sigmoid(logits)  # shape: (B, L)

    results = []
    for row in probs:
        picked = np.where(row >= threshold)[0].tolist()
        preds = [(label_names[i], float(row[i])) for i in picked]
        results.append({
            "probs": {label_names[i]: float(row[i]) for i in range(len(label_names))},
            "preds": sorted(preds, key=lambda x: x[1], reverse=True)
        })
    return results

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(page_title="TraKasar – Deteksi Ujaran Kebencian", page_icon="🤖", layout="wide")
load_css("style.css")

# ================== STATE & NAV ==================
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

qp = st.query_params
if "page" in qp and qp.get("page") != st.session_state.page:
    st.session_state.page = qp.get("page")

def go(p):
    st.session_state.page = p
    st.query_params.update({"page": p})

# Navbar (logo base64)
try:
    with open("UNTAR LOGO.png", "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    logo_untar_html = f'<img src="data:image/png;base64,{img_base64}" width="75" />'
except FileNotFoundError:
    logo_untar_html = '<span class="untar-text">UNTAR</span>'

# helper untuk embed gambar lokal jadi base64 (aman untuk <img>/CSS inline)
def img_b64(path):
    try:
        with open(path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

# siapkan base64 
MARCHELINO_IMG = img_b64("marchelino.jpeg")
VINY_IMG       = img_b64("Viny-Christanti.jpg")
SEKPRODI_IMG   = img_b64("Sekprodi-FTI.jpg")

# --- Navbar ---
# navbtn mengembalikan HTML SATU BARIS (tanpa indentasi)
def navbtn(label: str) -> str:
    active = "active" if st.session_state.page == label else ""
    return f'<form class="navform" method="get"><input type="hidden" name="page" value="{label}"><button class="navbtn {active}" type="submit">{label}</button></form>'

# rakit links 
links_html = "".join([
    navbtn("Beranda"),
    navbtn("Deteksi"),
    navbtn("Tentang"),
    navbtn("Bantuan"),
])

# string HTML tanpa indent 
nav_html = (
    '<div class="navbar">'
      '<div class="nav-inner">'
        '<div class="brand"><div class="logo">🤖</div>TraKasar</div>'
        f'<div class="links">{links_html}</div>'
        f'<div class="partner">{logo_untar_html}</div>'
      '</div>'
    '</div>'
)

# render dengan markdown + unsafe_allow_html
st.markdown(nav_html, unsafe_allow_html=True)

# ================== HALAMAN ==================
page = st.session_state.page

# beri atribut ke <body> agar bisa ditarget CSS per-halaman
st.markdown(
    f"<script>document.body.setAttribute('data-page','{page}')</script>",
    unsafe_allow_html=True
)

if page == "Beranda":
    st.markdown("""
    <div class="beranda-wrap">
      <section class="hero">
        <div class="kicker">Selamat Datang di Platform TraKasar</div>
        <div class="headline">Lindungi ruang digital dari kata-kata yang menyakiti warga Papua.</div>
      </section>
      <div class="card-tentang">   
        <h3 style="text-align: center; margin-top: 0; color: #333;">Tentang Proyek Ini</h3>
        <p style="color: #4b5563; font-size: 1rem; line-height: 1.6;">
            <b>TraKasar</b> adalah sistem untuk mendeteksi ujaran kebencian berbahasa Indonesia,
            dengan fokus konteks warga Papua.
        </p>
        <p style="color: #4b5563; font-size: 1rem; line-height: 1.6;">
            Tujuannya membantu moderasi konten agar lingkungan online lebih inklusif.
        </p>
        <br>
        <p style="color: #4b5563; font-size: 1rem; line-height: 1.6;">
            Untuk mulai menganalisis teks, silakan buka halaman <b>Deteksi</b>.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Deteksi":
    st.markdown("""
    <section class="hero">
      <div class="kicker">Deteksi</div>
      <div class="headline">Masukkan kalimat lalu klik <i>Mulai Deteksi</i>.</div>
    </section>
    """, unsafe_allow_html=True)

    
    text = st.text_area(
        "Kalimat",
        placeholder="Tulis kalimat yang ingin diperiksa...",
        key="detect_text",
        label_visibility="collapsed"
    )
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        threshold = st.slider(
            "Threshold",
            0.10, 0.95, 0.50, 0.05,
            help="Probabilitas minimal agar label dianggap terdeteksi."
        )
    with colB:
        run = st.button("Mulai Deteksi", key="run_detect")
    with colC:
        reset = st.button("Bersihkan", key="reset_detect")
    
   # inisialisasi session_state untuk hasil
    if "result_detect" not in st.session_state:
        st.session_state.result_detect = None
    if "detect_text" not in st.session_state:
        st.session_state.detect_text = ""
    
    # callback untuk tombol Bersihkan
    def on_reset():
        st.session_state.detect_text = ""          
        st.session_state.result_detect = None      

    if run:
        if not text or text.strip() == "":
            st.warning("Isi dulu kalimatnya ya bro 🙏")
        else:
            with st.spinner("⏳ Menjalankan model…"):
                try:
                    result = predict_multilabel([text], threshold=threshold)[0]
                except Exception as e:
                    st.error(f"Gagal menjalankan model: {e}")
                    result = None

            if result:
                preds = result["preds"]
                if len(preds) == 0:
                    st.success("✅ Tidak ada label yang melewati threshold.")
                else:
                    st.success("🎯 Label terdeteksi:")
                    chips = " ".join(
                        [f"<span class='chip'>{lbl} — {prob:.2f}</span>" for lbl, prob in preds]
                    )
                    st.markdown(f"<div class='chip-wrap'>{chips}</div>", unsafe_allow_html=True)

                # Tabel probabilitas lengkap
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                with st.expander("Lihat skor semua label"):
                    import pandas as pd
                    df = pd.DataFrame(
                        [{"Label": k, "Prob": v} for k, v in result["probs"].items()]
                    ).sort_values("Prob", ascending=False, ignore_index=True)
                    st.dataframe(df, use_container_width=True)

elif page == "Tentang":
    st.markdown("""
    <section class="hero">
      <div class="kicker">Tentang</div>
      <div class="headline">TraKasar membantu memfilter ujaran kebencian secara real-time.</div>
    </section>
    """, unsafe_allow_html=True)

    # Card "Mengenal lebih jauh TraKasar?"
    st.markdown("""
    <div class="about-card">
      <div class="about-left">
        <div class="about-eyebrow">Mengenal lebih jauh</div>
        <h3 class="about-title">TraKasar<span class="accent">?</span></h3>
        <p>TraKasar merupakan aplikasi masa kini yang digunakan untuk menganalisis ujaran kebencian dalam salah satu bahasa di Indonesia, yakni Papua.</p>
        <p>Dengan hanya sekali klik, ujaran akan dengan cepat terdeteksi.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Kartu profil pengembang 
    st.markdown(f"""
    <div class="profile-card">
      <div class="profile-photo" style="background-image:url('{MARCHELINO_IMG or ""}');"></div>
      <div class="profile-info">
        <div class="eyebrow">Mari bertemu dengan</div>
        <h4 class="profile-name">Marchelino Benediktus Leintan</h4>
        <div class="profile-role">Pengembang Aplikasi</div>
        <p>Mahasiswa Semester 7, Universitas Tarumanagara – Jurusan Teknik Informatika.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Section pembimbing 
    st.markdown(f"""
    <h3 class="section-title">Dosen Pembimbing</h3>
    <div class="mentors-grid">
      <div class="mentor-card">
        <div class="mentor-photo" style="background-image:url('{VINY_IMG or ""}');"></div>
        <div class="mentor-name">Viny Christanti Mawardi, S.Kom., M.Kom.</div>
        <div class="mentor-role">Dosen Pembimbing Utama</div>
      </div>
      <div class="mentor-card">
        <div class="mentor-photo" style="background-image:url('{SEKPRODI_IMG or ""}');"></div>
        <div class="mentor-name">Manatap Dolok Lauro Sitorus, S.Kom., M.M.S.I.</div>
        <div class="mentor-role">Dosen Pendamping</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:  # Bantuan
    # --- layout Bantuan ---
    st.markdown(
        """
        <h1 class="help-title">Cara Menggunakan TraKasar</h1>
        <div class="help-box">
            <ol>
                <li>Pada laman Deteksi, klik tombol "Mulai Deteksi"</li>
                <li>Masukkan Kalimat yang Akan Analisis kedalam kotak bertuliskan "Masukkan Kalimat"</li>
                <li>Setelah Kotak sudah terisi, klik tombol "Mulai Deteksi"</li>
                <li>Maka Hasil akan Muncul dilayar. Hasil Akan berupa label ujaran kebencian sesuai dengan kategori yang tersedia</li>
                <li>Selamat Mencoba!</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

# ================== FOOTER ==================
st.markdown(
    """
    <div class="footer">
        <div class="footer-inner">
            <div class="footer-logo">🤖 TraKasar</div>
            <div class="footer-text">@2025 TraKasar, All Right Reserved</div>
            <div class="footer-spacer">&nbsp;</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
