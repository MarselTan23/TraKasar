# TRAKASAR
**Transformer-based Hate Speech Detection System for Papuan Language**

**TraKasar** adalah aplikasi berbasis web yang dikembangkan untuk mendeteksi ujaran kebencian berbahasa Indonesia dengan fokus pada konteks ujaran terhadap warga Papua.  
Aplikasi ini menggunakan model **Transformer (encoder-only)** yang telah di-*fine-tuning* untuk tugas **multi-label classification**, sehingga satu kalimat dapat memiliki lebih dari satu kategori ujaran kebencian.

Aplikasi dibangun menggunakan **Streamlit** sebagai antarmuka web dan **PyTorch + Hugging Face Transformers** sebagai backend model inferensi.

---

## 🎯 Tujuan Pengembangan
TraKasar bertujuan untuk:
- Membantu proses **moderasi konten digital**
- Mengurangi penyebaran ujaran kebencian di ruang publik daring
- Mendukung lingkungan digital yang **inklusif dan aman**

---

## 🚀 Fitur Utama
- 🔍 **Deteksi ujaran kebencian berbasis teks**
- 🏷️ **Multi-label classification** (satu teks bisa memiliki banyak label)
- 🎚️ **Pengaturan threshold probabilitas**
- 📊 **Menampilkan skor probabilitas seluruh label**
- 💾 **Unduh hasil deteksi** dalam format:
  - JSON
  - CSV
  - TXT
- 🖥️ **Antarmuka web interaktif** berbasis Streamlit

---

## 🧠 Label Klasifikasi
Model TraKasar mendeteksi beberapa kategori ujaran kebencian, antara lain:

- Ujaran Kebencian  
- Abusive  
- Ujaran Kebencian Agama  
- Rasist  

---

## 🧩 Arsitektur & Teknologi
- **Bahasa Pemrograman**: Python  
- **Framework UI**: Streamlit  
- **Model NLP**: Transformer (NusaBERT-based)  
- **Library Utama**:
  - `torch`
  - `transformers`
  - `numpy`
  - `pandas`

## Struktur Folder
Seluruh file dan folder harus berada dalam satu direktori utama proyek.


## Prasyarat Sistem
- Python versi 3.10 atau lebih baru
- pip (Python package manager)

## Instalasi
1. Pastikan seluruh file proyek berada dalam satu folder.
2. Buka terminal atau command prompt pada direktori proyek.
3. Install seluruh dependensi dengan perintah:

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi
- Untuk menjalankan aplikasi gunakan command berikut pada terminal pc :
```bash
streamlit run dashboard.py
```
- Aplikasi dapat diakses melalui browser pada alamat :
```bash
https://huggingface.co/spaces/AmGod23/TraKasar
