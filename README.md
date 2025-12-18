# TRAKASAR
**Transformer-based Hate Speech Detection System for Papuan Language**

TRAKASAR adalah aplikasi berbasis web yang dikembangkan untuk mendeteksi ujaran kebencian pada teks berbahasa Indonesia dan Bahasa Papua Umum dengan memanfaatkan model NusaBERT berbasis arsitektur Transformer (encoder-only). Aplikasi ini dirancang sebagai prototype sistem deteksi ujaran kebencian dalam konteks multibahasa lokal Indonesia.

## Fitur Utama
- Deteksi ujaran kebencian secara otomatis
- Mendukung teks Bahasa Indonesia dan Bahasa Papua Umum
- Menggunakan model Fine-Tuning NusaBERT
- Backend berbasis FastAPI
- Antarmuka web sederhana dan mudah digunakan

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
