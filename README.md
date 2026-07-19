# Aplikasi Deteksi Bentuk Geometris — UAS Pengolahan Citra Digital

**Project UAS Pengolahan Citra Digital**

| | |
|---|---|
| **Nama** | Mufalah Syifa |
| **NIM** | 24041014 |
| **Kelas** | 4D |
| **Program Studi** | D3 Teknik Komputer |

Aplikasi desktop untuk mendeteksi dan mengklasifikasikan bentuk geometris (Lingkaran, Persegi, Segitiga, Bintang) dari gambar.

Aplikasi punya 2 mode:
- **Menu Utama** — langsung pakai: muat gambar, pilih model, lihat hasil.
- **Buat Model** — susun pipeline sendiri dari awal (wizard 6 langkah), lalu simpan jadi file model.

**Alur pipeline:**
`Input Citra → Praproses → Segmentasi → Ekstraksi Fitur → Pengenalan Pola → Evaluasi`

---

## 1. Isi Folder

```
CV_Deteksi_Bentuk_Geometris/
├── dataset/                  # Folder dataset (auto-generated)
│   ├── lingkaran/
│   ├── persegi/
│   ├── segitiga/
│   └── bintang/
├── models/                   # Tempat file model (.pkl)
│   └── model_default.pkl     # Model default dari create_default_model.py
├── create_default_model.py   # Buat model default
├── dataset_generator.py      # Generate dataset sintetis
├── image_processing.py       # Semua fungsi pengolahan citra
├── main_app.py                # Jalankan ini buat buka aplikasi
├── README.md                  # File ini
├── requirements.txt            # Daftar library
└── wizard_app.py               # Mode Buat Model (wizard 6 langkah)
```

---

## 2. Persiapan Sebelum Pakai

### 2.1 Buat Virtual Environment (Windows)

```powershell
python -m venv .venv
```

### 2.2 Aktifkan Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

> **Catatan:** Jika muncul error `"cannot be loaded because running scripts is disabled"`, jalankan dulu:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 2.3 Install Library

```powershell
pip install -r requirements.txt
```

**Catatan:**
- **Linux**: Tkinter harus diinstall sendiri: `sudo apt install python3-tk`
- **Windows/Mac**: Tkinter sudah bawaan Python.

---

## 3. Jalankan Pertama Kali

```powershell
python create_default_model.py   # sekali aja, bikin model default + dataset
python main_app.py               # buka aplikasi
```

---

## 4. Menu Utama

| Fitur | Fungsi |
|---|---|
| **Muat Gambar** | Buka file gambar (jpg/png). |
| **Pilih Model** | Pilih dari folder `models/`. Ringkasan pipeline muncul otomatis. |
| **Jalankan** | Jalankan deteksi. Mendukung banyak objek dalam satu gambar. |
| **Buat Model** | Buka wizard (window yang sama). Ada tombol balik. |
| **Evaluasi Model** | Tes performa model — confusion matrix, precision/recall/F1, akurasi total. |

---

## 5. Wizard "Buat Model"

**Navigasi:**
- Panah ➜ di samping langkah aktif untuk maju (harus selesai dulu).
- Klik langsung ke langkah sebelumnya untuk mundur.

**Langkah-langkah:**

| Step | Nama | Fungsi |
|---|---|---|
| 1 | **Input Citra** | Buka file / contoh acak dari dataset / generate dataset / pilih folder sendiri. |
| 2 | **Praproses** | Pilih kategori → teknik → atur parameter (slider/combobox, live preview) → Terapkan. Ada Undo/Redo dan riwayat klik. |
| 3 | **Segmentasi** | Pilih: Otsu, Adaptive Threshold, Watershed, atau K-Means. |
| 4 | **Ekstraksi Fitur** | Lihat fitur semua objek: luas, keliling, circularity, aspect ratio, solidity. |
| 5 | **Klasifikasi** | Pilih Rule-Based (langsung) atau k-NN (latih dulu dari dataset, atur k). |
| 6 | **Evaluasi & Simpan Model** | Pilih folder uji, lihat akurasi, confusion matrix, precision/recall/F1. Kalau oke, kasih nama & klik Simpan Model → file `.pkl` langsung muncul di dropdown Menu Utama. |

---

## 6. Dataset

Dataset **sintetis** dari OpenCV (`dataset_generator.py`). Gambar dibuat digital dengan variasi ukuran, posisi, rotasi, warna, plus blur & noise.

Mau pakai dataset sendiri? Taruh gambar di folder:

```
<dataset>/<nama_kelas>/
```

lalu pilih folder itu di wizard atau Evaluasi Model.

---

## 7. Rule-Based vs k-NN

| Aspek | Rule-Based | k-NN |
|---|---|---|
| Sumber aturan | IF-ELSE manual | Belajar dari data latih |
| Butuh data latih? | Tidak | Ya |
| Transparansi | Tinggi | Abstrak ("3 tetangga terdekat") |

Keduanya ada di wizard, bisa dibandingkan langsung.

---

## 8. Troubleshooting

| # | Masalah | Solusi |
|---|---|---|
| 1 | `"Tkinter not found"` (Linux) | `sudo apt install python3-tk` |
| 2 | `"No module named 'cv2'"` | `pip install opencv-python` |
| 3 | Error aktivasi venv | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| 4 | Dataset kosong | `python create_default_model.py` atau `python dataset_generator.py` |
| 5 | Model tidak muncul di dropdown | Pastikan folder `models/` ada isinya. Kalau belum, jalankan `create_default_model.py`. |

---

Selamat mencoba! 🚀
