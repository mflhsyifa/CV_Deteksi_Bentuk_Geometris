================================================================================
APLIKASI DETEKSI BENTUK GEOMETRIS — UAS PENGOLAHAN CITRA DIGITAL
================================================================================

Project UAS Pengolahan Citra Digital
Nama          : Mufalah Syifa
NIM           : 24041014
Kelas         : 4D
Program Studi : D3 Teknik Komputer

Aplikasi desktop buat mendeteksi dan mengklasifikasikan bentuk geometris 
(Lingkaran, Persegi, Segitiga, Bintang) dari gambar.

Ada 2 mode:
- Menu Utama   : langsung pake — muat gambar, pilih model, lihat hasil.
- Buat Model   : susun pipeline sendiri dari awal (wizard 6 langkah), 
                 terus simpan jadi file model.

Alur pipeline: Input Citra -> Praproses -> Segmentasi -> Ekstraksi Fitur ->
Pengenalan Pola -> Evaluasi


================================================================================
1. ISI FOLDER
================================================================================

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
├── main_app.py               # Jalankan ini buat buka aplikasi
├── README.md                 # File ini
├── requirements.txt          # Daftar library
└── wizard_app.py             # Mode Buat Model (wizard 6 langkah)


================================================================================
2. PERSIAPAN SEBELUM PAKAI
================================================================================

2.1 Aktifkan Virtual Environment (Windows)
------------------------------------------
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1


2.2 Install Library
-------------------
pip install -r requirements.txt

Catatan:
- Linux: Tkinter harus install sendiri: sudo apt install python3-tk
- Windows/Mac: Tkinter udah bawaan Python.


================================================================================
3. JALANKAN PERTAMA KALI
================================================================================

python create_default_model.py   # sekali aja, bikin model default + dataset
python main_app.py               # buka aplikasi


================================================================================
4. MENU UTAMA
================================================================================

- Muat Gambar   : buka file gambar (jpg/png).
- Pilih Model   : pilih dari folder models/. Ringkasan pipeline muncul otomatis.
- Jalankan      : jalankan deteksi. Support banyak objek dalam satu gambar.
- Buat Model    : buka wizard (window yang sama). Ada tombol balik.
- Evaluasi Model: tes performa model — confusion matrix, precision/recall/F1,
                  akurasi total.


================================================================================
5. WIZARD "BUAT MODEL"
================================================================================

Navigasi:
- Panah ➜ di samping langkah aktif buat maju (harus selesai dulu).
- Klik langsung ke langkah sebelumnya buat mundur.

+-------+------------------+---------------------------------------------------+
| Step  | Nama             | Fungsi                                            |
+-------+------------------+---------------------------------------------------+
| 1     | Input Citra      | Buka file / contoh acak dari dataset / generate   |
|       |                  | dataset / pilih folder sendiri.                   |
+-------+------------------+---------------------------------------------------+
| 2     | Praproses        | Pilih kategori -> teknik -> atur parameter        |
|       |                  | (slider/combobox, live preview) -> Terapkan.      |
|       |                  | Ada Undo/Redo dan riwayat klik.                   |
+-------+------------------+---------------------------------------------------+
| 3     | Segmentasi       | Pilih: Otsu, Adaptive Threshold, Watershed,       |
|       |                  | atau K-Means.                                     |
+-------+------------------+---------------------------------------------------+
| 4     | Ekstraksi Fitur  | Lihat fitur semua objek: luas, keliling,          |
|       |                  | circularity, aspect ratio, solidity.              |
+-------+------------------+---------------------------------------------------+
| 5     | Klasifikasi      | Pilih Rule-Based (langsung) atau k-NN             |
|       |                  | (latih dulu dari dataset, atur k).                |
+-------+------------------+---------------------------------------------------+
| 6     | Evaluasi &       | Pilih folder uji, lihat akurasi, confusion        |
|       | Simpan Model     | matrix, precision/recall/F1. Kalo oke, kasih nama |
|       |                  | & klik Simpan Model -> file .pkl langsung muncul  |
|       |                  | di dropdown Menu Utama.                           |
+-------+------------------+---------------------------------------------------+


================================================================================
6. DATASET
================================================================================

Dataset SINTETIS dari OpenCV (dataset_generator.py). Gambar dibuat digital dengan
variasi ukuran, posisi, rotasi, warna, plus blur & noise.

Mau pake dataset sendiri? Taruh gambar di:
  <dataset>/<nama_kelas>/
terus pilih folder itu di wizard atau Evaluasi Model.


================================================================================
7. RULE-BASED vs k-NN
================================================================================

+------------------+------------------+----------------------------------------+
|                  | Rule-Based       | k-NN                                   |
+------------------+------------------+----------------------------------------+
| Sumber aturan    | IF-ELSE manual   | Belajar dari data latih                |
+------------------+------------------+----------------------------------------+
| Butuh data latih?| Tidak            | Ya                                     |
+------------------+------------------+----------------------------------------+
| Transparansi     | Tinggi           | Abstrak ("3 tetangga terdekat")        |
+------------------+------------------+----------------------------------------+

Keduanya ada di wizard, bisa dibandingin langsung.


================================================================================
8. KUSTOMISASI (KALO MAU)
================================================================================

- Dataset 9 kelas: tambah aturan di classify_shape() di image_processing.py.
- Multi-objek beda kelas: sekarang cuma support satu kelas per gambar.
- Ganti warna: ubah C_NAVY, C_NAVY_DARK dll di main_app.py & wizard_app.py.


================================================================================
9. TROUBLESHOOTING
================================================================================

1. "Tkinter not found" (Linux)
   sudo apt install python3-tk

2. "No module named 'cv2'"
   pip install opencv-python

3. Error aktivasi venv
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

4. Dataset kosong
   Jalankan create_default_model.py atau dataset_generator.py

5. Model ga muncul di dropdown
   Pastikan models/ ada isinya. Kalo belum, jalankan create_default_model.py

