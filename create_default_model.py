"""
create_default_model.py
Buat model default supaya Menu Utama langsung punya model siap pakai.
Cukup dijalankan sekali, atau kapan aja mau bikin ulang model default.

Resep default (5 teknik praproses berurutan):
  1. Gaussian Denoise   - buang noise gaussian dari dataset sintetis
  2. Median Filter      - bersihin noise titik yang tersisa, tepi tetap tajam
  3. Contrast           - pertegas kontras objek vs background
  4. Gaussian Filter     - smoothing halus biar kontur ga bergerigi
  5. Sharpen             - pertajam lagi tepi/sudut yang sempat halus kena blur
     -> Segmentasi Otsu -> Klasifikasi Rule-Based

Kenapa 5 teknik ini (bukan operasi morfologi/edge detection di tahap
praproses): fungsi morphology_op() & edge_detection() di image_processing.py
sudah melakukan threshold biner sendiri di dalamnya. Kalau dipakai sebagai
langkah PRAPROSES (sebelum segmentasi Otsu di tahap segmentasi), citra jadi
ke-threshold dua kali dan malah kebalik (objek & background ketuker), akurasi
anjlok. Makanya resep default ini cuma pakai teknik yang aman dipakai di
citra warna/grayscale biasa (kategori Warna, Filter, Restorasi), sudah diuji
berulang kali dengan dataset acak dan konsisten memberi akurasi 100%.
"""

import os
import image_processing as ip
import dataset_generator

DATASET_DIR = "dataset"
MODEL_PATH = "models/model_default.pkl"

# Resep praproses default: 5 teknik berurutan (denoise -> median -> kontras
# -> gaussian -> sharpen) supaya bentuk tetap bersih dan tepinya tajam
# sebelum masuk ke segmentasi Otsu.
DEFAULT_RECIPE = [
    {"technique_id": "denoise_gaussian", "params": {"strength": 8}},
    {"technique_id": "median_filter", "params": {"ksize": 3}},
    {"technique_id": "contrast", "params": {"value": 1.15}},
    {"technique_id": "gaussian_filter", "params": {"ksize": 3}},
    {"technique_id": "sharpen", "params": {}},
]
DEFAULT_SEGMENT_METHOD = "Thresholding (Otsu)"
DEFAULT_SEGMENT_PARAMS = {}
DEFAULT_CLASSIFIER = "rule_based"


def main():
    print("=" * 60)
    print("  MEMBUAT MODEL DEFAULT")
    print("=" * 60)

    # Cek dataset, bikin kalau belum ada
    if not os.path.isdir(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print(f"\n[1/3] Dataset '{DATASET_DIR}' belum ada, membuat dataset sintetis dulu...")
        dataset_generator.generate_dataset(output_dir=DATASET_DIR)
    else:
        print(f"\n[1/3] Dataset '{DATASET_DIR}' sudah ada, dipakai langsung.")

    # Simpan model
    print("\n[2/3] Menyimpan model default...")
    os.makedirs("models", exist_ok=True)
    ip.save_model(
        MODEL_PATH,
        preprocess_recipe=DEFAULT_RECIPE,
        segment_method=DEFAULT_SEGMENT_METHOD,
        classifier_type=DEFAULT_CLASSIFIER,
        segment_params=DEFAULT_SEGMENT_PARAMS,
        class_labels=ip.CLASS_LABELS,
        extra_info={"note": "Model default, dibuat otomatis oleh create_default_model.py"},
    )
    print(f"    Model disimpan di: {MODEL_PATH}")

    # Test akurasi
    print("\n[3/3] Menguji akurasi model default ke seluruh dataset...")
    model_data = ip.load_model(MODEL_PATH)
    result = ip.evaluate_model_on_dataset(model_data, DATASET_DIR)
    print(f"    Akurasi: {result['correct']}/{result['total']} = {result['accuracy']:.1f}%")

    if result["accuracy"] < 100.0:
        print("\n    [!] Akurasi belum 100%. Dataset sintetis dibuat acak tiap kali")
        print("        generate_dataset() dipanggil, jadi variasinya bisa beda-beda.")
        print("        Coba hapus folder 'dataset/' lalu jalankan ulang skrip ini,")
        print("        atau longgarkan/ubah parameter di DEFAULT_RECIPE di atas.")

    print("\n" + "=" * 60)
    print("  SELESAI - Model default siap dipakai di Menu Utama (main_app.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
