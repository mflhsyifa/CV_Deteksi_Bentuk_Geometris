"""
image_processing.py
Berisi seluruh fungsi pengolahan citra untuk pipeline:
Praproses -> Segmentasi -> Ekstraksi Fitur -> Pengenalan Pola

Dipisah dari kode GUI supaya mudah diuji tanpa perlu membuka jendela GUI.
Semua fungsi menerima dan mengembalikan citra dalam format BGR (OpenCV) uint8,
kecuali disebutkan lain.
"""

import cv2
import numpy as np
import pickle
import os
import glob
from datetime import datetime
from scipy.signal import wiener as _scipy_wiener


# =====================================================================
# LAPISAN 1: PRAPROSES CITRA
# =====================================================================

def blur_filter(img, method="gaussian", ksize=5):
    """
    Filter smoothing: mean, gaussian, atau median.
    Dipakai buat ngurangin noise. ksize harus ganjil.
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    if method == "mean":
        return cv2.blur(img, (ksize, ksize))
    elif method == "median":
        return cv2.medianBlur(img, ksize)
    else:
        return cv2.GaussianBlur(img, (ksize, ksize), 0)


def sharpen_filter(img, method="unsharp"):
    """
    Filter sharpening: laplacian, high-pass, atau unsharp masking.
    Dipakai buat ngejelasin tepi objek.
    """
    if method == "laplacian":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap = cv2.convertScaleAbs(lap)
        lap_bgr = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(img, 1.0, lap_bgr, 1.0, 0)
    elif method == "highpass":
        kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    else:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)


def edge_detection(img, method="canny"):
    """
    Deteksi tepi: sobel, prewitt, atau canny.
    Hasilnya format grayscale 3-channel biar bisa ditampilin.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "sobel":
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        edges = cv2.convertScaleAbs(mag)
    elif method == "prewitt":
        kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        gx = cv2.filter2D(gray.astype(np.float32), -1, kx)
        gy = cv2.filter2D(gray.astype(np.float32), -1, ky)
        mag = cv2.magnitude(gx, gy)
        edges = cv2.convertScaleAbs(mag)
    else:
        edges = cv2.Canny(gray, 80, 160)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def add_noise(img, noise_type="salt_pepper", amount=0.02):
    """
    Nambahin noise ke citra. Buat testing denoising.
    Salt-pepper: titik putih/hitam acak. Gaussian: distribusi normal.
    """
    out = img.copy()
    if noise_type == "salt_pepper":
        h, w = img.shape[:2]
        n_salt = int(amount * h * w / 2)
        for _ in range(n_salt):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            out[y, x] = [255, 255, 255]
        for _ in range(n_salt):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            out[y, x] = [0, 0, 0]
    else:
        noise = np.random.normal(0, 25, img.shape)
        out = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    return out


def denoise(img, noise_type="salt_pepper"):
    """
    Restorasi citra sesuai jenis noise.
    Median filter buat salt-pepper. NLM buat gaussian.
    """
    if noise_type == "salt_pepper":
        return cv2.medianBlur(img, 5)
    else:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def morphology_op(img, op="opening", ksize=5, iterations=1):
    """
    Operasi morfologi: erosi, dilasi, opening, closing.
    Dijalankan di citra biner. Opening = erosi+dilasi (hilangin noise).
    Closing = dilasi+erosi (tutup lubang).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if op == "erosi":
        result = cv2.erode(binary, kernel, iterations=iterations)
    elif op == "dilasi":
        result = cv2.dilate(binary, kernel, iterations=iterations)
    elif op == "closing":
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def adaptive_threshold(img, block_size=25, c=5):
    """
    Adaptive thresholding sebagai teknik pra-segmentasi.
    Threshold beda tiap region, cocok buat pencahayaan ga merata.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    result = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c
    )
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def histogram_equalization(img):
    """
    Histogram equalization buat perbaikan kontras.
    Dilakuin di channel Y (luminance) biar warna ga berubah.
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


PREPROCESS_FUNCTIONS = {
    "Blur / Smoothing": blur_filter,
    "Sharpening": sharpen_filter,
    "Edge Detection": edge_detection,
    "Denoising": denoise,
    "Operasi Morfologi": morphology_op,
    "Adaptive Threshold": adaptive_threshold,
    "Histogram Equalization": histogram_equalization,
}


# =====================================================================
# TEKNIK PRAPROSES + SISTEM RESEP
# =====================================================================

def to_rgb_passthrough(img, **params):
    """RGB: ga ngubah apa-apa."""
    return img.copy()


def to_grayscale(img, **params):
    """Ubah ke grayscale (tetep 3 channel biar bisa ditampilin)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def adjust_brightness(img, value=0, **params):
    """Atur kecerahan. value -100..100."""
    return cv2.convertScaleAbs(img, alpha=1.0, beta=float(value))


def adjust_contrast(img, value=1.0, **params):
    """Atur kontras. value 0.5..3.0."""
    return cv2.convertScaleAbs(img, alpha=float(value), beta=0)


def denoise_salt_pepper(img, ksize=5, **params):
    """Denoise salt-pepper pake median filter."""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(img, ksize)


def denoise_gaussian(img, strength=10, **params):
    """Denoise gaussian pake Non-Local Means."""
    return cv2.fastNlMeansDenoisingColored(img, None, float(strength), float(strength), 7, 21)


def wiener_filter(img, kernel_size=5, **params):
    """Wiener filter per-channel pake scipy."""
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    channels = cv2.split(img)
    out_channels = []
    for ch in channels:
        filtered = _scipy_wiener(ch.astype(np.float64), (kernel_size, kernel_size))
        filtered = np.nan_to_num(filtered, nan=0.0)
        out_channels.append(np.clip(filtered, 0, 255).astype(np.uint8))
    return cv2.merge(out_channels)


def sobel_edge(img, **params):
    return edge_detection(img, method="sobel")


def prewitt_edge(img, **params):
    return edge_detection(img, method="prewitt")


def canny_edge_filter(img, threshold1=50, threshold2=150, **params):
    """Canny edge detection. threshold1 low, threshold2 high."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, int(threshold1), int(threshold2))
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def mean_filter(img, ksize=5, **params):
    return blur_filter(img, method="mean", ksize=ksize)


def gaussian_filter_fn(img, ksize=5, **params):
    return blur_filter(img, method="gaussian", ksize=ksize)


def median_filter(img, ksize=5, **params):
    return blur_filter(img, method="median", ksize=ksize)


def sharpen(img, **params):
    return sharpen_filter(img, method="unsharp")


def erosion(img, ksize=5, iterations=1, **params):
    return morphology_op(img, op="erosi", ksize=ksize, iterations=iterations)


def dilation(img, ksize=5, iterations=1, **params):
    return morphology_op(img, op="dilasi", ksize=ksize, iterations=iterations)


def opening(img, ksize=5, iterations=1, **params):
    return morphology_op(img, op="opening", ksize=ksize, iterations=iterations)


def closing(img, ksize=5, iterations=1, **params):
    return morphology_op(img, op="closing", ksize=ksize, iterations=iterations)


# Registry semua teknik + parameternya. GUI pake ini buat bikin panel parameter.
TECHNIQUE_REGISTRY = {
    # -- Warna & Intensitas --
    "rgb":        {"category": "Warna", "label": "RGB (Tetap Warna)", "fn": to_rgb_passthrough, "params": {}},
    "grayscale":  {"category": "Warna", "label": "Grayscale", "fn": to_grayscale, "params": {}},
    "brightness": {"category": "Warna", "label": "Brightness", "fn": adjust_brightness,
                   "params": {"value": {"type": "slider", "min": -100, "max": 100, "step": 1, "default": 0}}},
    "contrast":   {"category": "Warna", "label": "Contrast", "fn": adjust_contrast,
                   "params": {"value": {"type": "slider", "min": 0.5, "max": 3.0, "step": 0.1, "default": 1.0}}},
    "histeq":     {"category": "Warna", "label": "Histogram Equalization", "fn": histogram_equalization, "params": {}},

    # -- Filter --
    "mean_filter":     {"category": "Filter", "label": "Mean Filter", "fn": mean_filter,
                         "params": {"ksize": {"type": "choice", "options": [3, 5, 7, 9], "default": 5}}},
    "gaussian_filter": {"category": "Filter", "label": "Gaussian Filter", "fn": gaussian_filter_fn,
                         "params": {"ksize": {"type": "choice", "options": [3, 5, 7, 9], "default": 5}}},
    "median_filter":   {"category": "Filter", "label": "Median Filter", "fn": median_filter,
                         "params": {"ksize": {"type": "choice", "options": [3, 5, 7, 9], "default": 5}}},
    "sharpen":         {"category": "Filter", "label": "Sharpen", "fn": sharpen, "params": {}},

    # -- Restorasi --
    "denoise_salt_pepper": {"category": "Restorasi", "label": "Salt & Pepper Denoise", "fn": denoise_salt_pepper,
                             "params": {"ksize": {"type": "choice", "options": [3, 5, 7], "default": 5}}},
    "denoise_gaussian":    {"category": "Restorasi", "label": "Gaussian Denoise", "fn": denoise_gaussian,
                             "params": {"strength": {"type": "slider", "min": 3, "max": 20, "step": 1, "default": 10}}},
    "wiener":              {"category": "Restorasi", "label": "Wiener Filter", "fn": wiener_filter,
                             "params": {"kernel_size": {"type": "choice", "options": [3, 5, 7], "default": 5}}},

    # -- Deteksi Tepi --
    "sobel":   {"category": "Deteksi Tepi", "label": "Sobel", "fn": sobel_edge, "params": {}},
    "prewitt": {"category": "Deteksi Tepi", "label": "Prewitt", "fn": prewitt_edge, "params": {}},
    "canny":   {"category": "Deteksi Tepi", "label": "Canny Edge Detection", "fn": canny_edge_filter,
                 "params": {"threshold1": {"type": "slider", "min": 10, "max": 200, "step": 5, "default": 50},
                            "threshold2": {"type": "slider", "min": 10, "max": 255, "step": 5, "default": 150}}},

    # -- Morfologi --
    "erosion":  {"category": "Morfologi", "label": "Erosi", "fn": erosion,
                 "params": {"ksize": {"type": "choice", "options": [3, 5, 7, 9], "default": 5}}},
    "dilation": {"category": "Morfologi", "label": "Dilasi", "fn": dilation,
                 "params": {"ksize": {"type": "choice", "options": [3, 5, 7, 9], "default": 5}}},
    "opening":  {"category": "Morfologi", "label": "Opening", "fn": opening,
                 "params": {"ksize": {"type": "choice", "options": [3, 5, 7, 9], "default": 5}}},
    "closing":  {"category": "Morfologi", "label": "Closing", "fn": closing,
                 "params": {"ksize": {"type": "choice", "options": [3, 5, 7, 9], "default": 5}}},
}

TECHNIQUE_CATEGORIES = ["Warna", "Filter", "Restorasi", "Deteksi Tepi", "Morfologi"]


def compute_histogram_image(img, width=320, height=200, bg_color=(255, 255, 255)):
    """Bikin gambar histogram dari citra. Pake OpenCV/numpy, tanpa matplotlib."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.max() + 1e-6)

    canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)
    margin = 10
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    bin_w = max(1, plot_w // 256)

    for i in range(256):
        x = margin + i * bin_w
        bar_h = int(hist_norm[i] * plot_h)
        y_top = margin + (plot_h - bar_h)
        y_bottom = margin + plot_h
        cv2.rectangle(canvas, (x, y_top), (x + bin_w, y_bottom), (90, 60, 20), -1)

    cv2.rectangle(canvas, (margin, margin), (margin + plot_w, margin + plot_h), (180, 180, 180), 1)
    return canvas


def apply_technique(img, technique_id, params=None):
    """Jalankan satu teknik praproses dengan parameter tertentu."""
    if technique_id not in TECHNIQUE_REGISTRY:
        raise ValueError(f"Teknik tidak dikenal: {technique_id}")
    fn = TECHNIQUE_REGISTRY[technique_id]["fn"]
    params = params or {}
    return fn(img, **params)


def apply_recipe(img, recipe):
    """
    Jalankan urutan teknik praproses.
    recipe: list of {"technique_id": str, "params": dict}
    """
    result = img
    for step in recipe:
        result = apply_technique(result, step["technique_id"], step.get("params", {}))
    return result


def get_default_params(technique_id):
    """Dapetin nilai default parameter suatu teknik."""
    spec = TECHNIQUE_REGISTRY[technique_id]["params"]
    return {name: p["default"] for name, p in spec.items()}


# =====================================================================
# SEGMENTASI
# =====================================================================

def segment_threshold_manual(img, thresh_value=127):
    """Threshold manual. Cocok kalo kontras objek-background jelas."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, thresh_value, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def segment_adaptive_threshold(img, block_size=25, c=5):
    """Adaptive threshold. Bagus buat pencahayaan yang ga merata."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def segment_threshold(img):
    """Otsu thresholding. Otomatis nemuin threshold terbaik."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def segment_kmeans(img, k=3):
    """Segmentasi pake K-Means clustering di ruang warna BGR."""
    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img.shape)

    labels_img = labels.reshape(img.shape[:2])
    border_labels = np.concatenate([
        labels_img[0, :], labels_img[-1, :], labels_img[:, 0], labels_img[:, -1]
    ])
    bg_label = np.bincount(border_labels).argmax()
    binary = np.where(labels_img == bg_label, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return segmented, binary


def segment_watershed(img, marker_size=3):
    """Segmentasi watershed pake marker. marker_size buat bersihin noise."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((marker_size, marker_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    markers = np.uint8(markers)
    _, markers = cv2.connectedComponents(markers)
    markers = markers + 1
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_rgb, markers)
    binary = np.zeros_like(gray)
    binary[markers > 1] = 255
    return img_rgb, binary


# 4 metode segmentasi yang tersedia di wizard
SEGMENT_METHODS = ["Thresholding (Otsu)", "Adaptive Threshold", "Watershed", "K-Means Clustering"]

# Parameter tiap metode segmentasi. GUI pake ini buat bikin slider/combobox.
SEGMENT_PARAM_SPECS = {
    "Thresholding (Otsu)": {},
    "Adaptive Threshold": {
        "block_size": {"type": "choice", "options": [15, 21, 25, 31, 35, 45], "default": 25},
        "c": {"type": "slider", "min": 0, "max": 15, "step": 1, "default": 5},
    },
    "Watershed": {
        "marker_size": {"type": "slider", "min": 1, "max": 10, "step": 1, "default": 3},
    },
    "K-Means Clustering": {
        "k": {"type": "choice", "options": [2, 3, 4, 5, 6], "default": 3},
    },
}


def run_segmentation(img, method="Thresholding (Otsu)", **params):
    """
    Jalankan segmentasi. Kembalikan (citra_visual, citra_biner).
    visual = buat ditampilin, binary = buat diproses lebih lanjut.
    """
    if method == "Thresholding (Otsu)":
        binary = segment_threshold(img)
        binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary_vis, binary
    elif method == "Adaptive Threshold":
        block_size = params.get("block_size", 25)
        c = params.get("c", 5)
        binary = segment_adaptive_threshold(img, block_size=block_size, c=c)
        binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary_vis, binary
    elif method == "Watershed":
        marker_size = int(params.get("marker_size", 3))
        vis, binary = segment_watershed(img, marker_size=marker_size)
        return vis, binary
    elif method == "K-Means Clustering":
        k = params.get("k", 3)
        segmented_vis, binary = segment_kmeans(img, k=k)
        return segmented_vis, binary
    else:
        raise ValueError(f"Metode segmentasi tidak dikenal: {method}")


# =====================================================================
# EKSTRAKSI FITUR
# =====================================================================

def extract_shape_features(binary):
    """
    Ekstrak fitur dari setiap kontur di citra biner.
    Fitur: area, perimeter, circularity, vertices, aspect_ratio, solidity.
    Kontur kecil (<0.5% luas gambar) diabaikan sebagai noise.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = binary.shape[:2]
    min_area = 0.005 * h_img * w_img

    features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h != 0 else 0
        extent = area / float(w * h) if (w * h) != 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area != 0 else 0

        features.append({
            "contour": cnt,
            "bbox": (x, y, w, h),
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "vertices": int(vertices),
            "aspect_ratio": float(aspect_ratio),
            "extent": float(extent),
            "solidity": float(solidity),
        })

    features.sort(key=lambda f: f["area"], reverse=True)
    return features


# =====================================================================
# KLASIFIKASI RULE-BASED
# =====================================================================

def classify_shape(feat):
    """
    Klasifikasi pake aturan sederhana berdasarkan circularity dan aspect_ratio.
    Angka threshold udah disesuaikan sama dataset yang dipake.
    """
    circ = feat["circularity"]
    ar = feat["aspect_ratio"]

    if circ >= 0.80:
        return "Lingkaran", min(0.95, circ)
    elif circ >= 0.65 and 0.85 <= ar <= 1.15:
        return "Persegi", 0.85
    elif circ >= 0.45:
        return "Segitiga", 0.75
    else:
        return "Bintang", 0.70


CLASS_LABELS = ["Lingkaran", "Persegi", "Segitiga", "Bintang"]
FEATURE_KEYS = ["area", "perimeter", "circularity", "vertices", "aspect_ratio", "solidity"]


def feature_dict_to_vector(feat):
    """Ubah dict fitur ke list angka buat input k-NN."""
    return [feat[key] for key in FEATURE_KEYS]


def classify_image_multi(img, segment_method="Thresholding (Otsu)", classifier="rule_based",
                          knn_model=None, knn_scaler=None, **seg_params):
    """
    Pipeline lengkap: segmentasi -> ekstraksi fitur -> klasifikasi semua objek.
    Kembalikan (results, binary). results = list of (label, confidence, feature_dict).
    """
    _, binary = run_segmentation(img, method=segment_method, **seg_params)
    features = extract_shape_features(binary)

    results = []
    for feat in features:
        if classifier == "knn":
            if knn_model is None or knn_scaler is None:
                raise ValueError("classifier='knn' butuh knn_model dan knn_scaler.")
            vec = np.array([feature_dict_to_vector(feat)])
            vec_scaled = knn_scaler.transform(vec)
            label = str(knn_model.predict(vec_scaled)[0])
            try:
                proba = knn_model.predict_proba(vec_scaled)[0]
                conf = float(np.max(proba))
            except Exception:
                conf = 1.0
        else:
            label, conf = classify_shape(feat)
        results.append((label, conf, feat))
    return results, binary


# =====================================================================
# k-NN
# =====================================================================

def train_knn(X_train, y_train, n_neighbors=3):
    """Latih k-NN. Fitur dinormalisasi pake StandardScaler."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X_scaled, y_train)
    return knn, scaler


def build_training_data_from_dataset(dataset_dir, recipe=None, segment_method="Thresholding (Otsu)",
                                      **seg_params):
    """
    Bangun data training dari folder dataset.
    Ambil objek terbesar dari tiap gambar sebagai sampel.
    """
    X, y = [], []
    paths = sorted(
        glob.glob(os.path.join(dataset_dir, "*", "*.png"))
        + glob.glob(os.path.join(dataset_dir, "*", "*.jpg"))
    )
    for path in paths:
        label = os.path.basename(os.path.dirname(path))
        img = cv2.imread(path)
        if img is None:
            continue
        processed = apply_recipe(img, recipe) if recipe else img
        _, binary = run_segmentation(processed, method=segment_method, **seg_params)
        feats = extract_shape_features(binary)
        if not feats:
            continue
        X.append(feature_dict_to_vector(feats[0]))
        y.append(label)
    return np.array(X), np.array(y)


# =====================================================================
# SIMPAN & MUAT MODEL
# =====================================================================

def save_model(filepath, preprocess_recipe, segment_method, classifier_type,
               class_labels=None, segment_params=None, knn_model=None, knn_scaler=None,
               extra_info=None):
    """Simpan model ke file .pkl. Model = resep praproses + segmentasi + klasifikasi."""
    data = {
        "format_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preprocess_recipe": preprocess_recipe,
        "segment_method": segment_method,
        "segment_params": segment_params or {},
        "classifier_type": classifier_type,
        "class_labels": class_labels or CLASS_LABELS,
        "knn_model": knn_model,
        "knn_scaler": knn_scaler,
        "extra_info": extra_info or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    return filepath


def load_model(filepath):
    """Muat model dari file .pkl."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def classify_with_model(img, model_data):
    """
    Jalankan pipeline lengkap dari model yang sudah dimuat.
    Kembalikan (results, binary, processed_img).
    """
    recipe = model_data.get("preprocess_recipe", [])
    processed = apply_recipe(img, recipe) if recipe else img.copy()

    seg_method = model_data.get("segment_method", "Thresholding (Otsu)")
    seg_params = model_data.get("segment_params", {}) or {}
    classifier_type = model_data.get("classifier_type", "rule_based")

    results, binary = classify_image_multi(
        processed, segment_method=seg_method, classifier=classifier_type,
        knn_model=model_data.get("knn_model"), knn_scaler=model_data.get("knn_scaler"),
        **seg_params
    )
    return results, binary, processed


def describe_model(model_data):
    """Buat ringkasan pipeline model buat ditampilkan di GUI."""
    recipe = model_data.get("preprocess_recipe", [])
    if recipe:
        step_labels = []
        for step in recipe:
            tid = step["technique_id"]
            label = TECHNIQUE_REGISTRY.get(tid, {}).get("label", tid)
            params = step.get("params", {})
            if params:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                step_labels.append(f"{label} ({param_str})")
            else:
                step_labels.append(label)
        pre_txt = " → ".join(step_labels)
    else:
        pre_txt = "(tidak ada)"

    classifier_type = model_data.get("classifier_type", "rule_based")
    classifier_txt = "Rule-Based" if classifier_type == "rule_based" else "k-NN"

    return (
        f"Praproses: {pre_txt}\n"
        f"Segmentasi: {model_data.get('segment_method', '-')}\n"
        f"Klasifikasi: {classifier_txt}\n"
        f"Dibuat: {model_data.get('created_at', '-')}"
    )


# =====================================================================
# EVALUASI
# =====================================================================

def evaluate_model_on_dataset(model_data, dataset_dir):
    """
    Evaluasi model ke seluruh folder dataset.
    Kembalikan: rows, classes, pred_classes, confusion, metrics, accuracy, total, correct.
    """
    candidates = sorted(
        glob.glob(os.path.join(dataset_dir, "*", "*.png"))
        + glob.glob(os.path.join(dataset_dir, "*", "*.jpg"))
    )
    rows = []
    
    for path in candidates:
        img0 = cv2.imread(path)
        if img0 is None:
            continue
        true_label = os.path.basename(os.path.dirname(path))
        try:
            results, binary, processed = classify_with_model(img0, model_data)
            label = results[0][0] if results else None
        except Exception:
            label = None
        pred = label or "Tidak Dikenali"
        status = "Benar" if pred.lower() == true_label.lower() else "Salah"
        rows.append({"file": os.path.basename(path), "asli": true_label, "prediksi": pred, "status": status})

    classes = sorted(set(r["asli"] for r in rows))
    pred_classes = sorted(set(r["prediksi"] for r in rows))
    
    confusion = {c: {p: 0 for p in pred_classes} for c in classes}
    for r in rows:
        confusion[r["asli"]][r["prediksi"]] += 1

    metrics = []
    for c in classes:
        tp = confusion[c].get(c, 0)
        fp = sum(confusion[other].get(c, 0) for other in classes if other != c)
        fn = sum(confusion[c].get(other, 0) for other in classes if other != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        metrics.append({"kelas": c, "precision": precision, "recall": recall, "f1": f1, "support": tp + fn})

    total = len(rows)
    correct = sum(1 for r in rows if r["status"] == "Benar")
    accuracy = (correct / total * 100) if total else 0.0

    return {
        "rows": rows, 
        "classes": classes, 
        "pred_classes": pred_classes,
        "confusion": confusion, 
        "metrics": metrics, 
        "accuracy": accuracy,
        "total": total, 
        "correct": correct,
    }