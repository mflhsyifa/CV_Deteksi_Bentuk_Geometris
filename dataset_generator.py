"""
dataset_generator.py
Bikin dataset sintetis 4 kelas bentuk geometris pake OpenCV.
Ada variasi ukuran, posisi, rotasi, warna, plus noise & blur kecil.
"""

import cv2
import numpy as np
import os
import random
import math

CLASSES = ["lingkaran", "persegi", "segitiga", "bintang"]
IMG_SIZE = 300
IMAGES_PER_CLASS = 20


def random_background(size):
    """Bikin background warna terang dengan sedikit variasi."""
    base_color = np.random.randint(180, 245, size=3)
    bg = np.ones((size, size, 3), dtype=np.uint8) * base_color.astype(np.uint8)
    noise = np.random.randint(-8, 8, (size, size, 3))
    bg = np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)
    return bg


def random_object_color():
    """Warna objek yang kontras sama background (gelap)."""
    return tuple(int(c) for c in np.random.randint(0, 140, size=3))


def rotate_points(points, center, angle_deg):
    """Rotasi titik-titik poligon."""
    angle = math.radians(angle_deg)
    cx, cy = center
    rotated = []
    for x, y in points:
        x0, y0 = x - cx, y - cy
        xr = x0 * math.cos(angle) - y0 * math.sin(angle)
        yr = x0 * math.sin(angle) + y0 * math.cos(angle)
        rotated.append((int(xr + cx), int(yr + cy)))
    return rotated


def draw_star(img, center, r_outer, r_inner, angle_offset, color):
    """Gambar bintang 5 sudut."""
    cx, cy = center
    points = []
    for i in range(10):
        angle = math.radians(90 + angle_offset) - i * math.radians(36)
        r = r_outer if i % 2 == 0 else r_inner
        x = int(cx + r * math.cos(angle))
        y = int(cy - r * math.sin(angle))
        points.append((x, y))
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(img, [pts], color)


def draw_shape(shape, size=IMG_SIZE):
    """Gambar satu bentuk dengan variasi acak."""
    img = random_background(size)
    color = random_object_color()
    margin = int(size * 0.22)
    cx = random.randint(margin, size - margin)
    cy = random.randint(margin, size - margin)
    max_r = min(cx, cy, size - cx, size - cy) - 10
    max_r = max(max_r, 40)
    scale = random.uniform(0.6, 1.0)
    angle = random.uniform(0, 360)

    if shape == "lingkaran":
        r = int(max_r * scale)
        cv2.circle(img, (cx, cy), r, color, -1, lineType=cv2.LINE_AA)

    elif shape == "persegi":
        half = int(max_r * scale * 0.8)
        pts = [(cx - half, cy - half), (cx + half, cy - half),
               (cx + half, cy + half), (cx - half, cy + half)]
        pts = rotate_points(pts, (cx, cy), angle)
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)

    elif shape == "segitiga":
        r = int(max_r * scale)
        pts = []
        for i in range(3):
            a = math.radians(90) + i * math.radians(120)
            x = int(cx + r * math.cos(a))
            y = int(cy - r * math.sin(a))
            pts.append((x, y))
        pts = rotate_points(pts, (cx, cy), angle)
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)

    elif shape == "bintang":
        r_outer = int(max_r * scale)
        r_inner = int(r_outer * 0.45)
        draw_star(img, (cx, cy), r_outer, r_inner, angle, color)

    # Tambah blur & noise biar lebih realistis
    if random.random() < 0.7:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.5:
        noise = np.random.normal(0, 6, img.shape).astype(np.int16)
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    return img


def generate_dataset(output_dir="dataset", per_class=IMAGES_PER_CLASS):
    """Generate semua kelas."""
    for cls in CLASSES:
        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(per_class):
            img = draw_shape(cls)
            fname = os.path.join(cls_dir, f"{cls}_{i+1:02d}.png")
            cv2.imwrite(fname, img)
        print(f"[OK] {per_class} gambar '{cls}' disimpan di {cls_dir}")


if __name__ == "__main__":
    generate_dataset()
    print("Dataset sintetis selesai dibuat di folder ./dataset")