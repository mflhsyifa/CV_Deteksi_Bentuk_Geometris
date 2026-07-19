"""
main_app.py
Mode 1 - "Menu Utama": aplikasi utama yang dibuka pertama kali.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import glob
import os

import image_processing as ip
from wizard_app import WizardFrame, ImageCanvas, apply_screen_fit

APP_TITLE = "Computer Vision - Deteksi Gambar Geometri"
MODELS_DIR = "models"

# Identitas mahasiswa - ganti sesuai data kamu
STUDENT_NAME = "Mufalah Syifa"
STUDENT_NIM = "24041014"
STUDENT_CLASS = "4D"

# Warna-warna yang dipakai biar konsisten di semua halaman
C_NAVY = "#12172b"
C_NAVY_DARK = "#0a0e1a"
C_NAVY_TEXT = "#ffffff"
C_NAVY_MUTED = "#8d97b5"
C_RED = "#dc2626"
C_RED_DARK = "#b91c1c"
C_GREEN = "#1a7f37"
C_GREEN_DARK = "#155e2b"
C_SIDEBAR_BG = "#f5f7fa"
C_BODY_BG = "#ffffff"
C_CARD_BG = "#f5f5f6"
C_CARD_BORDER = "#d5d9e0"


def _colored_button(parent, text, command, color="navy", **kwargs):
    """
    Bikin tombol warna solid biar konsisten tampilannya.
    Ada 3 pilihan warna: navy, red, green.
    """
    palette = {
        "navy": (C_NAVY, C_NAVY_DARK),
        "red": (C_RED, C_RED_DARK),
        "green": (C_GREEN, C_GREEN_DARK),
    }
    bg, active_bg = palette[color]
    defaults = dict(bg=bg, fg="white", font=("Segoe UI", 9, "bold"), relief="flat",
                     pady=5, activebackground=active_bg, activeforeground="white", cursor="hand2", bd=0)
    defaults.update(kwargs)
    return tk.Button(parent, text=text, command=command, **defaults)


class MainApp(tk.Tk):
    """
    Window utama aplikasi.
    Header navy di atas, konten di bawahnya berganti-ganti sesuai mode.
    """
    
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.state('zoomed')  # mulai dalam keadaan maximized
        self.minsize(1000, 650)
        self.configure(bg=C_BODY_BG)

        # Buat ngikutin model terakhir yang disimpan dari wizard
        self.last_saved_model_name = None

        self._build_style()
        self._build_header()

        # Container buat semua tampilan (menu, wizard, evaluasi)
        self.container = tk.Frame(self, bg=C_BODY_BG)
        self.container.pack(fill="both", expand=True)

        self.show_menu()

    def _build_style(self):
        """Setting style dasar buat widget ttk biar keliatan rapi."""
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Sub.TLabel", font=("Segoe UI", 8), foreground="#666666")
        style.configure("CardTitle.TLabel", font=("Segoe UI", 8, "bold"))

    def _build_header(self):
        """Bikin header navy dengan judul dan identitas mahasiswa."""
        header = tk.Frame(self, bg=C_NAVY)
        header.pack(fill="x")
        tk.Label(header, text="✨  COMPUTER VISION - DETEKSI GAMBAR GEOMETRI  ✨",
                 bg=C_NAVY, fg=C_NAVY_TEXT, font=("Segoe UI", 13, "bold")).pack(pady=(10, 2))
        tk.Label(header, text=f"{STUDENT_NAME}   |   {STUDENT_NIM}   |   Kelas {STUDENT_CLASS}",
                 bg=C_NAVY, fg=C_NAVY_MUTED, font=("Segoe UI", 8)).pack(pady=(0, 8))

    def _clear_container(self):
        """Bersihin isi container sebelum ganti tampilan."""
        for w in self.container.winfo_children():
            w.destroy()

    def show_menu(self):
        """Tampilkan halaman Menu Utama."""
        self._clear_container()
        MenuUtamaFrame(self.container, app=self).pack(fill="both", expand=True)

    def show_wizard(self):
        """Tampilkan halaman wizard Buat Model."""
        self._clear_container()
        def _on_saved(path):
            self.last_saved_model_name = os.path.splitext(os.path.basename(path))[0]
        WizardFrame(self.container, on_model_saved=_on_saved, on_back=self.show_menu).pack(fill="both", expand=True)

    def show_evaluation(self):
        """Tampilkan halaman Evaluasi Model."""
        self._clear_container()
        EvaluationFrame(self.container, on_back=self.show_menu).pack(fill="both", expand=True)


class MenuUtamaFrame(tk.Frame):
    """
    Tampilan Menu Utama.
    Isinya: muat gambar, pilih model, jalankan klasifikasi.
    """
    
    def __init__(self, parent, app):
        super().__init__(parent, bg=C_BODY_BG)
        self.app = app
        self.original_img = None
        self.models_map = {}  # nama model -> path file

        self._build_body()
        self.refresh_model_list()

        # Kalau baru selesai simpan model dari wizard, langsung pilih model itu
        if self.app.last_saved_model_name:
            name = self.app.last_saved_model_name
            self.app.last_saved_model_name = None
            if name in self.models_map:
                self.model_var.set(name)
                self._on_model_change()

    def _build_body(self):
        """Buat layout: sidebar kiri + area gambar di kanan."""
        # Sidebar - lebar 260px
        sidebar = tk.Frame(self, bg=C_SIDEBAR_BG, width=260)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        inner = tk.Frame(sidebar, bg=C_SIDEBAR_BG, padx=10, pady=10)
        inner.pack(fill="both", expand=True)

        # ===== 1. Muat Gambar =====
        tk.Label(inner, text="1. Muat Gambar", bg=C_SIDEBAR_BG, font=("Segoe UI", 9, "bold"),
                 fg=C_NAVY).pack(anchor="w")
        ttk.Button(inner, text="📂 Buka Citra...", command=self.load_image_dialog).pack(fill="x", pady=(3, 2))
        self.info_var = tk.StringVar(value="Belum ada citra dimuat.")
        tk.Label(inner, textvariable=self.info_var, bg=C_SIDEBAR_BG, font=("Segoe UI", 8),
                 fg="#555555", wraplength=230, justify="left").pack(anchor="w", pady=(2, 6))

        # ===== 2. Pilih Model =====
        tk.Label(inner, text="2. Pilih Model", bg=C_SIDEBAR_BG, font=("Segoe UI", 9, "bold"),
                 fg=C_NAVY).pack(anchor="w")
        model_row = tk.Frame(inner, bg=C_SIDEBAR_BG)
        model_row.pack(fill="x", pady=(3, 2))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var, state="readonly")
        self.model_combo.pack(side="left", fill="x", expand=True)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        ttk.Button(model_row, text="⟲", width=3, command=self.refresh_model_list).pack(side="left", padx=(4, 0))

        # Card buat nampilin ringkasan pipeline model
        pipeline_container = tk.Frame(inner, bg="#ffffff", highlightbackground=C_CARD_BORDER, highlightthickness=1)
        pipeline_container.pack(fill="x", pady=(4, 8))
        
        pipeline_canvas = tk.Canvas(pipeline_container, bg="#ffffff", height=120, highlightthickness=0)
        pipeline_canvas.pack(side="left", fill="both", expand=True)
        
        pipeline_vsb = ttk.Scrollbar(pipeline_container, orient="vertical", command=pipeline_canvas.yview)
        pipeline_vsb.pack(side="right", fill="y")
        pipeline_canvas.configure(yscrollcommand=pipeline_vsb.set)
        
        pipeline_inner = tk.Frame(pipeline_canvas, bg="#ffffff")
        pipeline_canvas.create_window((0, 0), window=pipeline_inner, anchor="nw")
        
        tk.Label(pipeline_inner, text="Pipeline model ini:", bg="#ffffff", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(4, 2))
        self.pipeline_desc_var = tk.StringVar(value="(pilih model dulu)")
        tk.Label(pipeline_inner, textvariable=self.pipeline_desc_var, bg="#ffffff", font=("Segoe UI", 8),
                 fg="#444444", justify="left", wraplength=220).pack(anchor="w", pady=(0, 4))
        
        def _configure_pipeline_scroll(event):
            pipeline_canvas.configure(scrollregion=pipeline_canvas.bbox("all"))
        pipeline_inner.bind("<Configure>", _configure_pipeline_scroll)

        # ===== 3. Jalankan =====
        tk.Label(inner, text="3. Jalankan", bg=C_SIDEBAR_BG, font=("Segoe UI", 9, "bold"),
                 fg=C_NAVY).pack(anchor="w", pady=(2, 0))
        _colored_button(inner, "▶  Jalankan", self.run_classification, "navy",
                         font=("Segoe UI", 9, "bold"), pady=4).pack(fill="x", pady=(3, 3))

        self.result_summary_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self.result_summary_var, bg=C_SIDEBAR_BG, font=("Segoe UI", 8),
                 fg=C_NAVY, wraplength=230, justify="left").pack(anchor="w", pady=(3, 5))

        tk.Frame(inner, bg=C_CARD_BORDER, height=1).pack(fill="x", pady=(0, 5))

        # Tombol navigasi ke mode lain
        _colored_button(inner, "🛠️  Buat Model", self.app.show_wizard, "navy",
                         font=("Segoe UI", 9, "bold"), pady=6).pack(fill="x", pady=(0, 3))

        eval_btn = tk.Button(inner, text="📊  Evaluasi Model", command=self.app.show_evaluation,
                              bg="#ffffff", fg=C_NAVY, font=("Segoe UI", 9, "bold"),
                              relief="flat", pady=6, highlightbackground=C_NAVY, highlightthickness=1,
                              activebackground=C_SIDEBAR_BG, activeforeground=C_NAVY, cursor="hand2", bd=0)
        eval_btn.pack(fill="x")

        # ===== Area gambar di kanan =====
        right = tk.Frame(self, bg=C_BODY_BG, padx=10, pady=10)
        right.pack(side="left", fill="both", expand=True)
        self.canvas_result = ImageCanvas(right, "Citra", fill_height=True)
        self.canvas_result.pack(fill="both", expand=True)

    def refresh_model_list(self):
        """Refresh daftar model dari folder models/."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        paths = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pkl")))
        self.models_map = {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
        names = list(self.models_map.keys())
        self.model_combo.configure(values=names)
        if names:
            if self.model_var.get() not in names:
                self.model_var.set(names[0])
            self._on_model_change()
        else:
            self.model_var.set("")
            self.pipeline_desc_var.set("Belum ada model. Klik 'Buat Model' di bawah untuk membuatnya.")

    def _on_model_change(self, event=None):
        """Pas milih model, tampilkan deskripsi pipelinenya."""
        name = self.model_var.get()
        if not name or name not in self.models_map:
            return
        try:
            model_data = ip.load_model(self.models_map[name])
            self.pipeline_desc_var.set(ip.describe_model(model_data))
        except Exception as e:
            self.pipeline_desc_var.set(f"Gagal membaca model: {e}")

    def load_image_dialog(self):
        """Buka dialog pilih file gambar."""
        path = filedialog.askopenfilename(
            title="Pilih citra",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Gagal membuka citra.")
            return
        self.original_img = img
        h, w = img.shape[:2]
        self.info_var.set(f"File: {os.path.basename(path)}\nUkuran: {w} x {h} px")
        self.canvas_result.set_image(img)
        self.result_summary_var.set("")

    def run_classification(self):
        """Jalankan klasifikasi pakai model yang dipilih."""
        if self.original_img is None:
            messagebox.showwarning("Peringatan", "Muat citra terlebih dahulu.")
            return
        name = self.model_var.get()
        if not name or name not in self.models_map:
            messagebox.showwarning("Peringatan", "Pilih model terlebih dahulu (atau buat model baru dulu).")
            return

        try:
            model_data = ip.load_model(self.models_map[name])
            results, binary, processed = ip.classify_with_model(self.original_img, model_data)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menjalankan model: {e}")
            return

        # Gambar hasil dengan bounding box dan label
        vis = self.original_img.copy()
        summary = []
        for label, conf, feat in results:
            x, y, w, h = feat["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(vis, label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            summary.append(label)

        self.canvas_result.set_image(vis)
        self.result_summary_var.set(
            f"{len(results)} objek terdeteksi: " + ", ".join(summary) if summary else "Tidak ada objek dikenali."
        )


class EvaluationFrame(tk.Frame):
    """
    Tampilan Evaluasi Model.
    Isinya: pilih model + folder dataset, lihat performa (akurasi, confusion matrix, dll).
    """
    
    def __init__(self, parent, on_back):
        super().__init__(parent, bg=C_BODY_BG)
        self.on_back = on_back
        self.dataset_dir_var = tk.StringVar(value="dataset")
        self.eval_results = None
        self.models_map = {}

        # Bar navigasi atas
        top_bar = tk.Frame(self, bg=C_NAVY_DARK)
        top_bar.pack(fill="x")
        back_btn = tk.Button(top_bar, text="← Kembali ke Menu Utama", command=self.on_back,
                              bg=C_NAVY_DARK, fg="white", font=("Segoe UI", 9, "bold"), relief="flat",
                              activebackground=C_NAVY, activeforeground="white",
                              cursor="hand2", bd=0, padx=12, pady=5)
        back_btn.pack(side="left")
        tk.Label(top_bar, text="📊 Evaluasi Model", bg=C_NAVY_DARK, fg=C_NAVY_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=10)

        body = tk.Frame(self, bg=C_BODY_BG)
        body.pack(fill="both", expand=True)

        # Sidebar kiri
        left = ttk.Frame(body, padding=10)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Evaluasi Model", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))

        # Pilih model
        ttk.Label(left, text="Model:", font=("Segoe UI", 8)).pack(anchor="w")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(left, textvariable=self.model_var, state="readonly", width=28)
        self.model_combo.pack(fill="x", pady=(2, 4))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        # Card pipeline
        pipeline_container = tk.Frame(left, bg="#ffffff", highlightbackground=C_CARD_BORDER, highlightthickness=1)
        pipeline_container.pack(fill="x", pady=(0, 8))
        
        pipeline_canvas = tk.Canvas(pipeline_container, bg="#ffffff", height=120, highlightthickness=0)
        pipeline_canvas.pack(side="left", fill="both", expand=True)
        
        pipeline_vsb = ttk.Scrollbar(pipeline_container, orient="vertical", command=pipeline_canvas.yview)
        pipeline_vsb.pack(side="right", fill="y")
        pipeline_canvas.configure(yscrollcommand=pipeline_vsb.set)
        
        pipeline_inner = tk.Frame(pipeline_canvas, bg="#ffffff")
        pipeline_canvas.create_window((0, 0), window=pipeline_inner, anchor="nw")
        
        tk.Label(pipeline_inner, text="Pipeline model ini:", bg="#ffffff", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(4, 2))
        self.pipeline_desc_var = tk.StringVar(value="(pilih model dulu)")
        tk.Label(pipeline_inner, textvariable=self.pipeline_desc_var, bg="#ffffff", font=("Segoe UI", 8),
                 fg="#444444", justify="left", wraplength=230).pack(anchor="w", pady=(0, 4))
        
        def _configure_scroll(event):
            pipeline_canvas.configure(scrollregion=pipeline_canvas.bbox("all"))
        pipeline_inner.bind("<Configure>", _configure_scroll)

        # Folder dataset
        ttk.Label(left, text="Folder dataset:", font=("Segoe UI", 8)).pack(anchor="w")
        ttk.Label(left, textvariable=self.dataset_dir_var, foreground=C_NAVY, wraplength=230, font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 3))
        ttk.Button(left, text="📁 Pilih Folder...", command=self.choose_dataset_folder).pack(fill="x", pady=(0, 6))

        _colored_button(left, "▶ Jalankan Evaluasi", self.run_evaluation, "navy",
                         font=("Segoe UI", 9, "bold"), pady=5).pack(fill="x", pady=(0, 6))

        # Ringkasan akurasi
        acc_card = tk.Frame(left, bg=C_CARD_BG, padx=10, pady=10)
        acc_card.pack(fill="x")
        tk.Label(acc_card, text="Akurasi total", bg=C_CARD_BG, fg="#555555", font=("Segoe UI", 9)).pack()
        self.accuracy_var = tk.StringVar(value="-")
        tk.Label(acc_card, textvariable=self.accuracy_var, bg=C_CARD_BG, fg=C_NAVY,
                 font=("Segoe UI", 18, "bold")).pack()
        self.count_var = tk.StringVar(value="")
        tk.Label(acc_card, textvariable=self.count_var, bg=C_CARD_BG, fg="#555555", font=("Segoe UI", 8)).pack()

        # Area kanan: tab hasil evaluasi
        right = ttk.Frame(body, padding=10)
        right.pack(side="left", fill="both", expand=True)

        # Tombol-tombol tab
        pill_frame = tk.Frame(right)
        pill_frame.pack(fill="x", pady=(0, 6))
        self.pill_labels = {"detail": "Rincian per Gambar", "confusion": "Confusion Matrix", "metrics": "Metrik per Kelas"}
        self.pill_widgets = {}
        for key, label in self.pill_labels.items():
            b = tk.Button(pill_frame, text=label, font=("Segoe UI", 8), relief="flat", padx=8, pady=3, bd=0,
                          command=lambda k=key: self._show_pill(k))
            b.pack(side="left", padx=(0, 4))
            self.pill_widgets[key] = b

        # Container buat konten tab
        self.content = ttk.Frame(right)
        self.content.pack(fill="both", expand=True)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)
        self.subframes = {}
        for key in self.pill_labels:
            f = ttk.Frame(self.content)
            f.grid(row=0, column=0, sticky="nsew")
            self.subframes[key] = f

        # Tab Detail: rincian per gambar
        cols = ("file", "kelas_asli", "prediksi", "status")
        self.detail_tree = ttk.Treeview(self.subframes["detail"], columns=cols, show="headings", height=20)
        headers = {"file": "File", "kelas_asli": "Kelas Asli", "prediksi": "Prediksi", "status": "Status"}
        for c in cols:
            self.detail_tree.heading(c, text=headers[c])
            self.detail_tree.column(c, width=130, anchor="center")
        self.detail_tree.pack(fill="both", expand=True)

        # Tab Confusion Matrix
        self.confusion_container = ttk.Frame(self.subframes["confusion"])
        self.confusion_container.pack(fill="both", expand=True)

        # Tab Metrics: precision, recall, f1, support
        mcols = ("kelas", "precision", "recall", "f1", "support")
        self.metrics_tree = ttk.Treeview(self.subframes["metrics"], columns=mcols, show="headings", height=20)
        mheaders = {"kelas": "Kelas", "precision": "Precision", "recall": "Recall", "f1": "F1-score", "support": "Support"}
        for c in mcols:
            self.metrics_tree.heading(c, text=mheaders[c])
            self.metrics_tree.column(c, width=130, anchor="center")
        self.metrics_tree.pack(fill="both", expand=True)

        self._show_pill("detail")
        self._refresh_model_list()

    def _refresh_model_list(self):
        """Refresh daftar model dari folder models/."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        paths = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pkl")))
        self.models_map = {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
        names = list(self.models_map.keys())
        self.model_combo.configure(values=names)
        if names:
            self.model_var.set(names[0])
            self._on_model_change()
        else:
            self.pipeline_desc_var.set("Belum ada model.")

    def _on_model_change(self, event=None):
        """Pas milih model, tampilkan deskripsi pipelinenya."""
        name = self.model_var.get()
        if not name or name not in self.models_map:
            return
        try:
            model_data = ip.load_model(self.models_map[name])
            self.pipeline_desc_var.set(ip.describe_model(model_data))
        except Exception as e:
            self.pipeline_desc_var.set(f"Gagal membaca model: {e}")

    def choose_dataset_folder(self):
        """Buka dialog pilih folder dataset."""
        folder = filedialog.askdirectory(title="Pilih folder dataset")
        if folder:
            self.dataset_dir_var.set(folder)

    def _show_pill(self, key):
        """Tampilkan tab sesuai key yang dipilih."""
        self.subframes[key].tkraise()
        for k, b in self.pill_widgets.items():
            b.configure(bg=C_NAVY if k == key else "#eeeeee", fg="#ffffff" if k == key else "#333333")

    def run_evaluation(self):
        """Jalankan evaluasi model ke dataset."""
        name = self.model_var.get()
        if not name or name not in self.models_map:
            messagebox.showwarning("Peringatan", "Pilih model terlebih dahulu.")
            return
        folder = self.dataset_dir_var.get()
        if not (glob.glob(os.path.join(folder, "*", "*.png")) + glob.glob(os.path.join(folder, "*", "*.jpg"))):
            messagebox.showinfo("Dataset kosong", f"Tidak ada gambar di folder: {folder}")
            return

        try:
            model_data = ip.load_model(self.models_map[name])
            self.eval_results = ip.evaluate_model_on_dataset(model_data, folder)
        except Exception as e:
            messagebox.showerror("Error", f"Evaluasi gagal: {e}")
            return

        # Tampilkan hasil di semua tab
        res = self.eval_results
        
        # Detail per gambar
        for row in self.detail_tree.get_children():
            self.detail_tree.delete(row)
        for r in res["rows"]:
            self.detail_tree.insert("", "end", values=(r["file"], r["asli"], r["prediksi"], r["status"]))

        # Confusion Matrix
        for child in self.confusion_container.winfo_children():
            child.destroy()
        
        actual_classes = res["classes"]
        pred_classes = res["pred_classes"]
        
        cols = ["asli"] + pred_classes
        conf_tree = ttk.Treeview(self.confusion_container, columns=cols, show="headings", height=len(actual_classes) + 1)
        conf_tree.heading("asli", text="Asli \\ Prediksi")
        conf_tree.column("asli", width=130, anchor="center")
        for p in pred_classes:
            conf_tree.heading(p, text=p)
            conf_tree.column(p, width=100, anchor="center")
        for c in actual_classes:
            row_values = [c]
            for p in pred_classes:
                row_values.append(res["confusion"][c].get(p, 0))
            conf_tree.insert("", "end", values=row_values)
        conf_tree.pack(fill="both", expand=True)

        # Metrics
        for row in self.metrics_tree.get_children():
            self.metrics_tree.delete(row)
        for m in res["metrics"]:
            self.metrics_tree.insert("", "end", values=(
                m["kelas"], f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}", m["support"]
            ))

        self.accuracy_var.set(f"{res['accuracy']:.1f}%")
        self.count_var.set(f"({res['correct']}/{res['total']} benar, {res['total']} gambar diuji)")


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()