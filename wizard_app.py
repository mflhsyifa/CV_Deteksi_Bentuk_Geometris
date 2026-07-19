"""
wizard_app.py
Mode 2 - "Buat Model": wizard 6 langkah untuk menyusun pipeline dari nol.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import glob

import image_processing as ip

DISPLAY_SIZE = 340

# Warna-warna konsisten biar satu tema sama main_app
C_NAVY = "#12172b"
C_NAVY_DARK = "#0a0e1a"
C_NAVY_TEXT = "#ffffff"
C_RED = "#dc2626"
C_RED_DARK = "#b91c1c"
C_GREEN_BG = "#e3f3e6"
C_GREEN_FG = "#1a7f37"
C_GREEN = "#1a7f37"
C_GREEN_DARK = "#155e2b"
C_LOCKED_BG = "#eeeeee"
C_LOCKED_FG = "#a3a3a3"
C_CARD_BG = "#f5f5f6"
C_ACCENT_BG = "#eef0f5"
C_PREVIEW_BG = "#eef0f3"
C_CARD_BORDER = "#d5d9e0"


def apply_screen_fit(root, pref_w=1200, pref_h=750, margin=0.85, min_frac=0.75):
    """
    Atur ukuran window biar selalu muat di layar.
    margin: persentase layar yang dipakai.
    min_frac: ukuran minimum window (persentase dari ukuran awal).
    """
    root.update_idletasks()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    usable_h = max(400, screen_h - 60)  # sisain buat taskbar
    w = min(pref_w, int(screen_w * margin))
    h = min(pref_h, int(usable_h * margin))
    x = max(0, (screen_w - w) // 2)
    y = max(0, (screen_h - h) // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.minsize(int(w * min_frac), int(h * min_frac))


class ImageCanvas(ttk.Frame):
    """
    Widget buat nampilin gambar plus label judul.
    Gambar otomatis diskalakan biar muat di box-nya.
    Ada 3 mode: fill_height=True (memenuhi tinggi), ukuran tetap, atau tinggi tetap.
    """
    DEFAULT_BOX_HEIGHT = 280

    def __init__(self, parent, title, size=None, height=None, fill_height=False):
        super().__init__(parent)
        ttk.Label(self, text=title, font=("Segoe UI", 10, "bold")).pack(pady=(0, 4))

        self.box = tk.Frame(self, bg=C_PREVIEW_BG, highlightbackground=C_CARD_BORDER, highlightthickness=1, bd=0)
        if size is not None:
            self.box.configure(width=size, height=height or int(size * 0.75))
            self.box.pack_propagate(False)
            self.box.pack()
        elif fill_height:
            self.box.pack(fill="both", expand=True)
        else:
            self.box.configure(height=height or self.DEFAULT_BOX_HEIGHT)
            self.box.pack_propagate(False)
            self.box.pack(fill="x", expand=True)

        self.canvas = tk.Label(self.box, bg=C_PREVIEW_BG)
        self.canvas.place(relx=0.5, rely=0.5, anchor="center")
        self._cv_img = None
        self._imgtk = None
        self.box.bind("<Configure>", self._on_resize)

    def _on_resize(self, event=None):
        """Render ulang gambar pas ukuran box berubah."""
        if self._cv_img is not None and event is not None and event.width > 10 and event.height > 10:
            self._render(event.width, event.height)

    def set_image(self, cv_img, max_size=None):
        """Set gambar baru. Nanti di-render otomatis."""
        self._cv_img = cv_img
        w, h = self.box.winfo_width(), self.box.winfo_height()
        if w > 10 and h > 10:
            self._render(w, h)
        else:
            self.after(50, lambda: self._render(self.box.winfo_width(), self.box.winfo_height()))

    def _render(self, box_w, box_h):
        """Render gambar dengan contain-fit (skala proporsional)."""
        if self._cv_img is None:
            return
        img = self._cv_img
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]
        pad = 10
        avail_w, avail_h = max(10, box_w - pad), max(10, box_h - pad)
        scale = min(avail_w / iw, avail_h / ih)
        new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
        self._imgtk = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.configure(image=self._imgtk, text="")

    def clear(self):
        """Kosongkan gambar."""
        self._cv_img = None
        self._imgtk = None
        self.canvas.configure(image="", text="(kosong)")


class WizardFrame(ttk.Frame):
    """
    Frame utama wizard. Ada 6 langkah dengan stepper navigasi.
    Nyimpen semua state pipeline: citra, history, segmentasi, fitur, dll.
    """
    
    def __init__(self, master, on_model_saved=None, on_back=None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_model_saved = on_model_saved  # dipanggil pas model disimpan
        self.on_back = on_back  # dipanggil pas tombol "Kembali" diklik

        # State pipeline
        self.original_img = None
        self.preprocessed_img = None
        self.dataset_dir = "dataset"
        self.segmented_binary = None
        self.segmented_vis = None
        self.current_features = None
        self.last_classification = None
        self.eval_results = None
        self.knn_model = None
        self.knn_scaler = None
        self._eval_dir_manually_set = False

        # History praproses (buat undo/redo)
        self.history_images = []
        self.history_labels = []
        self.history_recipe = []
        self.history_index = 0

        # Stepper
        self.active_index = 0
        self._clear_fns = {}

        # Variabel buat k-NN
        self.knn_k_var = tk.IntVar(value=3)
        self.knn_log_var = tk.StringVar(value="Belum dilatih.")

        self._build_style()
        self._build_layout()

        # Keyboard shortcut undo/redo
        self.bind_all("<Control-z>", lambda e: self.undo_preprocess())
        self.bind_all("<Control-y>", lambda e: self.redo_preprocess())

    def _build_style(self):
        """Setting style dasar buat widget ttk."""
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Sub.TLabel", font=("Segoe UI", 8), foreground="#666666")
        style.configure("CardTitle.TLabel", font=("Segoe UI", 9, "bold"))

    def _build_layout(self):
        """Bikin layout utama: top bar (kalo ada), stepper, content container, status bar."""
        # Bar navigasi atas (kalo dipanggil dari main_app)
        if self.on_back is not None:
            top_bar = tk.Frame(self, bg=C_NAVY_DARK)
            top_bar.pack(fill="x")
            back_btn = tk.Button(top_bar, text="← Kembali ke Menu Utama", command=self.on_back,
                                  bg=C_NAVY_DARK, fg="white", font=("Segoe UI", 9, "bold"), relief="flat",
                                  activebackground=C_NAVY, activeforeground="white",
                                  cursor="hand2", bd=0, padx=12, pady=5)
            back_btn.pack(side="left")
            tk.Label(top_bar, text="🛠️ Buat Model", bg=C_NAVY_DARK, fg="#8d97b5",
                     font=("Segoe UI", 9, "bold")).pack(side="left", padx=10)

        # Stepper bar
        self.stepper_bar = tk.Frame(self, bg="#ffffff")
        self.stepper_bar.pack(fill="x")

        # Container buat konten tiap langkah
        self.content_container = tk.Frame(self)
        self.content_container.pack(fill="both", expand=True, padx=8, pady=8)
        self.content_container.grid_rowconfigure(0, weight=1)
        self.content_container.grid_columnconfigure(0, weight=1)

        # Definisikan 6 langkah
        step_defs = [
            ("Input Citra", self._build_step_load),
            ("Praproses", self._build_step_preprocess),
            ("Segmentasi", self._build_step_segment),
            ("Ekstraksi Fitur", self._build_step_features),
            ("Klasifikasi", self._build_step_classify),
            ("Evaluasi Akhir", self._build_step_eval),
        ]
        frames = []
        for _label, _builder in step_defs:
            f = ttk.Frame(self.content_container)
            f.grid(row=0, column=0, sticky="nsew")
            frames.append(f)
        self.steps = [(label, frame) for (label, _b), frame in zip(step_defs, frames)]

        # Fungsi buat reset hasil di langkah-langkah setelahnya
        self._clear_fns = {
            1: self._clear_preprocess_history,
            2: self._clear_segmentation,
            3: self._clear_features,
            4: self._clear_classification,
            5: self._clear_evaluation,
        }

        # Bangun konten tiap langkah
        for (_label, builder), frame in zip(step_defs, frames):
            builder(frame)

        # Status bar di bawah
        self.status_var = tk.StringVar(value="Silakan muat citra untuk memulai.")
        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w", padding=3, font=("Segoe UI", 8)).pack(fill="x", side="bottom")

        self._render_stepper()
        self.show_step(0)

    def set_status(self, msg):
        """Update pesan di status bar."""
        self.status_var.set(msg)

    # =====================================================================
    # STEPPER NAVIGASI
    # =====================================================================

    def _prereq_ok(self, step_index):
        """Cek apakah syarat buat maju ke step berikutnya udah terpenuhi."""
        if step_index == 0:
            return self.original_img is not None
        if step_index == 1:
            return self.history_index > 0
        if step_index == 2:
            return self.segmented_binary is not None
        if step_index == 3:
            return bool(self.current_features)
        if step_index == 4:
            return bool(self.last_classification)
        return True

    def _prereq_message(self, step_index):
        """Pesan yang muncul kalo syarat belum terpenuhi."""
        messages = {
            0: "Muat citra terlebih dahulu.",
            1: "Terapkan minimal satu teknik praproses terlebih dahulu.",
            2: "Jalankan segmentasi terlebih dahulu.",
            3: "Ekstrak fitur terlebih dahulu (pastikan ada objek terdeteksi).",
            4: "Jalankan klasifikasi terlebih dahulu.",
        }
        return messages.get(step_index, "Selesaikan langkah ini terlebih dahulu.")

    def _render_stepper(self):
        """Gambar ulang stepper sesuai state aktif."""
        for child in self.stepper_bar.winfo_children():
            child.destroy()
        n = len(self.steps)
        for i, (label, _frame) in enumerate(self.steps):
            box = tk.Label(self.stepper_bar, text=f"{i + 1}. {label}",
                            font=("Segoe UI", 9, "bold"), padx=6, pady=7, anchor="center")
            if i < self.active_index:
                # Langkah yang udah lewat -> hijau dan bisa diklik
                box.configure(bg=C_GREEN_BG, fg=C_GREEN_FG, cursor="hand2")
                box.bind("<Button-1>", lambda e, idx=i: self.go_to_step(idx))
            elif i == self.active_index:
                # Langkah aktif -> biru
                box.configure(bg=C_NAVY, fg=C_NAVY_TEXT)
            else:
                # Langkah yang belum kebuka -> abu-abu
                box.configure(bg=C_LOCKED_BG, fg=C_LOCKED_FG)
            box.pack(side="left", fill="both", expand=True)
            # Panah maju di sebelah langkah aktif
            if i == self.active_index and i < n - 1:
                enabled = self._prereq_ok(i)
                arrow = tk.Label(self.stepper_bar, text="➜", font=("Segoe UI", 12, "bold"), width=3,
                                  bg=C_NAVY if enabled else "#cfcfcf",
                                  fg="#ffffff" if enabled else "#8a8a8a",
                                  cursor="hand2" if enabled else "arrow")
                arrow.bind("<Button-1>", lambda e: self.advance_step())
                arrow.pack(side="left", fill="y")

    def go_to_step(self, idx):
        """Pindah ke langkah sebelumnya (bisa klik langsung)."""
        if idx >= self.active_index:
            return
        self.active_index = idx
        self.show_step(idx)

    def advance_step(self):
        """Pindah ke langkah berikutnya (via panah)."""
        if not self._prereq_ok(self.active_index):
            messagebox.showwarning("Belum lengkap", self._prereq_message(self.active_index))
            return
        if self.active_index < len(self.steps) - 1:
            self.active_index += 1
            self.show_step(self.active_index)

    def show_step(self, idx):
        """Tampilkan langkah tertentu."""
        self.steps[idx][1].tkraise()
        if idx == 5:
            self._refresh_eval_summary()
        self._render_stepper()

    def invalidate_from(self, step_index):
        """Reset semua langkah setelah step_index."""
        for idx, fn in self._clear_fns.items():
            if idx > step_index:
                fn()
        self._render_stepper()

    # =====================================================================
    # HELPER: BIKIN KONTROL PARAMETER
    # =====================================================================

    def _build_param_controls(self, parent, param_spec, on_change):
        """
        Bangun kontrol parameter (slider/combobox) sesuai spec.
        Dipakai buat panel parameter di Praproses dan Segmentasi.
        """
        for w in parent.winfo_children():
            w.destroy()
        var_map = {}
        for name, pspec in param_spec.items():
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=name.replace("_", " ").capitalize() + ":", width=12, font=("Segoe UI", 8)).pack(side="left")
            if pspec["type"] == "slider":
                var = tk.DoubleVar(value=pspec["default"])
                tk.Scale(row, from_=pspec["min"], to=pspec["max"], resolution=pspec.get("step", 1),
                         orient="horizontal", variable=var, length=120,
                         command=lambda v: on_change()).pack(side="left", fill="x", expand=True)
            else:
                var = tk.IntVar(value=pspec["default"])
                cb = ttk.Combobox(row, values=pspec["options"], state="readonly", width=8, textvariable=var)
                cb.bind("<<ComboboxSelected>>", lambda e: on_change())
                cb.pack(side="left")
            var_map[name] = var
        return var_map

    def _read_param_values(self, var_map):
        """Baca nilai dari semua parameter."""
        return {name: var.get() for name, var in var_map.items()}

    def _colored_button(self, parent, text, command, color="blue"):
        """Bikin tombol warna solid (biru, merah, hijau)."""
        palette = {
            "blue": (C_NAVY, C_NAVY_DARK),
            "red": (C_RED, C_RED_DARK),
            "green": (C_GREEN, C_GREEN_DARK),
        }
        bg, active_bg = palette[color]
        return tk.Button(parent, text=text, command=command, bg=bg, fg="white",
                          font=("Segoe UI", 9, "bold"), relief="flat", pady=5,
                          activebackground=active_bg, activeforeground="white", cursor="hand2")

    # ================================================================
    # STEP 1: INPUT CITRA
    # ================================================================

    def _build_step_load(self, frame):
        """Bikin tampilan Langkah 1: input citra."""
        left = ttk.Frame(frame, padding=12)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Input Citra", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        ttk.Button(left, text="📂 Buka Citra dari File...", command=self.load_image_dialog).pack(fill="x", pady=3)
        ttk.Button(left, text="🎲 Contoh Acak dari Dataset", command=self.load_random_dataset_image).pack(fill="x", pady=3)
        ttk.Button(left, text="🧪 Generate Dataset Sintetis", command=self.generate_dataset_action).pack(fill="x", pady=3)

        ttk.Separator(left).pack(fill="x", pady=6)
        ttk.Label(left, text="Folder Dataset", style="CardTitle.TLabel").pack(anchor="w")
        self.dataset_dir_var = tk.StringVar(value=self.dataset_dir)
        ttk.Label(left, textvariable=self.dataset_dir_var, wraplength=230, foreground="#1a5fb4", font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 3))
        ttk.Button(left, text="📁 Pilih Folder Dataset...", command=self.choose_dataset_folder).pack(fill="x", pady=2)
        ttk.Label(left, text="Dipakai untuk contoh acak, training k-NN, dan evaluasi",
                  style="Sub.TLabel", wraplength=230, justify="left").pack(anchor="w", pady=(4, 0))

        ttk.Separator(left).pack(fill="x", pady=6)
        self.info_var = tk.StringVar(value="Belum ada citra dimuat.")
        ttk.Label(left, textvariable=self.info_var, wraplength=230, justify="left", font=("Segoe UI", 8)).pack(anchor="w")

        right = ttk.Frame(frame, padding=12)
        right.pack(side="left", fill="both", expand=True)
        self.canvas_load = ImageCanvas(right, "Citra Asli", fill_height=True)
        self.canvas_load.pack(fill="both", expand=True)

    def choose_dataset_folder(self):
        """Pilih folder dataset."""
        folder = filedialog.askdirectory(title="Pilih folder dataset")
        if folder:
            self.dataset_dir = folder
            self.dataset_dir_var.set(folder)
            self.set_status(f"Folder dataset diset ke: {folder}")

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
            messagebox.showerror("Error", "Gagal membuka citra. Pastikan format didukung.")
            return
        self._set_new_image(img, os.path.basename(path))

    def load_random_dataset_image(self):
        """Ambil gambar acak dari dataset."""
        candidates = glob.glob(os.path.join(self.dataset_dir, "*", "*.png")) + \
                     glob.glob(os.path.join(self.dataset_dir, "*", "*.jpg"))
        if not candidates:
            messagebox.showinfo("Dataset kosong", "Folder dataset kosong. Generate dulu atau pilih folder lain.")
            return
        path = np.random.choice(candidates)
        img = cv2.imread(path)
        self._set_new_image(img, os.path.basename(path) + f"  (kelas: {os.path.basename(os.path.dirname(path))})")

    def generate_dataset_action(self):
        """Generate dataset sintetis."""
        try:
            import dataset_generator
            self.set_status("Membuat dataset sintetis...")
            self.update_idletasks()
            dataset_generator.generate_dataset(output_dir=self.dataset_dir)
            messagebox.showinfo("Selesai", f"Dataset sintetis dibuat di folder: {self.dataset_dir}")
            self.set_status("Dataset sintetis berhasil dibuat.")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membuat dataset: {e}")

    def _set_new_image(self, img, name):
        """Set citra baru dan reset state yang bergantung."""
        self.original_img = img
        self.canvas_load.set_image(img)
        h, w = img.shape[:2]
        self.info_var.set(f"File: {name}\nUkuran: {w} x {h} px")
        self.set_status(f"Citra '{name}' dimuat.")
        self.invalidate_from(0)

    # ================================================================
    # STEP 2: PRAPROSES
    # ================================================================

    def _build_step_preprocess(self, frame):
        """Bikin tampilan Langkah 2: praproses."""
        left = ttk.Frame(frame, padding=10)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Praproses Citra", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))

        # Category
        ttk.Label(left, text="Kategori:", font=("Segoe UI", 8)).pack(anchor="w")
        self.pre_category_var = tk.StringVar(value=ip.TECHNIQUE_CATEGORIES[0])
        self.pre_category_combo = ttk.Combobox(left, textvariable=self.pre_category_var,
                                                values=ip.TECHNIQUE_CATEGORIES, state="readonly", width=26)
        self.pre_category_combo.pack(fill="x", pady=(2, 3))
        self.pre_category_combo.bind("<<ComboboxSelected>>", self._on_pre_category_change)

        # Technique
        ttk.Label(left, text="Teknik:", font=("Segoe UI", 8)).pack(anchor="w")
        self.pre_technique_var = tk.StringVar()
        self.pre_technique_combo = ttk.Combobox(left, textvariable=self.pre_technique_var,
                                                 state="readonly", width=26)
        self.pre_technique_combo.pack(fill="x", pady=(2, 3))
        self.pre_technique_combo.bind("<<ComboboxSelected>>", self._on_pre_technique_change)

        # Parameter
        ttk.Label(left, text="Parameter:", font=("Segoe UI", 8)).pack(anchor="w")
        self.pre_param_frame = ttk.Frame(left)
        self.pre_param_frame.pack(fill="x", pady=(2, 3))
        self._pre_label_to_id = {}
        self._pre_param_vars = {}

        # Tombol aksi
        self._colored_button(left, "▶ Terapkan", self.apply_preprocess, "blue").pack(fill="x", pady=(3, 2))
        
        undo_redo_frame = ttk.Frame(left)
        undo_redo_frame.pack(fill="x", pady=2)
        self.btn_undo = ttk.Button(undo_redo_frame, text="↶ Undo", command=self.undo_preprocess)
        self.btn_undo.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.btn_redo = ttk.Button(undo_redo_frame, text="↷ Redo", command=self.redo_preprocess)
        self.btn_redo.pack(side="left", expand=True, fill="x", padx=(2, 0))
        
        self._colored_button(left, "↺ Reset", self.reset_preprocess, "red").pack(fill="x", pady=(2, 3))

        # Riwayat
        ttk.Label(left, text="Riwayat", style="CardTitle.TLabel").pack(anchor="w", pady=(3, 2))
        history_container = tk.Frame(left, bg=C_CARD_BG, highlightbackground=C_CARD_BORDER, highlightthickness=1)
        history_container.pack(fill="x", pady=(0, 0))
        
        history_frame = tk.Frame(history_container, bg=C_CARD_BG)
        history_frame.pack(fill="both", expand=True)
        
        self.history_listbox = tk.Listbox(history_frame, height=6, exportselection=False,
                                           activestyle="none", font=("Segoe UI", 8), bg=C_CARD_BG)
        self.history_listbox.pack(side="left", fill="both", expand=True)
        
        history_scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_listbox.yview)
        history_scrollbar.pack(side="right", fill="y")
        self.history_listbox.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_listbox.bind("<<ListboxSelect>>", self._on_history_select)

        # Preview gambar
        right = ttk.Frame(frame, padding=10)
        right.pack(side="left", fill="both", expand=True)
        
        imgs = tk.Frame(right, bg="#ffffff")
        imgs.pack(fill="both", expand=True)
        imgs.grid_columnconfigure(0, weight=1, uniform="pre_col")
        imgs.grid_columnconfigure(1, weight=1, uniform="pre_col")
        imgs.grid_rowconfigure(0, weight=1)
        self.canvas_pre_before = ImageCanvas(imgs, "Sebelum", fill_height=True)
        self.canvas_pre_before.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.canvas_pre_after = ImageCanvas(imgs, "Sesudah (preview)", fill_height=True)
        self.canvas_pre_after.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        self._on_pre_category_change()

    def _on_pre_category_change(self, event=None):
        """Pas kategori berubah, update daftar teknik."""
        cat = self.pre_category_var.get()
        techs = [(tid, spec["label"]) for tid, spec in ip.TECHNIQUE_REGISTRY.items() if spec["category"] == cat]
        self._pre_label_to_id = {label: tid for tid, label in techs}
        self.pre_technique_combo.configure(values=[label for _tid, label in techs])
        if techs:
            self.pre_technique_var.set(techs[0][1])
        self._on_pre_technique_change()

    def _on_pre_technique_change(self, event=None):
        """Pas teknik berubah, update parameter."""
        tid = self._current_pre_technique_id()
        if tid is None:
            return
        spec = ip.TECHNIQUE_REGISTRY[tid]["params"]
        self._pre_param_vars = self._build_param_controls(self.pre_param_frame, spec, self._update_preprocess_preview)
        self._update_preprocess_preview()

    def _current_pre_technique_id(self):
        """Dapetin ID teknik yang lagi dipilih."""
        return self._pre_label_to_id.get(self.pre_technique_var.get())

    def _update_preprocess_preview(self):
        """Update preview hasil praproses."""
        if self.original_img is None:
            return
        tid = self._current_pre_technique_id()
        if tid is None:
            return
        params = self._read_param_values(self._pre_param_vars)
        try:
            preview = ip.apply_technique(self.preprocessed_img, tid, params)
        except Exception:
            return
        self.canvas_pre_after.set_image(preview)

    def apply_preprocess(self):
        """Terapkan teknik praproses yang dipilih."""
        if self.original_img is None:
            messagebox.showwarning("Peringatan", "Muat citra terlebih dahulu.")
            return
        tid = self._current_pre_technique_id()
        if tid is None:
            return
        params = self._read_param_values(self._pre_param_vars)
        try:
            result = ip.apply_technique(self.preprocessed_img, tid, params)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menerapkan teknik: {e}")
            return

        # Simpan ke history
        self.history_images = self.history_images[: self.history_index + 1]
        self.history_labels = self.history_labels[: self.history_index + 1]
        self.history_recipe = self.history_recipe[: self.history_index + 1]

        label = ip.TECHNIQUE_REGISTRY[tid]["label"]
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            label_full = f"{label} ({param_str})"
        else:
            label_full = label

        self.history_images.append(result.copy())
        self.history_labels.append(label_full)
        self.history_recipe.append({"technique_id": tid, "params": params})
        self.history_index = len(self.history_images) - 1

        self.preprocessed_img = result
        self.canvas_pre_before.set_image(result)
        self.canvas_pre_after.set_image(result)
        self._refresh_history_ui()
        self.set_status(f"Teknik '{label}' diterapkan.")
        self.invalidate_from(1)

    def undo_preprocess(self):
        """Undo satu langkah praproses."""
        if self.history_index <= 0:
            self.set_status("Sudah di citra asli, tidak ada yang bisa di-undo.")
            return
        self.history_index -= 1
        self._jump_to_history_index(self.history_index)
        self.set_status(f"Undo -> {self.history_labels[self.history_index]}")

    def redo_preprocess(self):
        """Redo satu langkah praproses."""
        if self.history_index >= len(self.history_images) - 1:
            self.set_status("Sudah di titik terbaru, tidak ada yang bisa di-redo.")
            return
        self.history_index += 1
        self._jump_to_history_index(self.history_index)
        self.set_status(f"Redo -> {self.history_labels[self.history_index]}")

    def _on_history_select(self, event=None):
        """Pas klik item di history, langsung lompat ke state itu."""
        if getattr(self, "_syncing_history", False):
            return
        selection = self.history_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        if idx == self.history_index:
            return
        self.history_index = idx
        self._jump_to_history_index(idx)
        self.set_status(f"Lompat ke: {self.history_labels[idx]}")

    def _jump_to_history_index(self, idx):
        """Lompat ke state history tertentu."""
        self.preprocessed_img = self.history_images[idx].copy()
        self.canvas_pre_before.set_image(self.preprocessed_img)
        self.canvas_pre_after.set_image(self.preprocessed_img)
        self._refresh_history_ui()
        self.invalidate_from(1)

    def _refresh_history_ui(self):
        """Refresh tampilan listbox history."""
        self._syncing_history = True
        try:
            self.history_listbox.delete(0, tk.END)
            for i, label in enumerate(self.history_labels):
                prefix = "●" if i == self.history_index else " "
                self.history_listbox.insert(tk.END, f"{prefix} {i}. {label}")
            self.history_listbox.selection_clear(0, tk.END)
            self.history_listbox.selection_set(self.history_index)
            self.history_listbox.see(self.history_index)
        finally:
            self._syncing_history = False
        self.btn_undo.configure(state="normal" if self.history_index > 0 else "disabled")
        self.btn_redo.configure(state="normal" if self.history_index < len(self.history_images) - 1 else "disabled")

    def _clear_preprocess_history(self):
        """Reset history praproses."""
        if self.original_img is None:
            self.history_images, self.history_labels, self.history_recipe = [], [], []
            self.history_index = 0
            return
        self.history_images = [self.original_img.copy()]
        self.history_labels = ["Citra Asli"]
        self.history_recipe = [None]
        self.history_index = 0
        self.preprocessed_img = self.original_img.copy()
        self.canvas_pre_before.set_image(self.original_img)
        self.canvas_pre_after.set_image(self.preprocessed_img)
        self._refresh_history_ui()

    def reset_preprocess(self):
        """Reset semua praproses ke citra asli."""
        if self.original_img is None:
            return
        self._clear_preprocess_history()
        self.set_status("Praproses direset ke citra asli.")
        self.invalidate_from(1)

    def _current_recipe(self):
        """Dapetin recipe yang aktif (history yang udah dipilih)."""
        return [s for s in self.history_recipe[1:self.history_index + 1] if s is not None]

    # ================================================================
    # STEP 3: SEGMENTASI
    # ================================================================

    def _build_step_segment(self, frame):
        """Bikin tampilan Langkah 3: segmentasi."""
        left = ttk.Frame(frame, padding=10)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Segmentasi Citra", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))

        # 4 metode segmentasi
        seg_methods = ["Thresholding (Otsu)", "Adaptive Threshold", "Watershed", "K-Means Clustering"]
        self.seg_method_var = tk.StringVar(value=seg_methods[0])
        ttk.Label(left, text="Metode:", font=("Segoe UI", 8)).pack(anchor="w")
        seg_combo = ttk.Combobox(left, textvariable=self.seg_method_var, values=seg_methods,
                                  state="readonly", width=26)
        seg_combo.pack(fill="x", pady=(2, 4))
        seg_combo.bind("<<ComboboxSelected>>", self._on_seg_method_change)

        # Parameter
        ttk.Label(left, text="Parameter:", font=("Segoe UI", 8)).pack(anchor="w")
        self.seg_param_frame = ttk.Frame(left)
        self.seg_param_frame.pack(fill="x", pady=(2, 6))
        self.seg_param_vars = {}

        # Tombol
        self._colored_button(left, "▶ Jalankan Segmentasi", self.run_segmentation, "blue").pack(fill="x", pady=(4, 2))

        # Preview gambar
        right = tk.Frame(frame, bg="#ffffff", padx=10, pady=10)
        right.pack(side="left", fill="both", expand=True)
        imgs = tk.Frame(right, bg="#ffffff")
        imgs.pack(fill="both", expand=True)
        imgs.grid_columnconfigure(0, weight=1, uniform="seg_col")
        imgs.grid_columnconfigure(1, weight=1, uniform="seg_col")
        imgs.grid_rowconfigure(0, weight=1)
        self.canvas_seg_before = ImageCanvas(imgs, "Input Segmentasi", fill_height=True)
        self.canvas_seg_before.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.canvas_seg_after = ImageCanvas(imgs, "Hasil Biner", fill_height=True)
        self.canvas_seg_after.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # Langsung munculin parameter
        self._on_seg_method_change()

    def _on_seg_method_change(self, event=None):
        """Pas metode segmentasi berubah, update parameter."""
        method = self.seg_method_var.get()
        spec = ip.SEGMENT_PARAM_SPECS.get(method, {})
        self.seg_param_vars = self._build_param_controls(self.seg_param_frame, spec, lambda: None)

    def _current_seg_params(self):
        """Baca nilai parameter segmentasi."""
        return self._read_param_values(self.seg_param_vars)

    def run_segmentation(self):
        """Jalankan segmentasi dengan metode dan parameter yang dipilih."""
        if self.original_img is None:
            messagebox.showwarning("Peringatan", "Muat citra terlebih dahulu.")
            return
        img = self.preprocessed_img
        method = self.seg_method_var.get()
        params = self._current_seg_params()

        self.canvas_seg_before.set_image(img)
        try:
            vis, binary = ip.run_segmentation(img, method=method, **params)
        except Exception as e:
            messagebox.showerror("Error", f"Segmentasi gagal: {e}")
            return

        self.segmented_binary = binary
        self.segmented_vis = vis
        self.canvas_seg_after.set_image(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        self.set_status(f"Segmentasi '{method}' selesai.")
        self.invalidate_from(2)

    def _clear_segmentation(self):
        """Reset hasil segmentasi."""
        if self.segmented_binary is not None:
            self.segmented_binary = None
            self.segmented_vis = None
            self.canvas_seg_before.clear()
            self.canvas_seg_after.clear()

    # ================================================================
    # STEP 4: EKSTRAKSI FITUR
    # ================================================================

    def _build_step_features(self, frame):
        """Bikin tampilan Langkah 4: ekstraksi fitur."""
        left = ttk.Frame(frame, padding=12)
        left.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="Representasi & Deskripsi Citra", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))
        self._colored_button(left, "▶ Ekstrak Fitur", self.run_feature_extraction, "blue").pack(anchor="w", pady=(0, 6))

        # Tabel fitur
        columns = ("obj", "area", "perimeter", "circularity", "aspect_ratio", "solidity")
        self.feature_tree = ttk.Treeview(left, columns=columns, show="headings", height=10)
        headers = {"obj": "Objek #", "area": "Luas", "perimeter": "Keliling", "circularity": "Circularity",
                   "aspect_ratio": "Aspect Ratio", "solidity": "Solidity"}
        for c in columns:
            self.feature_tree.heading(c, text=headers[c])
            self.feature_tree.column(c, width=90, anchor="center")
        self.feature_tree.pack(fill="both", expand=True, pady=4)

        right = ttk.Frame(frame, padding=12)
        right.pack(side="left", fill="both", expand=True)
        self.canvas_feat = ImageCanvas(right, "Kontur Terdeteksi (semua objek)", fill_height=True)
        self.canvas_feat.pack(fill="both", expand=True)

    def run_feature_extraction(self):
        """Ekstrak fitur dari hasil segmentasi."""
        if self.segmented_binary is None:
            messagebox.showwarning("Peringatan", "Jalankan segmentasi terlebih dahulu.")
            return
        features = ip.extract_shape_features(self.segmented_binary)
        self.current_features = features

        # Tampilkan di tabel
        for row in self.feature_tree.get_children():
            self.feature_tree.delete(row)
        for i, f in enumerate(features, start=1):
            self.feature_tree.insert("", "end", values=(
                i, f"{f['area']:.0f}", f"{f['perimeter']:.1f}", f"{f['circularity']:.3f}",
                f"{f['aspect_ratio']:.2f}", f"{f['solidity']:.3f}"
            ))

        # Gambar kontur
        vis = cv2.cvtColor(self.segmented_binary, cv2.COLOR_GRAY2BGR)
        for f in features:
            cv2.drawContours(vis, [f["contour"]], -1, (0, 255, 0), 2)
            x, y, w, h = f["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
        self.canvas_feat.set_image(vis)

        self.set_status(f"{len(features)} objek terdeteksi.")
        if not features:
            messagebox.showinfo("Info", "Tidak ada objek terdeteksi. Coba ubah metode segmentasi.")
        self.invalidate_from(3)

    def _clear_features(self):
        """Reset hasil ekstraksi fitur."""
        if self.current_features is not None:
            self.current_features = None
            for row in self.feature_tree.get_children():
                self.feature_tree.delete(row)
            self.canvas_feat.clear()

    # ================================================================
    # STEP 5: KLASIFIKASI
    # ================================================================

    def _build_step_classify(self, frame):
        """Bikin tampilan Langkah 5: klasifikasi."""
        left = ttk.Frame(frame, padding=(10, 8))
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Pengenalan Pola", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))

        # Pilih metode klasifikasi
        ttk.Label(left, text="Metode:", font=("Segoe UI", 8)).pack(anchor="w")
        self.classifier_var = tk.StringVar(value="Rule-Based")
        classifier_combo = ttk.Combobox(left, textvariable=self.classifier_var,
                                         values=["Rule-Based", "k-NN"], state="readonly", width=26)
        classifier_combo.pack(fill="x", pady=(2, 3))
        classifier_combo.bind("<<ComboboxSelected>>", self._on_classifier_change)

        # Panel k-NN (muncul kalo milih k-NN)
        self.knn_train_frame = ttk.Frame(left)
        ttk.Label(self.knn_train_frame, text="Latih k-NN dari folder dataset saat ini:",
                  style="Sub.TLabel", wraplength=230, justify="left").pack(anchor="w")
        k_row = ttk.Frame(self.knn_train_frame)
        k_row.pack(fill="x", pady=2)
        ttk.Label(k_row, text="n_neighbors (k):", font=("Segoe UI", 8)).pack(side="left")
        ttk.Spinbox(k_row, from_=1, to=15, textvariable=self.knn_k_var, width=5).pack(side="left", padx=6)
        self._colored_button(self.knn_train_frame, "🎓 Latih k-NN Sekarang", self.train_knn_action, "blue").pack(fill="x", pady=2)
        ttk.Label(self.knn_train_frame, textvariable=self.knn_log_var, style="Sub.TLabel",
                  wraplength=230, justify="left").pack(anchor="w", pady=(2, 0))

        # Tombol klasifikasi
        self._colored_button(left, "▶ Terapkan Klasifikasi", self.run_classification, "blue").pack(fill="x", pady=(4, 3))

        ttk.Separator(left).pack(fill="x", pady=4)
        ttk.Label(left, text="Hasil", style="CardTitle.TLabel").pack(anchor="w")
        self.result_var = tk.StringVar(value="-")
        ttk.Label(left, textvariable=self.result_var, font=("Segoe UI", 10, "bold"),
                  foreground=C_GREEN_FG, wraplength=230, justify="left").pack(anchor="w", pady=3)

        # Card aturan rule-based
        rules_card = tk.Frame(left, bg=C_CARD_BG, padx=8, pady=6)
        rules_card.pack(fill="x", pady=(3, 4))
        tk.Label(rules_card, text="Basis Aturan (Rule-Based)", bg=C_CARD_BG,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        rules_text = (
            "Circularity ≥ 0.80 → Lingkaran\n"
            "Circularity ≥ 0.65 & AR 0.85-1.15 → Persegi\n"
            "Circularity ≥ 0.45 → Segitiga\n"
            "Selain itu → Bintang"
        )
        tk.Label(rules_card, text=rules_text, bg=C_CARD_BG, font=("Segoe UI", 7),
                 fg="#444444", justify="left").pack(anchor="w", pady=(2, 0))

        # Detail per objek
        detail_card = tk.Frame(left, bg="#ffffff", padx=8, pady=6,
                                highlightbackground=C_CARD_BORDER, highlightthickness=1)
        detail_card.pack(fill="x", pady=(0, 4))
        tk.Label(detail_card, text="Detail Perhitungan per Objek", bg="#ffffff",
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.detail_var = tk.StringVar(value="(jalankan klasifikasi dulu)")
        tk.Label(detail_card, textvariable=self.detail_var, bg="#ffffff", font=("Segoe UI", 7),
                 fg="#444444", justify="left", wraplength=230).pack(anchor="w", pady=(2, 0))

        # Preview gambar hasil klasifikasi
        right = ttk.Frame(frame, padding=10)
        right.pack(side="left", fill="both", expand=True)
        self.canvas_class_img = ImageCanvas(right, "Objek Diklasifikasikan", fill_height=True)
        self.canvas_class_img.pack(fill="both", expand=True)

        self._on_classifier_change()

    def _on_classifier_change(self, event=None):
        """Pas metode klasifikasi berubah, tampilkan/sembunyikan panel k-NN."""
        if self.classifier_var.get() == "k-NN":
            self.knn_train_frame.pack(fill="x", pady=(0, 6))
        else:
            self.knn_train_frame.pack_forget()
        self._clear_classification()
        self.invalidate_from(4)

    def train_knn_action(self):
        """Latih model k-NN dari folder dataset."""
        if self.original_img is None:
            messagebox.showwarning("Peringatan", "Muat citra & siapkan folder dataset dulu di Langkah 1.")
            return
        folder = self.dataset_dir
        candidates = glob.glob(os.path.join(folder, "*", "*.png")) + glob.glob(os.path.join(folder, "*", "*.jpg"))
        if not candidates:
            messagebox.showinfo("Dataset kosong", f"Tidak ada gambar di folder: {folder}")
            return
        try:
            self.knn_log_var.set("[1/3] Memuat & mengekstrak fitur dataset...")
            self.update_idletasks()
            recipe = self._current_recipe()
            seg_method = self.seg_method_var.get()
            seg_params = self._current_seg_params()
            X, y = ip.build_training_data_from_dataset(folder, recipe=recipe, segment_method=seg_method, **seg_params)

            self.knn_log_var.set(f"[2/3] Melatih k-NN (k={self.knn_k_var.get()}, {len(X)} sampel)...")
            self.update_idletasks()
            self.knn_model, self.knn_scaler = ip.train_knn(X, y, n_neighbors=self.knn_k_var.get())

            n_classes = len(set(y))
            self.knn_log_var.set(f"[3/3] Selesai. {len(X)} sampel, {n_classes} kelas: {sorted(set(y))}")
            self.set_status("k-NN berhasil dilatih.")
            self._clear_classification()
            self.invalidate_from(4)
        except Exception as e:
            self.knn_log_var.set(f"Gagal melatih: {e}")
            messagebox.showerror("Error", f"Gagal melatih k-NN: {e}")

    def run_classification(self):
        """Jalankan klasifikasi semua objek."""
        if not self.current_features:
            messagebox.showwarning("Peringatan", "Lakukan ekstraksi fitur terlebih dahulu.")
            return
        use_knn = self.classifier_var.get() == "k-NN"
        if use_knn and self.knn_model is None:
            messagebox.showwarning("Peringatan", "Latih model k-NN terlebih dahulu.")
            return

        classifier_type = "knn" if use_knn else "rule_based"
        seg_method = self.seg_method_var.get()
        seg_params = self._current_seg_params()
        try:
            results, binary = ip.classify_image_multi(
                self.preprocessed_img, segment_method=seg_method, classifier=classifier_type,
                knn_model=self.knn_model, knn_scaler=self.knn_scaler, **seg_params
            )
        except Exception as e:
            messagebox.showerror("Error", f"Klasifikasi gagal: {e}")
            return

        self.last_classification = results
        vis = self.preprocessed_img.copy()
        summary = []
        detail_lines = []
        for i, (label, conf, feat) in enumerate(results, start=1):
            x, y, w, h = feat["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(vis, label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            summary.append(label)
            detail_lines.append(
                f"Objek {i}: {label}\n"
                f"  Area={feat['area']:.0f}  Perimeter={feat['perimeter']:.1f}\n"
                f"  Circularity={feat['circularity']:.3f}  AR={feat['aspect_ratio']:.2f}"
            )

        self.canvas_class_img.set_image(vis)
        self.result_var.set(", ".join(summary) if summary else "Tidak ada objek dikenali")

        # Tampilkan detail (maks 4 objek)
        MAX_DETAIL = 4
        if len(detail_lines) > MAX_DETAIL:
            shown = detail_lines[:MAX_DETAIL]
            shown.append(f"... dan {len(detail_lines) - MAX_DETAIL} objek lainnya")
            self.detail_var.set("\n\n".join(shown))
        else:
            self.detail_var.set("\n\n".join(detail_lines) if detail_lines else "(tidak ada objek)")
        self.set_status(f"{len(results)} objek diklasifikasikan ({'k-NN' if use_knn else 'Rule-Based'}).")
        self.invalidate_from(4)

    def _clear_classification(self):
        """Reset hasil klasifikasi."""
        if self.last_classification is not None:
            self.last_classification = None
            self.result_var.set("-")
            self.canvas_class_img.clear()
        if hasattr(self, "detail_var"):
            self.detail_var.set("(jalankan klasifikasi dulu)")

    # ================================================================
    # STEP 6: EVALUASI AKHIR
    # ================================================================

    def _build_step_eval(self, frame):
        """Bikin tampilan Langkah 6: evaluasi akhir dan simpan model."""
        left = ttk.Frame(frame, padding=(10, 8))
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Evaluasi Akhir", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))

        # Folder dataset uji
        ttk.Label(left, text="Folder dataset uji:", style="CardTitle.TLabel").pack(anchor="w")
        self.eval_dataset_dir_var = tk.StringVar(value=self.dataset_dir)
        ttk.Label(left, textvariable=self.eval_dataset_dir_var, wraplength=230, foreground="#1a5fb4", font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 2))
        ttk.Button(left, text="📁 Ganti Folder...", command=self.choose_eval_dataset_folder).pack(fill="x", pady=(0, 6))

        # Ringkasan pipeline
        summary_container = tk.Frame(left, bg=C_CARD_BG, highlightbackground=C_CARD_BORDER, highlightthickness=1)
        summary_container.pack(fill="x", pady=(0, 6))
        
        summary_canvas = tk.Canvas(summary_container, bg=C_CARD_BG, height=120, highlightthickness=0)
        summary_canvas.pack(side="left", fill="both", expand=True)
        
        summary_vsb = ttk.Scrollbar(summary_container, orient="vertical", command=summary_canvas.yview)
        summary_vsb.pack(side="right", fill="y")
        summary_canvas.configure(yscrollcommand=summary_vsb.set)
        
        summary_inner = tk.Frame(summary_canvas, bg=C_CARD_BG)
        summary_canvas.create_window((0, 0), window=summary_inner, anchor="nw")
        
        tk.Label(summary_inner, text="Pipeline yang digunakan", bg=C_CARD_BG, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(3, 2))
        self.eval_summary_var = tk.StringVar(value="-")
        tk.Label(summary_inner, textvariable=self.eval_summary_var, bg=C_CARD_BG, justify="left",
                 anchor="w", font=("Segoe UI", 8), wraplength=220).pack(anchor="w", pady=(0, 3))
        
        def _config_summary_scroll(event):
            summary_canvas.configure(scrollregion=summary_canvas.bbox("all"))
        summary_inner.bind("<Configure>", _config_summary_scroll)

        # Tombol evaluasi
        self._colored_button(left, "▶ Jalankan Evaluasi", self.run_evaluation, "blue").pack(fill="x", pady=(0, 4))

        # Card akurasi
        acc_card = tk.Frame(left, bg=C_ACCENT_BG, padx=8, pady=6)
        acc_card.pack(fill="x")
        tk.Label(acc_card, text="Akurasi total", bg=C_ACCENT_BG, fg="#555555", font=("Segoe UI", 9)).pack()
        self.eval_accuracy_var = tk.StringVar(value="-")
        tk.Label(acc_card, textvariable=self.eval_accuracy_var, bg=C_ACCENT_BG, fg=C_NAVY,
                 font=("Segoe UI", 18, "bold")).pack()
        self.eval_count_var = tk.StringVar(value="")
        tk.Label(acc_card, textvariable=self.eval_count_var, bg=C_ACCENT_BG, fg="#555555", font=("Segoe UI", 8)).pack()

        ttk.Separator(left).pack(fill="x", pady=6)
        ttk.Label(left, text="Simpan Model", style="CardTitle.TLabel").pack(anchor="w")
        self.model_name_var = tk.StringVar(value="model_geometris")
        ttk.Entry(left, textvariable=self.model_name_var).pack(fill="x", pady=2)
        self._colored_button(left, "💾 Simpan Model", self.save_current_model, "green").pack(fill="x", pady=(2, 0))

        # Area kanan: tab hasil evaluasi
        right = ttk.Frame(frame, padding=10)
        right.pack(side="left", fill="both", expand=True)

        pill_frame = tk.Frame(right)
        pill_frame.pack(fill="x", pady=(0, 6))
        self.eval_pill_labels = {"detail": "Rincian per Gambar", "confusion": "Confusion Matrix", "metrics": "Metrik per Kelas"}
        self.eval_pill_widgets = {}
        for key, label in self.eval_pill_labels.items():
            b = tk.Button(pill_frame, text=label, font=("Segoe UI", 8), relief="flat", padx=8, pady=3,
                          command=lambda k=key: self._show_eval_pill(k))
            b.pack(side="left", padx=(0, 4))
            self.eval_pill_widgets[key] = b

        self.eval_content = ttk.Frame(right)
        self.eval_content.pack(fill="both", expand=True)
        self.eval_content.grid_rowconfigure(0, weight=1)
        self.eval_content.grid_columnconfigure(0, weight=1)
        self.eval_subframes = {}
        for key in self.eval_pill_labels:
            f = ttk.Frame(self.eval_content)
            f.grid(row=0, column=0, sticky="nsew")
            self.eval_subframes[key] = f

        # Tab Detail
        cols = ("file", "kelas_asli", "prediksi", "status")
        self.eval_detail_tree = ttk.Treeview(self.eval_subframes["detail"], columns=cols, show="headings", height=20)
        headers = {"file": "File", "kelas_asli": "Kelas Asli", "prediksi": "Prediksi", "status": "Status"}
        for c in cols:
            self.eval_detail_tree.heading(c, text=headers[c])
            self.eval_detail_tree.column(c, width=130, anchor="center")
        self.eval_detail_tree.pack(fill="both", expand=True)

        # Tab Confusion Matrix
        self.eval_confusion_container = ttk.Frame(self.eval_subframes["confusion"])
        self.eval_confusion_container.pack(fill="both", expand=True)

        # Tab Metrics
        mcols = ("kelas", "precision", "recall", "f1", "support")
        self.eval_metrics_tree = ttk.Treeview(self.eval_subframes["metrics"], columns=mcols, show="headings", height=20)
        mheaders = {"kelas": "Kelas", "precision": "Precision", "recall": "Recall", "f1": "F1-score", "support": "Support"}
        for c in mcols:
            self.eval_metrics_tree.heading(c, text=mheaders[c])
            self.eval_metrics_tree.column(c, width=130, anchor="center")
        self.eval_metrics_tree.pack(fill="both", expand=True)

        self._show_eval_pill("detail")

    def choose_eval_dataset_folder(self):
        """Pilih folder dataset buat evaluasi."""
        folder = filedialog.askdirectory(title="Pilih folder dataset untuk evaluasi")
        if folder:
            self.eval_dataset_dir_var.set(folder)
            self._eval_dir_manually_set = True

    def _show_eval_pill(self, key):
        """Tampilkan tab evaluasi sesuai key."""
        self.eval_subframes[key].tkraise()
        for k, b in self.eval_pill_widgets.items():
            b.configure(bg=C_NAVY if k == key else C_LOCKED_BG, fg="#ffffff" if k == key else "#333333")

    def _refresh_eval_summary(self):
        """Refresh ringkasan pipeline di evaluasi."""
        if not self._eval_dir_manually_set:
            self.eval_dataset_dir_var.set(self.dataset_dir)
        self.eval_summary_var.set(self._get_pipeline_summary_text())

    def _get_pipeline_summary_text(self):
        """Buat teks ringkasan pipeline yang dipakai."""
        recipe = self._current_recipe()
        if not recipe:
            pre_txt = "(belum ada teknik diterapkan)"
        else:
            parts = []
            for step in recipe:
                label = ip.TECHNIQUE_REGISTRY[step["technique_id"]]["label"]
                params = step.get("params", {})
                if params:
                    parts.append(f"{label} ({', '.join(f'{k}={v}' for k, v in params.items())})")
                else:
                    parts.append(label)
            pre_txt = " → ".join(parts)

        classifier_txt = self.classifier_var.get() if hasattr(self, "classifier_var") else "-"
        return (f"Praproses:\n{pre_txt}\n\n"
                f"Segmentasi:\n{self.seg_method_var.get()}\n\n"
                f"Klasifikasi:\n{classifier_txt}")

    def run_evaluation(self):
        """Jalankan evaluasi ke folder dataset uji."""
        eval_dir = self.eval_dataset_dir_var.get()
        if not (glob.glob(os.path.join(eval_dir, "*", "*.png")) + glob.glob(os.path.join(eval_dir, "*", "*.jpg"))):
            messagebox.showinfo("Dataset kosong", f"Tidak ada gambar di folder: {eval_dir}")
            return

        use_knn = self.classifier_var.get() == "k-NN"
        if use_knn and self.knn_model is None:
            messagebox.showwarning("Peringatan", "Latih model k-NN dulu di Langkah 5 sebelum evaluasi.")
            return

        # Bikin model virtual di memory (belum disimpan ke file)
        virtual_model = {
            "preprocess_recipe": self._current_recipe(),
            "segment_method": self.seg_method_var.get(),
            "segment_params": self._current_seg_params(),
            "classifier_type": "knn" if use_knn else "rule_based",
            "knn_model": self.knn_model,
            "knn_scaler": self.knn_scaler,
        }

        self.set_status("Menjalankan evaluasi...")
        self.update_idletasks()

        self.eval_results = ip.evaluate_model_on_dataset(virtual_model, eval_dir)
        self._populate_eval_ui()
        self.eval_accuracy_var.set(f"{self.eval_results['accuracy']:.1f}%")
        self.eval_count_var.set(
            f"({self.eval_results['correct']}/{self.eval_results['total']} benar, "
            f"{self.eval_results['total']} gambar diuji)"
        )
        self.set_status(f"Evaluasi selesai. Akurasi {self.eval_results['accuracy']:.1f}% "
                         f"dari {self.eval_results['total']} gambar.")

    def _populate_eval_ui(self):
        """Tampilkan hasil evaluasi di semua tab."""
        res = self.eval_results
        
        # Detail
        for row in self.eval_detail_tree.get_children():
            self.eval_detail_tree.delete(row)
        for r in res["rows"]:
            self.eval_detail_tree.insert("", "end", values=(r["file"], r["asli"], r["prediksi"], r["status"]))

        # Confusion Matrix
        for child in self.eval_confusion_container.winfo_children():
            child.destroy()
        
        actual_classes = res["classes"]
        pred_classes = res["pred_classes"]
        
        cols = ["asli"] + pred_classes
        conf_tree = ttk.Treeview(self.eval_confusion_container, columns=cols, show="headings", height=len(actual_classes) + 1)
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
        for row in self.eval_metrics_tree.get_children():
            self.eval_metrics_tree.delete(row)
        for m in res["metrics"]:
            self.eval_metrics_tree.insert("", "end", values=(
                m["kelas"], f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}", m["support"]
            ))

    def _reset_eval_ui(self):
        """Reset UI evaluasi."""
        if hasattr(self, "eval_detail_tree"):
            for row in self.eval_detail_tree.get_children():
                self.eval_detail_tree.delete(row)
        if hasattr(self, "eval_confusion_container"):
            for child in self.eval_confusion_container.winfo_children():
                child.destroy()
        if hasattr(self, "eval_metrics_tree"):
            for row in self.eval_metrics_tree.get_children():
                self.eval_metrics_tree.delete(row)
        if hasattr(self, "eval_accuracy_var"):
            self.eval_accuracy_var.set("-")
            self.eval_count_var.set("")

    def _clear_evaluation(self):
        """Reset hasil evaluasi."""
        if self.eval_results is not None:
            self.eval_results = None
            self._reset_eval_ui()

    def save_current_model(self):
        """Simpan model ke file .pkl."""
        if self.eval_results is None:
            messagebox.showwarning("Peringatan", "Jalankan evaluasi terlebih dahulu sebelum menyimpan model.")
            return
        name = self.model_name_var.get().strip()
        if not name:
            messagebox.showwarning("Peringatan", "Isi nama model terlebih dahulu.")
            return
        if not name.endswith(".pkl"):
            name += ".pkl"

        os.makedirs("models", exist_ok=True)
        filepath = os.path.join("models", name)

        use_knn = self.classifier_var.get() == "k-NN"
        classifier_type = "knn" if use_knn else "rule_based"
        if use_knn and self.knn_model is None:
            messagebox.showwarning("Peringatan", "Model k-NN belum dilatih.")
            return

        try:
            ip.save_model(
                filepath,
                preprocess_recipe=self._current_recipe(),
                segment_method=self.seg_method_var.get(),
                classifier_type=classifier_type,
                segment_params=self._current_seg_params(),
                knn_model=self.knn_model if use_knn else None,
                knn_scaler=self.knn_scaler if use_knn else None,
                class_labels=ip.CLASS_LABELS,
                extra_info={"accuracy": self.eval_results["accuracy"], "dataset_dir": self.eval_dataset_dir_var.get()},
            )
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan model: {e}")
            return

        messagebox.showinfo("Model Disimpan", f"Model berhasil disimpan:\n{filepath}")
        self.set_status(f"Model disimpan: {filepath}")
        if self.on_model_saved:
            self.on_model_saved(filepath)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Buat Model - Computer Vision Deteksi Bentuk Geometris")
    apply_screen_fit(root, 1200, 750)
    WizardFrame(root).pack(fill="both", expand=True)
    root.mainloop()