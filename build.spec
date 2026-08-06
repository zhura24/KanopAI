# build.spec
# Build: pyinstaller build.spec
#        (atau: pyinstaller --clean --noconfirm build.spec)
#
# CATATAN PENTING sebelum build:
# 1. Jalankan di venv yang SAMA dengan yang dipakai training/inference (torch versi
#    CUDA, ultralytics, rasterio, dst). PyInstaller membundel apa yang terinstall
#    di environment aktif saat build dijalankan.
# 2. best.pt / model .pt TIDAK dibundel ke dalam exe. Taruh di folder yang sama
#    dengan exe hasil build (dist/KanopiAI/), user pilih lewat GUI (panel Inference).
# 3. Build di Windows untuk hasil .exe Windows (PyInstaller tidak cross-compile).
# 4. Setelah build selesai, jalankan ISCC.exe KanopiAI.iss untuk membuat installer
#    .exe yang bisa didistribusikan -- KanopiAI.iss sudah menunjuk ke dist/KanopiAI/.

import sys
import os as _os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

datas    = []
binaries = []
hiddenimports = []

# ============================================================
# ASET STATIS -- file non-Python yang harus ikut ke dalam bundle
# ============================================================

_spec_dir = _os.path.dirname(_os.path.abspath(SPEC))

# logo.ico -- ikon aplikasi di taskbar/title bar (dipakai EXE.icon di bawah)
# Tidak perlu didaftarkan ke datas karena sudah dipakai langsung di EXE().

# Folder 'data' -- kalau ada file default (band_stats contoh, db seed, dll.)
# yang harus ada saat pertama kali app dibuka, daftarkan di sini:
#   _data_dir = _os.path.join(_spec_dir, "data")
#   if _os.path.isdir(_data_dir):
#       datas += [(_data_dir, "data")]

# Folder 'model' -- jika ada model default yang mau dibundle sekalian
# (opsional, bisa besar). Uncomment jika diperlukan:
#   _model_dir = _os.path.join(_spec_dir, "model")
#   if _os.path.isdir(_model_dir):
#       datas += [(_model_dir, "model")]

# ============================================================
# RASTERIO -- butuh GDAL data files + banyak submodule dinamis
# ============================================================
for _pkg in ["rasterio"]:
    _d, _b, _h = collect_all(_pkg)
    datas         += _d
    binaries      += _b
    hiddenimports += _h

# ============================================================
# ULTRALYTICS -- banyak modul di-import dinamis sesuai konfigurasi model
# ============================================================
for _pkg in ["ultralytics"]:
    _d, _b, _h = collect_all(_pkg)
    datas         += _d
    binaries      += _b
    hiddenimports += _h

# ============================================================
# PYPROJ -- wajib collect_all: butuh proj.db (database EPSG) sebagai data
# file, bukan modul Python biasa. Tanpa ini exe crash:
#   "PROJ: proj_create_from_database: Cannot find proj.db"
# Dipakai oleh utils/geospatial_utils.py (GeospatialMetrics), export Excel,
# dan konversi pixel → lat/lon di inference_overlay_handler.
# ============================================================
for _pkg in ["pyproj"]:
    _d, _b, _h = collect_all(_pkg)
    datas         += _d
    binaries      += _b
    hiddenimports += _h

# ============================================================
# OPENPYXL -- dipakai export_result_excel() di inference_engine.py dan
# _export_inference_excel() di inference_overlay_handler.py. Import-nya
# lazy (di dalam fungsi), jadi tidak terdeteksi otomatis oleh PyInstaller.
# ============================================================
hiddenimports += collect_submodules("openpyxl")

# ============================================================
# TORCH -- JANGAN pakai collect_all. Itu narik SEMUA submodule termasuk
# testing internals, distributed training, quantization, dll yang tidak
# dipakai inference single-GPU dan bikin ukuran bengkak parah (GB-an).
# Hook resmi PyInstaller (dari pyinstaller-hooks-contrib) sudah menangani
# kebutuhan dasar torch secara otomatis.
# PERHATIAN: JANGAN exclude torch.distributed -- torch/__init__.py
# meng-import-nya secara internal saat inisialisasi, exclude itu bikin
# exe crash "No module named 'torch.distributed'".
# ============================================================

# ============================================================
# HIDDEN IMPORTS TAMBAHAN -- modul yang sering luput terdeteksi otomatis
# ============================================================
hiddenimports += [
    # --- core KanopAI (pakai sub-package, PyInstaller kadang miss) ---
    "core.inference_engine",
    "core.detection_worker",
    "core.multispectral_worker",
    "core.raster_loader",
    "core.tile_loader",
    "core.tile_manager",
    "core.vector_loader",
    "core.smart_router",
    "core.image_pyramid",

    # --- handlers ---
    "handlers.inference_overlay_handler",
    "handlers.detection_handler",
    "handlers.centroid_handler",
    "handlers.export_handler",
    "handlers.layer_handler",
    "handlers.measurement_handler",
    "handlers.polygon_handler",
    "handlers.view_handler",
    "handlers.multispectral_detection_handler",

    # --- ui ---
    "ui.main_window",
    "ui.raster_viewer",
    "ui.measurement_tool",
    "ui.panels.inference_panel",
    "ui.panels.detection_panel",
    "ui.panels.centroid_panel",
    "ui.panels.display_panel",
    "ui.panels.export_panel",
    "ui.panels.file_panel",
    "ui.panels.measurement_panel",
    "ui.panels.polygon_panel",

    # --- utils ---
    "utils.geospatial_utils",
    "utils.logger_config",
    "utils.coordinate_utils",
    "utils.exception_utils",
    "utils.constants",

    # --- dependencies lain yang sering luput ---
    "cv2",
    "shapefile",          # pyshp -- dipakai save_shapefile / load_shapefile
    "sqlite3",            # dipakai DB manajemen model (inference_engine.py)
    "openpyxl",
    "openpyxl.styles",
    "openpyxl.utils",

    # --- scikit-learn -- dipakai core.hara_regression (eHARA extraction) ---
    "sklearn",
    "sklearn.decomposition",
    "sklearn.decomposition._pca",
    "sklearn.preprocessing",
    "sklearn.pipeline",

    # --- PyQt6 ---
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.QtNetwork",
]

a = Analysis(
    ["main.py"],           # entry point KanopiAI
    pathex=[_spec_dir],    # pastikan folder root project masuk path
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib.tests",
        "numpy.tests",
        # --- paket yang numpang di venv tapi tidak dipakai app ini ---
        # Kalau exe error "ModuleNotFoundError" menyebut salah satu nama
        # di bawah, hapus baris itu dari excludes lalu rebuild.
        "django",
        "tensorflow",
        "pygame",
        "boto3",
        "botocore",
        "sentry_sdk",
        "psycopg_binary",
        "psycopg2",
        "anyio",
        "win32com",
        "IPython",
        "notebook",
        "jupyter",
        "jupyterlab",
        "sphinx",
        "pytest",
        "dns",
        # Catatan: pandas, scipy, lxml, matplotlib SENGAJA TIDAK dikecualikan
        # -- beberapa dependency ultralytics/rasterio kadang diam-diam butuh
        # salah satu dari itu. Tambahkan ke excludes satu per satu sambil tes
        # kalau mau coba memperkecil ukuran lebih lanjut.
    ],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KanopiAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,             # UPX sering bikin torch/cv2 DLL gagal load -- biarkan off
    console=False,         # ganti True saat debugging biar keliatan traceback di terminal
    icon=_os.path.join(_spec_dir, "logo", "logo.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="KanopiAI",       # → dist/KanopiAI/ (cocok dengan KanopiAI.iss)
)
