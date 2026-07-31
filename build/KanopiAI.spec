# PyInstaller spec for KanopiAI.
# Build from the repository root with:
#   pyinstaller --clean --noconfirm build/KanopiAI.spec

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPEC).parent.parent

binaries = []
datas = [
    (str(ROOT / "logo" / "logo.ico"), "logo"),
]
hiddenimports = [
    "main",
    "core",
    "handlers",
    "ui",
    "utils",
    "openpyxl",
]

# These packages contain native libraries, Qt plugins, GDAL/PROJ data,
# or dynamically imported modules that PyInstaller cannot always infer.
for package in (
    "PyQt6",
    "numpy",
    "cv2",
    "rasterio",
    "fiona",
    "pyproj",
    "shapely",
    "onnxruntime",
    "ultralytics",
):
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hiddenimports
    except Exception:
        hiddenimports += collect_submodules(package)

# Ship the bundled ONNX models. Multispectral .pt and band_stats.json files
# remain user-selectable external assets, as the UI currently expects paths.
model_dir = ROOT / "model"
if model_dir.exists():
    for model_file in model_dir.glob("*.onnx"):
        datas.append((str(model_file), "model"))

# Avoid duplicate entries introduced by collect_all().
datas = list(dict.fromkeys(datas))
binaries = list(dict.fromkeys(binaries))
hiddenimports = list(dict.fromkeys(hiddenimports))

analysis = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "IPython",
        "jupyter",
        "notebook",
        "pandas",
        "scipy",
        "tensorflow",
    ],
    noarchive=False,
)

pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="KanopiAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ROOT / "logo" / "logo.ico"),
)
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="KanopiAI",
)
