"""
Nuitka Build Script for KanopiAI Application
Compiles Python application to standalone executable

Usage:
    python build_nuitka.py

Requirements:
    - Python 3.11+
    - Nuitka installed
    - All dependencies from requirements.txt
    - MSVC (Microsoft Visual C++) for Windows builds
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Build configuration
BUILD_CONFIG = {
    'app_name': 'KanopiAI',
    'main_script': 'main.py',
    'output_dir': 'dist',
    'icon_file': 'logo/logo.ico',

    # Nuitka options
    'standalone': True,
    'onefile': False,  # Set to True for single .exe (slower startup)
    'enable_console': False,  # Set to True for debugging
    'enable_plugins': [
        'pyqt6',  # PyQt6 support
    ],

    # Modules to include (only essentials)
    'include_packages': [
        'PyQt6',
        'numpy',
        'rasterio',
        'shapely',
        'pyproj',
        'fiona',
        'cv2',
        'onnxruntime',
        'PIL',  # Pillow - for image processing in detection_worker
        'shapefile',  # pyshp - for shapefile export
        'geopandas',  # For vector data handling
    ],

    # Modules to exclude (reduce compilation size and time)
    'nofollow_imports': [
        'torch',  # Not needed - using onnxruntime
        'tensorflow',  # Not needed
        'sympy',  # Not needed - causes out of memory
        'scipy',  # Not needed - 31 MB saved
        'pandas',  # Not needed - 17 MB saved
        'sklearn',  # Not needed - 12 MB saved
        'tokenizers',  # Not needed - 7 MB saved
        'transformers',  # Not needed
        'IPython',  # Interactive shell not needed
        'jupyter',  # Not needed
        'notebook',  # Not needed
        'setuptools',  # Build tools not needed in runtime
        'distutils',  # Build tools not needed
        'pip',  # Package manager not needed
        'doctest',  # Testing not needed
        'pytest',  # Testing not needed
        'unittest',  # Testing not needed
        'test',  # Testing not needed
        'tests',  # Testing not needed
        'catboost',  # Not used
        'numba',  # Not needed
        'branca',  # Mapping library not needed
        'folium',  # Not needed
        'PIL.ImageQt',  # Qt image support not needed
        'matplotlib.tests',  # Tests not needed
        'numpy.tests',  # Tests not needed
        'PyQt6.uic',  # UI compiler not needed at runtime
    ],

    # Data files to include (logo files explicitly)
    'include_data_files': [
        ('logo/logo.png', 'logo/logo.png'),
        ('logo/logo.ico', 'logo/logo.ico'),
    ],

    # Directories to include
    'include_data_dirs': [
        # Not using this - using explicit files above instead
    ],
}


def check_requirements():
    """Check if all build requirements are met"""
    print("=" * 80)
    print("Checking Build Requirements...")
    print("=" * 80)

    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 11):
        print("[WARNING] Python 3.11+ recommended for best compatibility")

    # Check Nuitka installation
    try:
        # Use shell=True for Windows compatibility with conda environments
        result = subprocess.run('nuitka --version',
                              capture_output=True, text=True, check=True, shell=True)
        nuitka_version = result.stdout.strip()
        print(f"Nuitka: {nuitka_version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Nuitka not found!")
        print("Install with: pip install nuitka")
        return False

    # Check for icon file
    if BUILD_CONFIG['icon_file'] and Path(BUILD_CONFIG['icon_file']).exists():
        print(f"Icon: {BUILD_CONFIG['icon_file']} [OK]")
    else:
        print(f"Icon: Not found (will use default)")
        BUILD_CONFIG['icon_file'] = None

    # Check main script
    if not Path(BUILD_CONFIG['main_script']).exists():
        print(f"[ERROR] Main script not found: {BUILD_CONFIG['main_script']}")
        return False

    print(f"Main Script: {BUILD_CONFIG['main_script']} [OK]")

    print("\n[OK] All requirements met!")
    return True


def clean_build():
    """Clean previous build artifacts"""
    print("\n" + "=" * 80)
    print("Cleaning Previous Build...")
    print("=" * 80)

    # Directories to clean
    clean_dirs = [
        BUILD_CONFIG['output_dir'],
        'build',
        f"{BUILD_CONFIG['app_name']}.build",
        f"{BUILD_CONFIG['app_name']}.dist",
        f"{BUILD_CONFIG['app_name']}.onefile-build",
    ]

    for dir_path in clean_dirs:
        if Path(dir_path).exists():
            print(f"Removing: {dir_path}")
            shutil.rmtree(dir_path, ignore_errors=True)

    print("[OK] Clean completed!")


def build_nuitka_command():
    """Build the Nuitka command with all options"""
    cmd = [
        'nuitka',
        BUILD_CONFIG['main_script'],
    ]

    # Output name - use = format for shell compatibility
    cmd.append(f'--output-filename={BUILD_CONFIG["app_name"]}.exe')

    # Standalone mode
    if BUILD_CONFIG['standalone']:
        cmd.append('--standalone')

    # Onefile mode
    if BUILD_CONFIG['onefile']:
        cmd.append('--onefile')

    # Console window (use new format)
    if BUILD_CONFIG['enable_console']:
        cmd.append('--windows-console-mode=attach')
    else:
        cmd.append('--windows-console-mode=disable')

    # Icon - use = format for shell compatibility
    if BUILD_CONFIG['icon_file']:
        cmd.append(f'--windows-icon-from-ico={BUILD_CONFIG["icon_file"]}')

    # Enable plugins
    for plugin in BUILD_CONFIG['enable_plugins']:
        cmd.append(f'--enable-plugin={plugin}')

    # Include packages
    for package in BUILD_CONFIG['include_packages']:
        cmd.append(f'--include-package={package}')

    # Exclude packages (nofollow imports)
    for package in BUILD_CONFIG.get('nofollow_imports', []):
        cmd.append(f'--nofollow-import-to={package}')

    # Include data files
    for src, dest in BUILD_CONFIG['include_data_files']:
        cmd.append(f'--include-data-file={src}={dest}')

    # Include data directories
    for src, dest in BUILD_CONFIG['include_data_dirs']:
        cmd.append(f'--include-data-dir={src}={dest}')

    # Additional optimizations
    cmd.extend([
        '--assume-yes-for-downloads',  # Auto-download dependencies
        '--follow-imports',  # Follow all imports
        '--prefer-source-code',  # Use source when possible
        '--lto=no',  # Disable LTO to reduce memory usage
        '--report=compilation-report.xml',  # Generate compilation report
        # Size optimizations
        '--noinclude-unittest-mode=nofollow',  # Don't include unittest
        '--noinclude-pytest-mode=nofollow',  # Don't include pytest
        '--noinclude-setuptools-mode=nofollow',  # Don't include setuptools
        '--noinclude-IPython-mode=nofollow',  # Don't include IPython
        # '--remove-output',  # Remove build directory after success
    ])

    # Windows specific
    if sys.platform == 'win32':
        cmd.extend([
            '--windows-company-name=KanopiAI',
            '--windows-product-name=KanopiAI',
            '--windows-file-description="Kanopi Palm Detection Application"',
            '--windows-product-version=1.0.0',
        ])

    return cmd


def run_build():
    """Execute the Nuitka build process"""
    print("\n" + "=" * 80)
    print("Building with Nuitka...")
    print("=" * 80)

    # Build command
    cmd = build_nuitka_command()

    # Print command
    print("\nBuild Command:")
    print(" ".join(cmd))
    print("\n" + "-" * 80)

    # Run build
    try:
        # Convert list to string for shell=True compatibility
        cmd_string = " ".join(cmd)
        subprocess.run(cmd_string, check=True, shell=True)
        print("\n" + "=" * 80)
        print("[SUCCESS] Build completed!")
        print("=" * 80)
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"[ERROR] Build failed with code {e.returncode}")
        print("=" * 80)
        return False


def post_build():
    """Post-build steps"""
    print("\n" + "=" * 80)
    print("Post-Build Steps...")
    print("=" * 80)

    # Move output to dist folder
    output_name = f"{BUILD_CONFIG['app_name']}.exe"

    if BUILD_CONFIG['standalone']:
        # Standalone mode creates a .dist folder
        source_dir = f"{BUILD_CONFIG['main_script'].replace('.py', '')}.dist"

        if Path(source_dir).exists():
            # Create dist directory
            dist_dir = Path(BUILD_CONFIG['output_dir'])
            dist_dir.mkdir(exist_ok=True)

            # Move contents
            print(f"Moving output from {source_dir} to {dist_dir}")

            for item in Path(source_dir).iterdir():
                dest = dist_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))

            # Remove source directory
            shutil.rmtree(source_dir)

            print(f"\n[OK] Executable location: {dist_dir / output_name}")
            print(f"[OK] Distribution folder: {dist_dir}")
    else:
        # Single file mode
        if Path(output_name).exists():
            dist_dir = Path(BUILD_CONFIG['output_dir'])
            dist_dir.mkdir(exist_ok=True)

            dest = dist_dir / output_name
            if dest.exists():
                dest.unlink()

            shutil.move(output_name, str(dest))

            print(f"\n[OK] Executable: {dest}")

    print("\n" + "=" * 80)
    print("Build Summary")
    print("=" * 80)
    print(f"Application: {BUILD_CONFIG['app_name']}")
    print(f"Output Directory: {BUILD_CONFIG['output_dir']}")
    print(f"Standalone: {BUILD_CONFIG['standalone']}")
    print(f"Onefile: {BUILD_CONFIG['onefile']}")
    print("\nTo run the application:")
    if BUILD_CONFIG['standalone']:
        print(f"  cd {BUILD_CONFIG['output_dir']}")
        print(f"  {BUILD_CONFIG['app_name']}.exe")
    else:
        print(f"  {BUILD_CONFIG['output_dir']}/{BUILD_CONFIG['app_name']}.exe")
    print("=" * 80)


def main():
    """Main build process"""
    print("\n" + "=" * 80)
    print(f"Nuitka Build Script for {BUILD_CONFIG['app_name']}")
    print("=" * 80)

    # Step 1: Check requirements
    if not check_requirements():
        sys.exit(1)

    # Step 2: Clean previous build
    clean_build()

    # Step 3: Run build
    if not run_build():
        sys.exit(1)

    # Step 4: Post-build steps
    post_build()

    print("\n[SUCCESS] Build process completed!")
    print("\nNext steps:")
    print("1. Test the executable in dist/ folder")
    print("2. Test on a clean Windows machine (no Python installed)")
    print("3. Check if all features work correctly")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Build cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
