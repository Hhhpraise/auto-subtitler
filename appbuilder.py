# build_app.py - Script to create standalone executable

import subprocess
import sys
import os


def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("✅ PyInstaller already installed")
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def create_spec_file():
    """Create a custom spec file for better control"""
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['subtitler_gui.py'],  # Your main Python file
    pathex=[],
    binaries=[],
    datas=[
        # Include any data files your app needs
        # ('path/to/data', 'destination/in/app'),
    ],
    hiddenimports=[
        'torch',
        'whisper',
        'moviepy.editor',
        'speech_recognition',
        'googletrans',
        'tkinter',
        'queue',
        'threading',
        'hashlib',
        'json',
        'requests',
        'urllib.request',
        'subprocess',
        'webbrowser',
        'wave'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused modules to reduce size
        'matplotlib',
        'numpy.random._examples',
        'test',
        'unittest',
        'doctest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AutoSubtitler',  # Name of your executable
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress executable (optional)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico'  # Add your icon file here (optional)
)
"""

    with open('../transcriper/AutoSubtitler.spec', 'w') as f:
        f.write(spec_content)
    print("✅ Spec file created: AutoSubtitler.spec")


def build_executable():
    """Build the executable using PyInstaller"""
    print("🔨 Building executable...")

    # Build using spec file
    cmd = [
        "pyinstaller",
        "--clean",  # Clean cache
        "AutoSubtitler.spec"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Build completed!")
        print("📁 Executable location: dist/AutoSubtitler.exe")
        print("💡 You can now distribute the 'dist' folder")
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")


def build_onefile():
    """Alternative: Build as single file (slower startup but portable)"""
    print("🔨 Building single-file executable...")

    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",  # No console window
        "--name=AutoSubtitler",
        "--clean",
        "subtitler_gui.py"  # Your main file
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Single-file build completed!")
        print("📁 Executable: dist/AutoSubtitler.exe")
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")


if __name__ == "__main__":
    print("🎬 Auto Subtitler - Build Script")
    print("=" * 40)

    # Install PyInstaller
    install_pyinstaller()

    choice = input(
        "\nChoose build type:\n1. Directory build (faster startup)\n2. Single file (more portable)\n3. Custom spec file\nEnter choice (1-3): ")

    if choice == "1":
        # Quick directory build
        cmd = [
            "pyinstaller",
            "--windowed",
            "--name=AutoSubtitler",
            "--clean",
            "subtitler_gui.py"
        ]
        subprocess.run(cmd)

    elif choice == "2":
        build_onefile()

    elif choice == "3":
        create_spec_file()
        build_executable()

    else:
        print("Invalid choice")

    print("\n🎉 Build process completed!")
    print("💡 Tips:")
    print("   - Test the executable before distributing")
    print("   - Include any required DLL files if needed")
    print("   - Consider antivirus false positives with single-file builds")