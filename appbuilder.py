# build_app.py - Enhanced script to create standalone executable, installer, and deploy to GitHub

import subprocess
import sys
import os
import shutil
import json
import tempfile
import glob
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'torch', 'whisper', 'moviepy', 'speech_recognition',
        'googletrans', 'requests', 'opencv-python'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ Missing packages installed")
    else:
        print("✅ All dependencies are installed")


def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("✅ PyInstaller already installed")
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def install_innosetup():
    """Check if Inno Setup is available or provide download instructions"""
    try:
        # Check if ISCC is available in PATH using where command (more reliable on Windows)
        result = subprocess.run(["where", "ISCC"], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("✅ Inno Setup is installed")
            # Verify it actually works by checking version
            version_result = subprocess.run(["ISCC", "/?"], capture_output=True, text=True, shell=True)
            if version_result.returncode == 0:
                return True
            else:
                print("❌ ISCC found but not working properly")
                return False
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Alternative check: Look for common installation paths
    common_paths = [
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
    ]

    for path in common_paths:
        if os.path.exists(path):
            print(f"✅ Inno Setup found at: {path}")
            # Add it to the PATH for this session
            innosetup_dir = os.path.dirname(path)
            os.environ['PATH'] = innosetup_dir + os.pathsep + os.environ['PATH']
            return True

    print("❌ Inno Setup not found")
    print("🔥 Please download and install Inno Setup from:")
    print("   https://jrsoftware.org/isdl.php")
    print("   After installation, make sure to add it to your PATH")
    print("   Or restart PyCharm after installation")
    return False


def find_executable():
    """Find the executable in various possible locations"""
    possible_paths = [
        "dist/AutoSubtitler.exe",
        "dist/AutoSubtitler/AutoSubtitler.exe",
        "build/AutoSubtitler/AutoSubtitler.exe",
        "dist/main/AutoSubtitler.exe"
    ]

    # Also search for any .exe files in dist and build directories
    search_patterns = [
        "dist/**/*.exe",
        "build/**/*.exe"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found executable at: {path}")
            return path

    # If not found in specific paths, search more broadly
    for pattern in search_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            if "AutoSubtitler" in file_path:
                print(f"✅ Found executable at: {file_path}")
                return file_path

    print("❌ Could not find executable")
    return None


def create_spec_file():
    """Create a custom spec file for better control"""
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['subtitler_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include any data files your app needs
        # ('path/to/data', 'destination/in/app'),
    ],
    hiddenimports=[
        'torch', 'whisper', 'moviepy.editor', 'speech_recognition', 
        'googletrans', 'tkinter', 'queue', 'threading', 'hashlib', 
        'json', 'requests', 'urllib.request', 'subprocess', 'webbrowser', 
        'wave', 'numpy', 'cv2', 'PIL', 'pydub', 'whisper.transcribe',
        'whisper.tokenizer', 'whisper.utils', 'whisper.decoding',
        'whisper.model', 'whisper.audio'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused modules to reduce size
        'matplotlib', 'numpy.random._examples', 'test', 'unittest', 
        'doctest', 'pandas', 'scipy', 'sklearn', 'tensorflow', 'keras'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Add necessary binaries (DLLs)
import os
import sys

python_dir = os.path.dirname(sys.executable)
python_dll = os.path.join(python_dir, 'python39.dll')
if os.path.exists(python_dll):
    a.binaries += [('python39.dll', python_dll, 'BINARY')]

# Add other potential DLLs
for dll in ['vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll']:
    dll_path = os.path.join(python_dir, dll)
    if os.path.exists(dll_path):
        a.binaries += [(dll, dll_path, 'BINARY')]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AutoSubtitler',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico'  # Optional: add an icon file
)
"""

    with open('AutoSubtitler.spec', 'w') as f:
        f.write(spec_content)
    print("✅ Spec file created: AutoSubtitler.spec")


def build_executable():
    """Build the executable using PyInstaller"""
    print("🔨 Building executable...")

    # Build using spec file
    cmd = [
        "pyinstaller",
        "--clean",
        "AutoSubtitler.spec"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Build completed!")

        # Find where the executable was actually created
        exe_path = find_executable()
        if exe_path:
            print(f"📁 Executable location: {exe_path}")
            return True
        else:
            print("⚠️  Build completed but could not locate executable")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False


def build_onefile():
    """Build as single file (slower startup but portable) - IMPROVED VERSION"""
    print("🔨 Building single-file executable...")

    # Get Python directory
    python_dir = os.path.dirname(sys.executable)

    # Build the command with ALL necessary options for a truly standalone executable
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name=AutoSubtitler",
        "--clean",
        # Hidden imports for whisper and dependencies
        "--hidden-import", "whisper",
        "--hidden-import", "whisper.transcribe",
        "--hidden-import", "whisper.tokenizer",
        "--hidden-import", "whisper.utils",
        "--hidden-import", "whisper.decoding",
        "--hidden-import", "whisper.model",
        "--hidden-import", "whisper.audio",
        "--hidden-import", "torch",
        "--hidden-import", "torchaudio",
        "--hidden-import", "numpy",
        "--hidden-import", "tkinter",
        "--hidden-import", "queue",
        "--hidden-import", "threading",
        "--hidden-import", "moviepy.editor",
        "--hidden-import", "speech_recognition",
        "--hidden-import", "googletrans",
        "--hidden-import", "requests",
        "--hidden-import", "cv2",
        "--hidden-import", "PIL",
        "--hidden-import", "pydub",
        # Collect all submodules for critical packages
        "--collect-submodules", "whisper",
        "--collect-submodules", "torch",
        "--collect-submodules", "torchaudio",
        # Copy metadata for packages that need it
        "--copy-metadata", "torch",
        "--copy-metadata", "torchaudio",
        "--copy-metadata", "whisper",
        # Add data files that might be needed
        "--collect-data", "torch",
        "--collect-data", "torchaudio",
        "--collect-data", "whisper",
        "subtitler_gui.py"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Single-file build completed!")

        # Find where the executable was actually created
        exe_path = find_executable()
        if exe_path:
            file_size = os.path.getsize(exe_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"📁 Executable: {exe_path}")
            print(f"📏 File size: {file_size_mb:.1f} MB")

            if file_size_mb > 1500:  # > 1.5GB
                print("⚠️  Large executable detected!")
                print("💡 Consider using the optimized build (Option 10) to reduce size")

            return True
        else:
            print("⚠️  Build completed but could not locate executable")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False


def build_onefile_optimized():
    """Build optimized single file (smaller size) - RECOMMENDED FOR LARGE APPS"""
    print("🔨 Building optimized single-file executable...")
    print("💡 This build excludes some components to reduce size")

    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name=AutoSubtitler",
        "--clean",
        "--optimize=2",  # Python optimization
        # Essential hidden imports only
        "--hidden-import", "whisper.transcribe",
        "--hidden-import", "whisper.tokenizer",
        "--hidden-import", "whisper.utils",
        "--hidden-import", "whisper.decoding",
        "--hidden-import", "whisper.model",
        "--hidden-import", "whisper.audio",
        "--hidden-import", "tkinter",
        "--hidden-import", "queue",
        "--hidden-import", "threading",
        # Exclude large unnecessary modules
        "--exclude-module", "matplotlib",
        "--exclude-module", "scipy",
        "--exclude-module", "pandas",
        "--exclude-module", "sklearn",
        "--exclude-module", "tensorflow",
        "--exclude-module", "keras",
        "--exclude-module", "jupyter",
        "--exclude-module", "IPython",
        "--exclude-module", "notebook",
        "--exclude-module", "sphinx",
        "--exclude-module", "pytest",
        "--exclude-module", "setuptools",
        "--exclude-module", "wheel",
        # Minimal data collection (only essential)
        "--copy-metadata", "whisper",
        "--collect-data", "whisper",
        "subtitler_gui.py"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Optimized build completed!")

        exe_path = find_executable()
        if exe_path:
            file_size = os.path.getsize(exe_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"📁 Executable: {exe_path}")
            print(f"📏 Optimized size: {file_size_mb:.1f} MB")
            return True
        else:
            print("⚠️  Build completed but could not locate executable")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False


def check_vc_redist():
    """Check if Microsoft Visual C++ Redistributable is installed"""
    print("🔍 Checking for Microsoft Visual C++ Redistributable...")

    # Common registry keys for VC++ Redistributable
    vc_redist_keys = [
        r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
    ]

    try:
        import winreg
        for key_path in vc_redist_keys:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                winreg.CloseKey(key)
                print("✅ Microsoft Visual C++ Redistributable is installed")
                return True
            except FileNotFoundError:
                continue
    except ImportError:
        # Not on Windows, or no winreg module
        pass

    print("❌ Microsoft Visual C++ Redistributable is not installed")
    print("💡 Download from: https://aka.ms/vs/16/release/vc_redist.x64.exe")
    return False


def create_installer():
    """Create an installer using Inno Setup - FIXED VERSION WITH DISK SPANNING"""
    # Find the executable first
    exe_path = find_executable()
    if not exe_path:
        print("❌ Executable not found")
        print("💡 Please build the executable first (options 1, 2, or 3)")
        return False

    # Convert to absolute path
    exe_path = os.path.abspath(exe_path)

    print(f"📁 Using executable: {exe_path}")

    # Check if the executable actually exists
    if not os.path.exists(exe_path):
        print(f"❌ Executable not found at: {exe_path}")
        return False

    # Check file size and warn user
    file_size = os.path.getsize(exe_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"📏 Executable size: {file_size_mb:.1f} MB")

    if file_size > 2000000000:  # ~2GB
        print("⚠️  Large executable detected! Enabling disk spanning for installer...")
        disk_spanning = "DiskSpanning=yes"
    else:
        disk_spanning = "; DiskSpanning not needed for smaller files"

    # For single-file builds, we don't need separate DLLs since they're embedded
    # Create a simpler ISS file that only includes the executable
    iss_content = f"""
; Script generated by the Auto Subtitler Build Script
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "Auto Subtitler"
#define MyAppVersion "2.0"
#define MyAppPublisher "Hhhpraise"
#define MyAppURL "https://github.com/hhhpraise/auto-subtitler"
#define MyAppExeName "AutoSubtitler.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}}
AppName={{#MyAppName}}
AppVersion={{#MyAppVersion}}
AppVerName={{#MyAppName}} {{#MyAppVersion}}
AppPublisher={{#MyAppPublisher}}
AppPublisherURL={{#MyAppURL}}
AppSupportURL={{#MyAppURL}}
AppUpdatesURL={{#MyAppURL}}
DefaultDirName={{autopf}}\\{{#MyAppName}}
DisableProgramGroupPage=yes
; Remove the following line to run in administrative install mode (install for all users.)
PrivilegesRequired=lowest
OutputDir=installer
OutputBaseFilename=AutoSubtitler_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
; Enable disk spanning for large files (>2GB)
{disk_spanning}
; Set disk size if spanning is enabled
DiskSliceSize=max

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked

[Files]
Source: "{exe_path}"; DestDir: "{{app}}"; Flags: ignoreversion
; NOTE: For single-file builds, all DLLs are embedded in the executable
; If you're using a directory build, uncomment and adjust the following lines:
; Source: "dist\\AutoSubtitler\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{autoprograms}}\\{{#MyAppName}}"; Filename: "{{app}}\\{{#MyAppExeName}}"
Name: "{{autodesktop}}\\{{#MyAppName}}"; Filename: "{{app}}\\{{#MyAppExeName}}"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\{{#MyAppExeName}}"; Description: "{{cm:LaunchProgram,{{#StringChange(MyAppName, '&', '&&')}}}}"; Flags: nowait postinstall skipifsilent

[Registry]
; Add any registry entries if needed for file associations, etc.

[UninstallDelete]
; Clean up any files created during runtime
Type: filesandordirs; Name: "{{app}}"
"""

    # Create installer directory if it doesn't exist
    os.makedirs("installer", exist_ok=True)

    with open("installer.iss", "w") as f:
        f.write(iss_content)

    print("📝 Created Inno Setup script: installer.iss")
    print(f"📁 Using executable from: {exe_path}")

    # Now check if Inno Setup is available
    if not install_innosetup():
        print("⚠️  Inno Setup not found, but ISS file has been created")
        print("💡 You can manually compile it with Inno Setup Compiler")
        return False

    # Try multiple ways to run ISCC
    iscc_commands = [
        ["ISCC", "installer.iss"],  # Try PATH first
        [r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe", "installer.iss"],  # Try common path
        [r"C:\Program Files\Inno Setup 6\ISCC.exe", "installer.iss"]  # Try another common path
    ]

    for cmd in iscc_commands:
        try:
            print(f"🔄 Trying: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, shell=True, capture_output=True, text=True)
            print("✅ Installer created successfully!")
            print("📁 Installer location: installer/AutoSubtitler_Setup.exe")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            continue
        except FileNotFoundError:
            continue

    print("❌ Installer compilation failed: Could not find or run ISCC")
    print("💡 Try restarting PyCharm or manually running:")
    print("   ISCC installer.iss")
    return False


def prepare_github_release():
    """Prepare files for GitHub release"""
    print("📦 Preparing files for GitHub release...")

    # Create a release directory
    release_dir = "github_release"
    if os.path.exists(release_dir):
        shutil.rmtree(release_dir)
    os.makedirs(release_dir)

    # Copy the executable
    exe_path = find_executable()
    if exe_path:
        shutil.copy2(exe_path, release_dir)
        print(f"📁 Copied executable: {exe_path}")
    else:
        print("❌ Could not find executable to copy")

    # Copy the installer if it exists
    installer_path = "installer/AutoSubtitler_Setup.exe"
    if os.path.exists(installer_path):
        shutil.copy2(installer_path, release_dir)
        print("📁 Copied installer")
    else:
        # Check if installer is in Output directory (default Inno Setup location)
        alternative_installer_path = "Output/AutoSubtitler_Setup.exe"
        if os.path.exists(alternative_installer_path):
            shutil.copy2(alternative_installer_path, release_dir)
            print("📁 Found installer in Output/ directory")

    # For single-file builds, we don't need separate DLLs
    # But include them anyway for users who might need them
    python_dir = os.path.dirname(sys.executable)
    for dll in ['python39.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll']:
        dll_path = os.path.join(python_dir, dll)
        if os.path.exists(dll_path):
            shutil.copy2(dll_path, release_dir)
            print(f"📁 Copied DLL: {dll}")

    # Create a README for the release
    readme_content = """
# Auto Subtitler v2.0

A GPU-accelerated video subtitle generator with smart caching and speed optimizations.

## Features

- Fast audio extraction and transcription
- Multiple language support
- Real-time processing simulation
- Smart caching for faster re-processing
- GPU acceleration support

## System Requirements

- Windows 10 or later
- 4GB RAM minimum (8GB recommended)
- Microsoft Visual C++ Redistributable 2019
- NVIDIA GPU with CUDA support (optional but recommended)

## Installation Options

### Option 1: Installer (Recommended)
1. Download `AutoSubtitler_Setup.exe`
2. Run the installer and follow the instructions
3. Launch Auto Subtitler from the Start menu or desktop shortcut

### Option 2: Portable Version
1. Download `AutoSubtitler.exe`
2. Create a new folder and place the executable there
3. Double-click to run (first run may take longer as it extracts files)

## Notes

- The first run may take longer as it downloads AI models
- Internet connection is required for translation features
- For best performance, use a GPU with CUDA support
- The portable version is a single file that contains everything needed

## Troubleshooting

If you encounter issues:
1. Make sure you have the latest Windows updates
2. Install the latest Microsoft Visual C++ Redistributable
3. Install the latest NVIDIA drivers if using a GPU
4. Check that your antivirus isn't blocking the application
5. Try running as administrator if you get permission errors
"""

    with open(os.path.join(release_dir, "README.md"), "w") as f:
        f.write(readme_content)

    print("✅ GitHub release files prepared in 'github_release' directory")
    return release_dir


def create_github_release_script():
    """Create a script to help with GitHub releases"""
    script_content = """#!/bin/bash
# Script to create a GitHub release and upload assets

# Set variables
VERSION="2.0"
RELEASE_DIR="github_release"
RELEASE_NAME="Auto Subtitler v$VERSION"
RELEASE_NOTES="Latest release of Auto Subtitler with performance improvements and bug fixes."

# Create a new release
echo "Creating GitHub release..."
gh release create "v$VERSION" --title "$RELEASE_NAME" --notes "$RELEASE_NOTES"

# Upload assets
echo "Uploading assets..."
gh release upload "v$VERSION" "$RELEASE_DIR/AutoSubtitler_Setup.exe#Installer"
gh release upload "v$VERSION" "$RELEASE_DIR/AutoSubtitler.exe#Portable Version"
gh release upload "v$VERSION" "$RELEASE_DIR/README.md#Release Notes"

echo "Release created successfully!"
"""

    with open("create_github_release.sh", "w") as f:
        f.write(script_content)

    # Also create a batch file for Windows users
    batch_content = """@echo off
REM Script to create a GitHub release and upload assets (Windows version)

REM Set variables
set VERSION=2.0
set RELEASE_DIR=github_release
set RELEASE_NAME=Auto Subtitler v%VERSION%
set RELEASE_NOTES=Latest release of Auto Subtitler with performance improvements and bug fixes.

REM Create a new release
echo Creating GitHub release...
gh release create "v%VERSION%" --title "%RELEASE_NAME%" --notes "%RELEASE_NOTES%"

REM Upload assets
echo Uploading assets...
gh release upload "v%VERSION%" "%RELEASE_DIR%/AutoSubtitler_Setup.exe#Installer"
gh release upload "v%VERSION%" "%RELEASE_DIR%/AutoSubtitler.exe#Portable Version"
gh release upload "v%VERSION%" "%RELEASE_DIR%/README.md#Release Notes"

echo Release created successfully!
pause
"""

    with open("create_github_release.bat", "w") as f:
        f.write(batch_content)

    print("✅ GitHub release scripts created")
    print("   - create_github_release.sh (for Linux/Mac)")
    print("   - create_github_release.bat (for Windows)")
    print("💡 Note: You need to install GitHub CLI (gh) and be authenticated")


def manual_installer_creation():
    """Provide manual instructions for creating installer"""
    print("🔧 Manual Installer Creation Instructions")
    print("=" * 50)
    print("Since automatic installer creation failed, here's how to do it manually:")
    print()
    print("1. Open Inno Setup Compiler (should be in your Start Menu)")
    print("2. Click 'File' > 'Open' and select 'installer.iss'")
    print("3. Click the 'Compile' button (green play button)")
    print("4. The installer will be created in the 'installer' directory")
    print()
    print("Alternatively, run this command in Command Prompt:")
    print(r'   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss')
    print()
    input("Press Enter to continue...")


def main():
    print("🎬 Auto Subtitler - Enhanced Build Script")
    print("=" * 50)

    # Check dependencies
    check_dependencies()

    # Check VC++ Redistributable
    check_vc_redist()

    # Install PyInstaller
    install_pyinstaller()

    while True:
        print("\nChoose an option:")
        print("1. Directory build (faster startup)")
        print("2. Single file build (more portable)")
        print("3. Custom spec file build")
        print("4. Create installer (requires Inno Setup)")
        print("5. Prepare GitHub release files")
        print("6. Create GitHub release scripts")
        print("7. Full build process (executable + installer + GitHub prep)")
        print("8. Manual installer creation instructions")
        print("9. Exit")
        print("10. Optimized single file build (RECOMMENDED for large apps)")

        choice = input("Enter your choice (1-10): ").strip()

        if choice == "1":
            # Quick directory build
            cmd = [
                "pyinstaller",
                "--windowed",
                "--name=AutoSubtitler",
                "--hidden-import", "whisper.transcribe",
                "--hidden-import", "whisper.tokenizer",
                "--hidden-import", "whisper.utils",
                "--hidden-import", "whisper.decoding",
                "--hidden-import", "whisper.model",
                "--hidden-import", "whisper.audio",
                "--collect-submodules", "whisper",
                "--collect-submodules", "torch",
                "--clean",
                "subtitler_gui.py"
            ]
            subprocess.run(cmd)
            find_executable()  # Show where the executable was created

        elif choice == "2":
            build_onefile()

        elif choice == "3":
            create_spec_file()
            build_executable()

        elif choice == "4":
            # Ask user which build to use for installer
            print("Which executable do you want to use for the installer?")
            print("a. Use existing executable (if available)")
            print("b. Build optimized executable first (recommended)")
            sub_choice = input("Choose (a/b): ").strip().lower()

            if sub_choice == "b":
                if build_onefile_optimized():
                    if not create_installer():
                        manual_installer_creation()
                else:
                    print("❌ Failed to build executable")
            else:
                if not create_installer():
                    manual_installer_creation()

        elif choice == "5":
            if find_executable():
                prepare_github_release()
            else:
                print("❌ Please build the executable first")

        elif choice == "6":
            create_github_release_script()

        elif choice == "7":
            # Full build process with optimized build
            if build_onefile_optimized():
                if create_installer():
                    prepare_github_release()
                    create_github_release_script()
                else:
                    manual_installer_creation()

        elif choice == "8":
            manual_installer_creation()

        elif choice == "9":
            print("👋 Goodbye!")
            break

        elif choice == "10":
            build_onefile_optimized()

        else:
            print("❌ Invalid choice. Please try again.")

    print("\n🎉 Build process completed!")
    print("💡 Tips:")
    print("   - Test the executable before distributing")
    print("   - For GitHub releases, install GitHub CLI (gh)")
    print("   - Sign up for a GitHub account if you don't have one")
    print("   - Consider code signing for better trust with users")
    print("   - The single-file build is recommended as it's truly standalone")


if __name__ == "__main__":
    main()