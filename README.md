# Auto Subtitler

A GPU-accelerated video subtitle generator with smart caching and speed optimizations.

## Features

- 🚀 Fast audio extraction and transcription
- 🌐 Multiple language support
- ⚡ Real-time processing simulation
- 💾 Smart caching for faster re-processing
- 🎯 GPU acceleration support
- 📁 SRT file export

## Installation

### Windows Installer
1. Download the latest installer from the [Releases](https://github.com/hhhpraise/auto-subtitler/releases) page
2. Run `AutoSubtitler_Setup.exe` and follow the installation wizard
3. Launch Auto Subtitler from the Start menu or desktop shortcut

### Portable Version
1. Download `AutoSubtitler.exe` from the [Releases](https://github.com/hhhpraise/auto-subtitler/releases) page
2. Run the executable directly (no installation required)

## System Requirements

- Windows 10 or later
- 4GB RAM minimum (8GB recommended)
- NVIDIA GPU with CUDA support (optional but recommended for best performance)

## Usage

1. Launch Auto Subtitler
2. Select a video file using the "Browse" button
3. Choose your preferred settings (language, model size, etc.)
4. Click "Start Processing" to generate subtitles
5. Save the resulting SRT file

## Building from Source

If you want to build from source:

1. Clone this repository
2. Install Python 3.8 or later
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `python subtitler_gui.py`

To build an executable:
1. Run the build script: `python build_app.py`
2. Follow the on-screen instructions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues:
1. Check the [Troubleshooting](https://github.com/hhhpraise/auto-subtitler/wiki/Troubleshooting) guide
2. Search existing [Issues](https://github.com/hhhpraise/auto-subtitler/issues)
3. Create a new issue if your problem isn't already reported