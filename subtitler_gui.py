import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import time
import hashlib
import json
import requests
import urllib.request
from datetime import timedelta
import subprocess
import sys
import webbrowser

# Import the subtitler classes
import speech_recognition as sr
import moviepy.editor as mp
from googletrans import Translator
import whisper
import wave
import torch


class SubtitlerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 Auto Subtitler - GPU Accelerated v2.0")
        self.root.geometry("1000x750")
        self.root.configure(bg='#2b2b2b')

        # Variables
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.target_language = tk.StringVar(value='en')
        self.model_size = tk.StringVar(value='base')
        self.processing_mode = tk.StringVar(value='complete')
        self.use_gpu = tk.BooleanVar(value=True)
        self.auto_translate = tk.BooleanVar(value=True)
        self.use_cache = tk.BooleanVar(value=True)

        # Queue for thread communication
        self.message_queue = queue.Queue()
        self.progress_queue = queue.Queue()

        # Processing variables
        self.is_processing = False
        self.current_process = None
        self.subtitler = None
        self.current_progress = 0
        self.total_steps = 0

        # Cache directory
        self.cache_dir = "../transcriper/subtitler_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_info = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.1f} GB)"
        else:
            self.gpu_info = "Not Available"

        # Internet connection status
        self.internet_connected = False
        self.check_internet_connection()

        self.setup_ui()
        self.check_queues()

    def check_internet_connection(self):
        """Check internet connectivity"""

        def check():
            try:
                # Try multiple reliable endpoints
                test_urls = [
                    "https://www.google.com",
                    "https://www.cloudflare.com",
                    "https://httpbin.org/get"
                ]

                for url in test_urls:
                    try:
                        response = urllib.request.urlopen(url, timeout=5)
                        if response.getcode() == 200:
                            self.internet_connected = True
                            self.message_queue.put("🌐 Internet connection: ✅ Connected")
                            return
                    except:
                        continue

                self.internet_connected = False
                self.message_queue.put("🌐 Internet connection: ❌ Disconnected")

            except Exception as e:
                self.internet_connected = False
                self.message_queue.put(f"🌐 Internet check failed: {str(e)}")

        # Run in separate thread to avoid blocking UI
        threading.Thread(target=check, daemon=True).start()

    def get_file_hash(self, filepath):
        """Generate hash for file caching"""
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.log_message(f"Error generating file hash: {e}")
            return None

    def get_cache_paths(self, video_path):
        """Get cache file paths for a video"""
        file_hash = self.get_file_hash(video_path)
        if not file_hash:
            return None, None, None

        cache_prefix = os.path.join(self.cache_dir, file_hash)
        audio_cache = f"{cache_prefix}_audio.wav"
        transcription_cache = f"{cache_prefix}_transcription.json"
        metadata_cache = f"{cache_prefix}_metadata.json"

        return audio_cache, transcription_cache, metadata_cache

    def save_cache_metadata(self, metadata_path, video_path, model_size, detected_lang):
        """Save metadata about cached files"""
        metadata = {
            'video_path': video_path,
            'model_size': model_size,
            'detected_language': detected_lang,
            'created_time': time.time(),
            'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
        }

        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.log_message(f"Error saving cache metadata: {e}")

    def load_cache_metadata(self, metadata_path):
        """Load metadata about cached files"""
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.log_message(f"Error loading cache metadata: {e}")
            return None

    def is_cache_valid(self, video_path, metadata_path, model_size):
        """Check if cached files are still valid"""
        if not os.path.exists(metadata_path):
            return False

        metadata = self.load_cache_metadata(metadata_path)
        if not metadata:
            return False

        # Check if video file hasn't changed
        if not os.path.exists(video_path):
            return False

        current_size = os.path.getsize(video_path)
        if current_size != metadata.get('file_size', 0):
            return False

        # Check if model size matches
        if metadata.get('model_size') != model_size:
            return False

        return True

    def setup_ui(self):
        """Create the main user interface"""

        # Title
        title_frame = tk.Frame(self.root, bg='#2b2b2b')
        title_frame.pack(fill='x', pady=10)

        title_label = tk.Label(title_frame, text="🎬 Auto Subtitler v2.0",
                               font=('Arial', 24, 'bold'), fg='#4CAF50', bg='#2b2b2b')
        title_label.pack()

        subtitle_label = tk.Label(title_frame, text="AI-Powered Video Subtitle Generation with Smart Caching",
                                  font=('Arial', 12), fg='#888', bg='#2b2b2b')
        subtitle_label.pack()

        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Left panel for settings
        left_panel = tk.Frame(main_frame, bg='#3b3b3b', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Right panel for output
        right_panel = tk.Frame(main_frame, bg='#3b3b3b', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))

        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)

    def setup_left_panel(self, parent):
        """Setup the left panel with all controls"""

        # Settings title
        settings_label = tk.Label(parent, text="⚙️ Configuration",
                                  font=('Arial', 16, 'bold'), fg='#4CAF50', bg='#3b3b3b')
        settings_label.pack(pady=10)

        # System status frame
        status_frame = tk.LabelFrame(parent, text="🔧 System Status",
                                     fg='white', bg='#3b3b3b', font=('Arial', 10, 'bold'))
        status_frame.pack(fill='x', padx=10, pady=5)

        # Internet connection status
        self.connection_status = tk.Label(status_frame, text="🌐 Checking connection...",
                                          fg='#FFC107', bg='#3b3b3b', font=('Arial', 10))
        self.connection_status.pack(anchor='w', padx=5, pady=2)

        # Internet check button
        check_internet_btn = tk.Button(status_frame, text="🔄 Check Internet",
                                       command=self.manual_internet_check, bg='#2196F3', fg='white',
                                       font=('Arial', 9, 'bold'), cursor='hand2')
        check_internet_btn.pack(anchor='w', padx=5, pady=2)

        # Cache status
        cache_info = self.get_cache_info()
        self.cache_status = tk.Label(status_frame, text=f"💾 Cache: {cache_info}",
                                     fg='#888', bg='#3b3b3b', font=('Arial', 10))
        self.cache_status.pack(anchor='w', padx=5, pady=2)

        # Clear cache button
        clear_cache_btn = tk.Button(status_frame, text="🗑️ Clear Cache",
                                    command=self.clear_cache, bg='#FF5722', fg='white',
                                    font=('Arial', 9, 'bold'), cursor='hand2')
        clear_cache_btn.pack(anchor='w', padx=5, pady=2)

        # Video selection
        video_frame = tk.LabelFrame(parent, text="📁 Video File",
                                    fg='white', bg='#3b3b3b', font=('Arial', 10, 'bold'))
        video_frame.pack(fill='x', padx=10, pady=5)

        video_entry_frame = tk.Frame(video_frame, bg='#3b3b3b')
        video_entry_frame.pack(fill='x', padx=5, pady=5)

        self.video_entry = tk.Entry(video_entry_frame, textvariable=self.video_path,
                                    bg='#555', fg='white', font=('Arial', 10), width=40)
        self.video_entry.pack(side='left', fill='x', expand=True)
        self.video_entry.bind('<KeyRelease>', self.on_video_path_change)

        video_browse_btn = tk.Button(video_entry_frame, text="Browse",
                                     command=self.browse_video, bg='#4CAF50', fg='white',
                                     font=('Arial', 10, 'bold'), cursor='hand2')
        video_browse_btn.pack(side='right', padx=(5, 0))

        # Cache status for current video
        self.video_cache_status = tk.Label(video_frame, text="",
                                           fg='#888', bg='#3b3b3b', font=('Arial', 9))
        self.video_cache_status.pack(anchor='w', padx=5, pady=2)

        # Output path
        output_frame = tk.LabelFrame(parent, text="💾 Output Location",
                                     fg='white', bg='#3b3b3b', font=('Arial', 10, 'bold'))
        output_frame.pack(fill='x', padx=10, pady=5)

        output_entry_frame = tk.Frame(output_frame, bg='#3b3b3b')
        output_entry_frame.pack(fill='x', padx=5, pady=5)

        self.output_entry = tk.Entry(output_entry_frame, textvariable=self.output_path,
                                     bg='#555', fg='white', font=('Arial', 10), width=40)
        self.output_entry.pack(side='left', fill='x', expand=True)

        output_browse_btn = tk.Button(output_entry_frame, text="Browse",
                                      command=self.browse_output, bg='#4CAF50', fg='white',
                                      font=('Arial', 10, 'bold'), cursor='hand2')
        output_browse_btn.pack(side='right', padx=(5, 0))

        # Language settings
        lang_frame = tk.LabelFrame(parent, text="🌐 Language Settings",
                                   fg='white', bg='#3b3b3b', font=('Arial', 10, 'bold'))
        lang_frame.pack(fill='x', padx=10, pady=5)

        # Auto-translate checkbox
        auto_translate_check = tk.Checkbutton(lang_frame, text="Auto-translate to target language",
                                              variable=self.auto_translate, fg='white', bg='#3b3b3b',
                                              selectcolor='#555', font=('Arial', 10))
        auto_translate_check.pack(anchor='w', padx=5, pady=2)

        # Target language
        lang_label = tk.Label(lang_frame, text="Target Language:", fg='white', bg='#3b3b3b')
        lang_label.pack(anchor='w', padx=5)

        languages = {
            'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de',
            'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Chinese': 'zh',
            'Japanese': 'ja', 'Korean': 'ko', 'Arabic': 'ar', 'Hindi': 'hi',
            'Dutch': 'nl', 'Swedish': 'sv', 'Norwegian': 'no', 'Danish': 'da',
            'Finnish': 'fi', 'Polish': 'pl', 'Czech': 'cs', 'Hungarian': 'hu'
        }

        lang_combo = ttk.Combobox(lang_frame, textvariable=self.target_language,
                                  values=list(languages.keys()), width=25)
        lang_combo.pack(padx=5, pady=2)
        lang_combo.set('English')

        # Store language mapping
        self.language_codes = {v: k for k, v in languages.items()}
        self.languages = languages

        # Test translation button
        test_translate_btn = tk.Button(lang_frame, text="🧪 Test Translation",
                                       command=self.test_translation, bg='#9C27B0', fg='white',
                                       font=('Arial', 9, 'bold'), cursor='hand2')
        test_translate_btn.pack(padx=5, pady=2)

        # AI Model settings
        model_frame = tk.LabelFrame(parent, text="🤖 AI Model Settings",
                                    fg='white', bg='#3b3b3b', font=('Arial', 10, 'bold'))
        model_frame.pack(fill='x', padx=10, pady=5)

        # GPU checkbox
        gpu_check = tk.Checkbutton(model_frame, text=f"Use GPU Acceleration ({self.gpu_info})",
                                   variable=self.use_gpu, fg='white', bg='#3b3b3b',
                                   selectcolor='#555', font=('Arial', 10),
                                   state='normal' if self.gpu_available else 'disabled')
        gpu_check.pack(anchor='w', padx=5, pady=2)

        if not self.gpu_available:
            self.use_gpu.set(False)

        # Cache checkbox
        cache_check = tk.Checkbutton(model_frame, text="Use smart caching (faster re-processing)",
                                     variable=self.use_cache, fg='white', bg='#3b3b3b',
                                     selectcolor='#555', font=('Arial', 10))
        cache_check.pack(anchor='w', padx=5, pady=2)

        # Model size
        model_label = tk.Label(model_frame, text="Model Size:", fg='white', bg='#3b3b3b')
        model_label.pack(anchor='w', padx=5)

        model_combo = ttk.Combobox(model_frame, textvariable=self.model_size,
                                   values=['tiny', 'base', 'small', 'medium', 'large'], width=25)
        model_combo.pack(padx=5, pady=2)
        model_combo.set('base')
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)

        # Model info
        model_info = {
            'tiny': '~39 MB, fastest, lowest accuracy',
            'base': '~142 MB, good balance',
            'small': '~466 MB, better accuracy',
            'medium': '~1.5 GB, high accuracy',
            'large': '~2.9 GB, highest accuracy'
        }

        self.model_info_label = tk.Label(model_frame, text=model_info.get('base', ''),
                                         fg='#888', bg='#3b3b3b', font=('Arial', 8))
        self.model_info_label.pack(anchor='w', padx=5)

        # Processing mode
        mode_frame = tk.LabelFrame(parent, text="⚡ Processing Mode",
                                   fg='white', bg='#3b3b3b', font=('Arial', 10, 'bold'))
        mode_frame.pack(fill='x', padx=10, pady=5)

        complete_radio = tk.Radiobutton(mode_frame, text="Complete Processing (Better Quality)",
                                        variable=self.processing_mode, value='complete',
                                        fg='white', bg='#3b3b3b', selectcolor='#555')
        complete_radio.pack(anchor='w', padx=5, pady=2)

        realtime_radio = tk.Radiobutton(mode_frame, text="Real-time Processing (Watch & Process)",
                                        variable=self.processing_mode, value='realtime',
                                        fg='white', bg='#3b3b3b', selectcolor='#555')
        realtime_radio.pack(anchor='w', padx=5, pady=2)

        # Action buttons
        button_frame = tk.Frame(parent, bg='#3b3b3b')
        button_frame.pack(fill='x', padx=10, pady=15)

        self.start_btn = tk.Button(button_frame, text="🚀 Start Processing",
                                   command=self.start_processing, bg='#4CAF50', fg='white',
                                   font=('Arial', 14, 'bold'), cursor='hand2', height=2)
        self.start_btn.pack(fill='x', pady=2)

        self.stop_btn = tk.Button(button_frame, text="⏹️ Stop Processing",
                                  command=self.stop_processing, bg='#f44336', fg='white',
                                  font=('Arial', 12, 'bold'), cursor='hand2', state='disabled')
        self.stop_btn.pack(fill='x', pady=2)

        self.watch_btn = tk.Button(button_frame, text="🎥 Watch Video",
                                   command=self.watch_video, bg='#FF9800', fg='white',
                                   font=('Arial', 12, 'bold'), cursor='hand2', state='disabled')
        self.watch_btn.pack(fill='x', pady=2)

    def setup_right_panel(self, parent):
        """Setup the right panel with progress and output"""

        # Progress title
        progress_label = tk.Label(parent, text="📊 Progress & Output",
                                  font=('Arial', 16, 'bold'), fg='#4CAF50', bg='#3b3b3b')
        progress_label.pack(pady=10)

        # Progress frame
        progress_frame = tk.Frame(parent, bg='#3b3b3b')
        progress_frame.pack(fill='x', padx=10, pady=5)

        # Progress bar with percentage
        self.progress = ttk.Progressbar(progress_frame, mode='determinate', maximum=100)
        self.progress.pack(fill='x', pady=2)

        self.progress_label = tk.Label(progress_frame, text="0% - Ready to process",
                                       fg='#4CAF50', bg='#3b3b3b', font=('Arial', 11, 'bold'))
        self.progress_label.pack(pady=2)

        # Status label
        self.status_label = tk.Label(parent, text="Ready to process video",
                                     fg='#4CAF50', bg='#3b3b3b', font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=5)

        # Estimated time
        self.time_label = tk.Label(parent, text="",
                                   fg='#888', bg='#3b3b3b', font=('Arial', 10))
        self.time_label.pack(pady=2)

        # Output text area
        output_frame = tk.LabelFrame(parent, text="📝 Processing Log",
                                     fg='white', bg='#3b3b3b', font=('Arial', 10, 'bold'))
        output_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.output_text = scrolledtext.ScrolledText(output_frame,
                                                     bg='#1e1e1e', fg='#00ff00',
                                                     font=('Consolas', 9),
                                                     height=12, wrap='word')
        self.output_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Results frame
        results_frame = tk.LabelFrame(parent, text="🎯 Results",
                                      fg='white', bg='#3b3b3b', font=('Arial', 10, 'bold'))
        results_frame.pack(fill='x', padx=10, pady=(0, 10))

        self.results_text = tk.Label(results_frame, text="No results yet",
                                     fg='#888', bg='#3b3b3b', font=('Arial', 10),
                                     wraplength=300, justify='left')
        self.results_text.pack(padx=5, pady=5)

        # Action buttons for results
        results_button_frame = tk.Frame(results_frame, bg='#3b3b3b')
        results_button_frame.pack(fill='x', padx=5, pady=5)

        self.open_srt_btn = tk.Button(results_button_frame, text="📄 Open SRT",
                                      command=self.open_srt_file, bg='#2196F3', fg='white',
                                      font=('Arial', 10, 'bold'), cursor='hand2', state='disabled')
        self.open_srt_btn.pack(side='left', padx=2)

        self.open_folder_btn = tk.Button(results_button_frame, text="📁 Folder",
                                         command=self.open_output_folder, bg='#2196F3', fg='white',
                                         font=('Arial', 10, 'bold'), cursor='hand2', state='disabled')
        self.open_folder_btn.pack(side='right', padx=2)

    def get_cache_info(self):
        """Get information about cache directory"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith(('.wav', '.json'))]
            cache_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files)
            cache_size_mb = cache_size / (1024 * 1024)
            return f"{len(cache_files) // 3} videos cached ({cache_size_mb:.1f} MB)"
        except:
            return "0 videos cached"

    def clear_cache(self):
        """Clear the cache directory"""
        try:
            cache_files = os.listdir(self.cache_dir)
            for file in cache_files:
                os.remove(os.path.join(self.cache_dir, file))
            self.cache_status.config(text="💾 Cache: Cleared")
            self.log_message("🗑️ Cache cleared successfully")
            messagebox.showinfo("Cache Cleared", "All cached files have been removed")
        except Exception as e:
            self.log_message(f"Error clearing cache: {e}")
            messagebox.showerror("Error", f"Could not clear cache: {e}")

    def manual_internet_check(self):
        """Manually check internet connection"""
        self.connection_status.config(text="🌐 Checking...", fg='#FFC107')
        threading.Thread(target=self.check_internet_connection, daemon=True).start()

    def test_translation(self):
        """Test translation functionality"""
        if not self.internet_connected:
            messagebox.showwarning("No Internet", "Internet connection required for translation testing")
            return

        def test():
            try:
                translator = Translator()
                test_text = "Hello, this is a test."
                target_lang = self.languages[self.target_language.get()]

                if target_lang == 'en':
                    # If target is English, translate from Spanish to test
                    result = translator.translate("Hola, esto es una prueba.", src='es', dest='en')
                else:
                    result = translator.translate(test_text, src='en', dest=target_lang)

                self.message_queue.put(f"🧪 Translation test: '{result.text}'")
                self.message_queue.put("✅ Translation service working correctly")

            except Exception as e:
                self.message_queue.put(f"❌ Translation test failed: {str(e)}")
                self.message_queue.put("💡 Try checking your internet connection or using a VPN")

        threading.Thread(target=test, daemon=True).start()

    def on_video_path_change(self, event=None):
        """Handle video path changes"""
        video_path = self.video_path.get()
        if video_path and os.path.exists(video_path):
            # Check cache status
            audio_cache, transcription_cache, metadata_cache = self.get_cache_paths(video_path)

            if self.use_cache.get() and audio_cache and os.path.exists(audio_cache):
                if self.is_cache_valid(video_path, metadata_cache, self.model_size.get()):
                    self.video_cache_status.config(text="💾 Cached audio found - will use cached data", fg='#4CAF50')
                else:
                    self.video_cache_status.config(text="💾 Cache invalid - will regenerate", fg='#FFC107')
            else:
                self.video_cache_status.config(text="💾 No cache - will extract audio", fg='#888')

    def on_model_change(self, event=None):
        """Handle model size changes"""
        model_info = {
            'tiny': '~39 MB, fastest, lowest accuracy',
            'base': '~142 MB, good balance',
            'small': '~466 MB, better accuracy',
            'medium': '~1.5 GB, high accuracy',
            'large': '~2.9 GB, highest accuracy'
        }

        self.model_info_label.config(text=model_info.get(self.model_size.get(), ''))
        self.on_video_path_change()  # Update cache status

    def browse_video(self):
        """Browse for video file"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.ts"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        if filename:
            self.video_path.set(filename)
            # Auto-generate output path
            base_name = os.path.splitext(filename)[0]
            self.output_path.set(f"{base_name}_subtitles.srt")
            self.watch_btn.config(state='normal')
            self.log_message(f"Selected video: {os.path.basename(filename)}")
            self.on_video_path_change()

    def browse_output(self):
        """Browse for output location"""
        filename = filedialog.asksaveasfilename(
            title="Save Subtitles As",
            defaultextension=".srt",
            filetypes=[("SRT files", "*.srt"), ("All files", "*.*")]
        )
        if filename:
            self.output_path.set(filename)

    def watch_video(self):
        """Open video in default player"""
        video_path = self.video_path.get()
        if video_path and os.path.exists(video_path):
            try:
                if sys.platform == "win32":
                    os.startfile(video_path)
                elif sys.platform == "darwin":
                    subprocess.call(["open", video_path])
                else:
                    subprocess.call(["xdg-open", video_path])
                self.log_message("Opened video in default player")
            except Exception as e:
                self.log_message(f"Error opening video: {e}")
        else:
            messagebox.showerror("Error", "Please select a valid video file first")

    def update_progress(self, percentage, status_text, time_estimate=None):
        """Update progress bar and status"""
        self.progress.config(value=percentage)
        self.progress_label.config(text=f"{percentage:.1f}% - {status_text}")
        self.status_label.config(text=status_text)

        if time_estimate:
            self.time_label.config(text=f"⏱️ Estimated time remaining: {time_estimate}")

        self.root.update_idletasks()

    def start_processing(self):
        """Start the subtitle processing"""
        if not self.validate_inputs():
            return

        # Check internet for translation
        if self.auto_translate.get() and not self.internet_connected:
            response = messagebox.askyesno(
                "No Internet Connection",
                "Translation requires internet connection, but you're offline.\n\n"
                "Continue without translation?"
            )
            if not response:
                return
            self.auto_translate.set(False)

        self.is_processing = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress.config(mode='determinate', value=0)
        self.update_progress(0, "Initializing...")

        # Start processing in separate thread
        self.current_process = threading.Thread(target=self.process_video_thread)
        self.current_process.daemon = True
        self.current_process.start()

    def validate_inputs(self):
        """Validate user inputs"""
        video_path = self.video_path.get()
        output_path = self.output_path.get()

        if not video_path:
            messagebox.showerror("Error", "Please select a video file")
            return False

        if not os.path.exists(video_path):
            messagebox.showerror("Error", "Video file does not exist")
            return False

        if not output_path:
            messagebox.showerror("Error", "Please specify output location")
            return False

        # Check available disk space
        try:
            output_dir = os.path.dirname(output_path)
            free_space = os.statvfs(output_dir).f_frsize * os.statvfs(output_dir).f_bavail / (1024 ** 3)
            if free_space < 1:  # Less than 1GB
                messagebox.showwarning("Low Disk Space",
                                       f"Low disk space ({free_space:.1f} GB). Processing may fail.")
        except:
            pass  # Skip check on Windows or if statvfs not available

        return True

    def process_video_thread(self):
        """Process video in separate thread"""
        start_time = time.time()

        try:
            self.update_progress(5, "Initializing AI models...")

            # Initialize subtitler
            self.subtitler = AutoSubtitlerCore(
                use_whisper=True,
                target_language=self.languages[self.target_language.get()],
                use_gpu=self.use_gpu.get(),
                whisper_model_size=self.model_size.get(),
                message_callback=self.message_queue.put,
                progress_callback=self.progress_queue.put,
                use_cache=self.use_cache.get(),
                cache_dir=self.cache_dir
            )

            video_path = self.video_path.get()
            output_path = self.output_path.get()

            if self.processing_mode.get() == 'realtime':
                self.update_progress(10, "Starting real-time processing...")
                success = self.subtitler.real_time_subtitles(video_path)
            else:
                self.update_progress(10, "Starting complete processing...")
                success = self.subtitler.process_video_file(
                    video_path, output_path,
                    translate=self.auto_translate.get()
                )

            if success:
                processing_time = time.time() - start_time
                self.message_queue.put(f"✅ Processing completed in {processing_time:.1f} seconds!")
                self.message_queue.put(("SUCCESS", output_path))
                self.update_progress(100, "Processing completed successfully!")
            else:
                self.message_queue.put("❌ Processing failed!")
                self.message_queue.put(("ERROR", "Processing failed"))
                self.update_progress(0, "Processing failed")

        except Exception as e:
            self.message_queue.put(f"❌ Error: {str(e)}")
            self.message_queue.put(("ERROR", str(e)))
            self.update_progress(0, "Error occurred")
        finally:
            self.message_queue.put("FINISHED")

    def stop_processing(self):
        """Stop the current processing"""
        self.is_processing = False
        self.message_queue.put("⏹️ Stopping processing...")
        if self.subtitler:
            self.subtitler.stop_processing = True
        self.reset_ui()

    def reset_ui(self):
        """Reset UI to initial state"""
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress.config(mode='determinate', value=0)
        self.progress_label.config(text="0% - Ready to process")
        self.status_label.config(text="Ready to process video", fg='#4CAF50')
        self.time_label.config(text="")
        self.is_processing = False

    def check_queues(self):
        """Check message and progress queues"""
        try:
            # Check progress queue
            while True:
                try:
                    progress_data = self.progress_queue.get_nowait()
                    if isinstance(progress_data, dict):
                        percentage = progress_data.get('percentage', 0)
                        status = progress_data.get('status', '')
                        time_estimate = progress_data.get('time_estimate')
                        self.update_progress(percentage, status, time_estimate)
                except queue.Empty:
                    break

            # Check message queue
            while True:
                message = self.message_queue.get_nowait()

                if message == "FINISHED":
                    self.reset_ui()
                elif isinstance(message, tuple) and message[0] == "SUCCESS":
                    output_path = message[1]
                    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                    self.results_text.config(
                        text=f"✅ Subtitles created successfully!\n📄 File: {os.path.basename(output_path)}\n💾 Size: {file_size / 1024:.1f} KB",
                        fg='#4CAF50'
                    )
                    self.open_srt_btn.config(state='normal')
                    self.open_folder_btn.config(state='normal')
                elif isinstance(message, tuple) and message[0] == "ERROR":
                    self.results_text.config(
                        text=f"❌ Processing failed!\n{message[1]}",
                        fg='#f44336'
                    )
                elif message.startswith("🌐 Internet connection:"):
                    # Update connection status
                    if "✅ Connected" in message:
                        self.connection_status.config(text=message, fg='#4CAF50')
                        self.internet_connected = True
                    else:
                        self.connection_status.config(text=message, fg='#f44336')
                        self.internet_connected = False
                else:
                    self.log_message(str(message))

        except queue.Empty:
            pass

        # Update cache info periodically
        self.cache_status.config(text=f"💾 Cache: {self.get_cache_info()}")

        # Schedule next check
        self.root.after(100, self.check_queues)

    def log_message(self, message):
        """Add message to output log"""
        timestamp = time.strftime('%H:%M:%S')
        self.output_text.insert(tk.END, f"{timestamp} - {message}\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()

    def open_srt_file(self):
        """Open SRT file in default text editor"""
        output_path = self.output_path.get()
        if output_path and os.path.exists(output_path):
            try:
                if sys.platform == "win32":
                    os.startfile(output_path)
                elif sys.platform == "darwin":
                    subprocess.call(["open", output_path])
                else:
                    subprocess.call(["xdg-open", output_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {e}")

    def open_output_folder(self):
        """Open output folder in file explorer"""
        output_path = self.output_path.get()
        if output_path:
            folder_path = os.path.dirname(output_path)
            try:
                if sys.platform == "win32":
                    os.startfile(folder_path)
                elif sys.platform == "darwin":
                    subprocess.call(["open", folder_path])
                else:
                    subprocess.call(["xdg-open", folder_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open folder: {e}")


# Enhanced core processing class with caching and progress tracking
class AutoSubtitlerCore:
    def __init__(self, use_whisper=True, target_language='en', use_gpu=True,
                 whisper_model_size='base', message_callback=None, progress_callback=None,
                 use_cache=True, cache_dir="subtitler_cache"):
        self.use_whisper = use_whisper
        self.target_language = target_language
        self.use_gpu = use_gpu
        self.whisper_model_size = whisper_model_size
        self.message_callback = message_callback or (lambda x: None)
        self.progress_callback = progress_callback or (lambda x: None)
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.stop_processing = False

        # Initialize translator with fallback options
        self.translator = None
        self.init_translator()

        # Check GPU availability
        self.device = self._check_gpu_availability()

        if use_whisper:
            self.message_callback(f"Loading Whisper model ({whisper_model_size})...")
            self.progress_callback({'percentage': 15, 'status': f'Loading {whisper_model_size} model...'})

            try:
                self.whisper_model = whisper.load_model(whisper_model_size, device=self.device)
                self.message_callback("✅ Whisper model loaded successfully")
                self.progress_callback({'percentage': 20, 'status': 'Model loaded successfully'})
            except Exception as e:
                self.message_callback(f"❌ Error loading Whisper model: {e}")
                raise
        else:
            self.recognizer = sr.Recognizer()

    def init_translator(self):
        """Initialize translator with error handling"""
        try:
            self.translator = Translator()
            # Test translator
            test_result = self.translator.translate("test", dest='es')
            self.message_callback("✅ Translator initialized successfully")
        except Exception as e:
            self.message_callback(f"⚠️ Translator initialization warning: {e}")
            self.translator = None

    def _check_gpu_availability(self):
        """Check if GPU is available and return appropriate device"""
        if self.use_gpu and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.message_callback(f"🚀 Using GPU: {gpu_name} ({memory_gb:.1f} GB)")
            return "cuda"
        else:
            self.message_callback("🖥️ Using CPU for processing")
            return "cpu"

    def get_file_hash(self, filepath):
        """Generate hash for file caching"""
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.message_callback(f"Error generating file hash: {e}")
            return None

    def get_cache_paths(self, video_path):
        """Get cache file paths for a video"""
        file_hash = self.get_file_hash(video_path)
        if not file_hash:
            return None, None, None

        cache_prefix = os.path.join(self.cache_dir, f"{file_hash}_{self.whisper_model_size}")
        audio_cache = f"{cache_prefix}_audio.wav"
        transcription_cache = f"{cache_prefix}_transcription.json"
        metadata_cache = f"{cache_prefix}_metadata.json"

        return audio_cache, transcription_cache, metadata_cache

    def extract_audio(self, video_path, audio_path="temp_audio.wav"):
        """Extract audio from video file with caching"""
        if self.stop_processing:
            return None

        try:
            # Check for cached audio first
            if self.use_cache:
                audio_cache, _, metadata_cache = self.get_cache_paths(video_path)
                if audio_cache and os.path.exists(audio_cache):
                    if self.is_cache_valid(video_path, metadata_cache):
                        self.message_callback("💾 Using cached audio file")
                        self.progress_callback({'percentage': 40, 'status': 'Using cached audio'})
                        return audio_cache

            self.message_callback("🎵 Extracting audio from video...")
            self.progress_callback({'percentage': 25, 'status': 'Extracting audio...'})

            # Get video duration for progress tracking
            video = mp.VideoFileClip(video_path)
            duration = video.duration
            self.message_callback(f"📹 Video duration: {duration:.1f} seconds")

            audio = video.audio

            # Use cache path if caching is enabled
            if self.use_cache:
                audio_cache, _, _ = self.get_cache_paths(video_path)
                if audio_cache:
                    audio_path = audio_cache

            audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()

            self.message_callback("✅ Audio extraction completed")
            self.progress_callback({'percentage': 40, 'status': 'Audio extraction completed'})

            return audio_path
        except Exception as e:
            self.message_callback(f"❌ Error extracting audio: {e}")
            return None

    def is_cache_valid(self, video_path, metadata_path):
        """Check if cached files are still valid"""
        if not os.path.exists(metadata_path):
            return False

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Check if video file hasn't changed
            if not os.path.exists(video_path):
                return False

            current_size = os.path.getsize(video_path)
            if current_size != metadata.get('file_size', 0):
                return False

            # Check if model size matches
            if metadata.get('model_size') != self.whisper_model_size:
                return False

            return True
        except Exception as e:
            self.message_callback(f"Cache validation error: {e}")
            return False

    def save_transcription_cache(self, transcription_cache, segments, detected_lang):
        """Save transcription to cache"""
        try:
            cache_data = {
                'segments': segments,
                'detected_language': detected_lang,
                'model_size': self.whisper_model_size,
                'created_time': time.time()
            }

            with open(transcription_cache, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)

            self.message_callback("💾 Transcription saved to cache")
        except Exception as e:
            self.message_callback(f"Error saving transcription cache: {e}")

    def load_transcription_cache(self, transcription_cache):
        """Load transcription from cache"""
        try:
            with open(transcription_cache, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            return cache_data['segments'], cache_data['detected_language']
        except Exception as e:
            self.message_callback(f"Error loading transcription cache: {e}")
            return None, None

    def transcribe_with_whisper(self, audio_path, video_path=None):
        """Transcribe audio using Whisper with caching"""
        if self.stop_processing:
            return None, None

        try:
            # Check for cached transcription
            if self.use_cache and video_path:
                _, transcription_cache, metadata_cache = self.get_cache_paths(video_path)

                if (transcription_cache and os.path.exists(transcription_cache) and
                        self.is_cache_valid(video_path, metadata_cache)):

                    self.message_callback("💾 Using cached transcription")
                    self.progress_callback({'percentage': 80, 'status': 'Using cached transcription'})

                    segments, detected_lang = self.load_transcription_cache(transcription_cache)
                    if segments and detected_lang:
                        self.message_callback(f"📝 Loaded {len(segments)} cached segments")
                        return segments, detected_lang

            self.message_callback("🗣️ Transcribing speech to text...")
            self.progress_callback({'percentage': 50, 'status': 'Transcribing audio...'})

            start_time = time.time()

            # Get audio duration for better progress tracking
            try:
                with wave.open(audio_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / float(sample_rate)
                    self.message_callback(f"🎵 Audio duration: {duration:.1f} seconds")
            except:
                duration = 0

            result = self.whisper_model.transcribe(
                audio_path,
                fp16=self.device == "cuda",
                verbose=False
            )

            if self.stop_processing:
                return None, None

            end_time = time.time()
            processing_time = end_time - start_time

            detected_lang = result.get('language', 'unknown')
            segments = result["segments"]

            self.message_callback(f"🌐 Detected language: {detected_lang}")
            self.message_callback(f"⚡ Transcription completed in {processing_time:.1f} seconds")
            self.message_callback(f"📝 Found {len(segments)} speech segments")

            # Calculate processing speed
            if duration > 0:
                speed_factor = duration / processing_time
                self.message_callback(f"🚀 Processing speed: {speed_factor:.1f}x real-time")

            # Cache the transcription
            if self.use_cache and video_path:
                _, transcription_cache, metadata_cache = self.get_cache_paths(video_path)
                if transcription_cache:
                    self.save_transcription_cache(transcription_cache, segments, detected_lang)
                    self.save_cache_metadata(metadata_cache, video_path, detected_lang)

            self.progress_callback({'percentage': 80, 'status': 'Transcription completed'})
            return segments, detected_lang

        except Exception as e:
            self.message_callback(f"❌ Error with Whisper transcription: {e}")
            return None, None

    def save_cache_metadata(self, metadata_path, video_path, detected_lang):
        """Save metadata about cached files"""
        metadata = {
            'video_path': video_path,
            'model_size': self.whisper_model_size,
            'detected_language': detected_lang,
            'created_time': time.time(),
            'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
        }

        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.message_callback(f"Error saving cache metadata: {e}")

    def translate_text(self, text, source_lang='auto'):
        """Translate text to target language with better error handling"""
        if self.stop_processing:
            return text

        try:
            if source_lang == self.target_language:
                return text

            if not self.translator:
                self.message_callback("⚠️ Translator not available, skipping translation")
                return text

            # Try translation with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    translation = self.translator.translate(text,
                                                            src=source_lang,
                                                            dest=self.target_language)
                    return translation.text
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.message_callback(f"⚠️ Translation attempt {attempt + 1} failed, retrying...")
                        time.sleep(1)  # Wait before retry
                    else:
                        self.message_callback(f"⚠️ Translation failed after {max_retries} attempts: {e}")
                        return text

        except Exception as e:
            self.message_callback(f"⚠️ Translation error: {e}")
            return text

    def format_time(self, seconds):
        """Convert seconds to SRT time format"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"

    def create_srt(self, segments, output_path, translate=False, detected_lang=None):
        """Create SRT subtitle file from segments with progress tracking"""
        if self.stop_processing:
            return False

        try:
            self.message_callback("📝 Creating SRT subtitle file...")
            self.progress_callback({'percentage': 85, 'status': 'Creating subtitle file...'})

            total_segments = len(segments)
            translated_count = 0
            failed_translations = 0

            with open(output_path, 'w', encoding='utf-8') as srt_file:
                for i, segment in enumerate(segments, 1):
                    if self.stop_processing:
                        return False

                    # Update progress for subtitle creation
                    progress = 85 + (10 * i / total_segments)  # 85-95%
                    self.progress_callback({
                        'percentage': progress,
                        'status': f'Creating subtitles... ({i}/{total_segments})'
                    })

                    start_time = self.format_time(segment['start'])
                    end_time = self.format_time(segment['end'])
                    text = segment['text'].strip()

                    # Translate if requested and not in target language
                    if translate and detected_lang != self.target_language and self.translator:
                        original_text = text
                        try:
                            text = self.translate_text(text, detected_lang)
                            if text != original_text:
                                translated_count += 1
                        except Exception as e:
                            failed_translations += 1
                            self.message_callback(f"⚠️ Translation failed for segment {i}: {e}")
                            text = original_text

                    # Write SRT format
                    srt_file.write(f"{i}\n")
                    srt_file.write(f"{start_time} --> {end_time}\n")
                    srt_file.write(f"{text}\n\n")

            # Report translation results
            if translate and translated_count > 0:
                self.message_callback(f"🌐 Translated {translated_count} segments to {self.target_language}")
                if failed_translations > 0:
                    self.message_callback(f"⚠️ {failed_translations} translations failed")

            self.message_callback(f"📄 SRT file created: {os.path.basename(output_path)}")
            self.progress_callback({'percentage': 95, 'status': 'Subtitle file created'})
            return True

        except Exception as e:
            self.message_callback(f"❌ Error creating SRT file: {e}")
            return False

    def process_video_file(self, video_path, output_srt, translate=False):
        """Process entire video file with enhanced progress tracking"""
        if self.stop_processing:
            return False

        self.message_callback(f"🎬 Processing video: {os.path.basename(video_path)}")

        # Get video info
        try:
            video = mp.VideoFileClip(video_path)
            duration = video.duration
            file_size = os.path.getsize(video_path) / (1024 ** 2)  # MB
            video.close()

            self.message_callback(f"📊 Video info: {duration:.1f}s, {file_size:.1f} MB")

            # Estimate processing time
            estimated_time = self.estimate_processing_time(duration, file_size)
            if estimated_time:
                self.progress_callback({
                    'percentage': 5,
                    'status': 'Analyzing video...',
                    'time_estimate': estimated_time
                })
        except Exception as e:
            self.message_callback(f"⚠️ Could not analyze video: {e}")

        # Extract audio (with caching)
        audio_path = self.extract_audio(video_path)
        if not audio_path or self.stop_processing:
            return False

        # Transcribe audio (with caching)
        segments, detected_lang = self.transcribe_with_whisper(audio_path, video_path)

        if not segments or self.stop_processing:
            self.message_callback("❌ Failed to transcribe audio")
            return False

        # Create SRT file
        success = self.create_srt(segments, output_srt, translate, detected_lang)

        if self.stop_processing:
            return False

        # Cleanup temporary files (but keep cache)
        if not self.use_cache or not self.get_cache_paths(video_path)[0] == audio_path:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                self.message_callback("🧹 Cleaned up temporary files")

        return success

    def estimate_processing_time(self, duration, file_size_mb):
        """Estimate processing time based on video properties"""
        try:
            # Base time estimates (very rough)
            if self.device == "cuda":
                base_factor = 0.1  # GPU is ~10x faster
            else:
                base_factor = 1.0

            model_factors = {
                'tiny': 0.5, 'base': 1.0, 'small': 2.0, 'medium': 4.0, 'large': 8.0
            }

            model_factor = model_factors.get(self.whisper_model_size, 1.0)
            estimated_seconds = duration * base_factor * model_factor

            # Add time for audio extraction and file operations
            estimated_seconds += min(30, file_size_mb * 0.5)

            if estimated_seconds < 60:
                return f"{estimated_seconds:.0f} seconds"
            else:
                minutes = estimated_seconds / 60
                return f"{minutes:.1f} minutes"

        except Exception as e:
            return None

    def real_time_subtitles(self, video_path):
        """Generate subtitles in real-time simulation"""
        self.message_callback("🔄 Starting real-time processing simulation...")
        self.progress_callback({'percentage': 10, 'status': 'Starting real-time mode...'})

        # For demonstration, process the video but simulate real-time output
        audio_path = self.extract_audio(video_path)
        if not audio_path or self.stop_processing:
            return False

        segments, detected_lang = self.transcribe_with_whisper(audio_path, video_path)

        if segments and not self.stop_processing:
            self.message_callback("🎥 Real-time subtitle preview:")
            self.message_callback("-" * 50)

            total_preview = min(5, len(segments))
            for idx, segment in enumerate(segments[:total_preview]):
                if self.stop_processing:
                    break

                progress = 80 + (15 * (idx + 1) / total_preview)
                self.progress_callback({
                    'percentage': progress,
                    'status': f'Previewing segment {idx + 1}/{total_preview}'
                })

                start_time = self.format_time(segment['start'])
                end_time = self.format_time(segment['end'])
                text = segment['text']

                self.message_callback(f"[{start_time} --> {end_time}]")
                self.message_callback(f"   {text}")
                self.message_callback("")

                # Simulate timing delay
                time.sleep(1)

            if not self.stop_processing:
                self.message_callback("📝 Real-time processing complete!")
                self.message_callback("💡 In a full implementation, this would sync with video playback")
                self.progress_callback({'percentage': 100, 'status': 'Real-time preview completed'})

        # Cleanup temporary files (but keep cache)
        if not self.use_cache or not self.get_cache_paths(video_path)[0] == audio_path:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        return not self.stop_processing


def main():
    """Main function to run the GUI"""
    # Check dependencies
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox, scrolledtext
        import torch
        import whisper
        import moviepy.editor as mp
        import speech_recognition as sr
        from googletrans import Translator
        import requests

        print("✅ All dependencies loaded successfully")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("pip install openai-whisper moviepy speechrecognition googletrans==4.0.0rc1 pyaudio requests")
        return

    # Create and run GUI
    root = tk.Tk()

    # Set window icon if available
    try:
        # You can add an icon file here if you have one
        # root.iconbitmap("icon.ico")
        pass
    except:
        pass

    app = SubtitlerGUI(root)

    print("🎬 Auto Subtitler v2.0 GUI Started")
    print("GPU Available:", torch.cuda.is_available())
    print("Cache Directory:", app.cache_dir)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()