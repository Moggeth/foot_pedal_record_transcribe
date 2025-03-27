import os
import time
import threading
import datetime
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import keyboard
import pyperclip
import pystray
from PIL import Image, ImageDraw
from openai import OpenAI
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# ================================
# Configuration and Global Variables
# ================================
HOTKEY = "F9"             # Remappable hotkey (change as needed)
SAMPLE_RATE = 44100       # in Hz
CHANNELS = 1              # mono recording

recording = False         # True while recording
recorded_frames = []      # List to store audio chunks
stream = None             # The active InputStream

# Variables for adaptive stop
stop_requested = False    # Set to True when hotkey is released
stop_request_time = None  # Time when stop was requested
silence_start = None      # Time when silence was first detected

# Silence detection parameters
SILENCE_THRESHOLD = 0.01   # RMS value below which is considered silence
MIN_SILENCE_DURATION = 0.5  # Seconds of continuous silence required
MAX_WAIT_AFTER_STOP_REQUEST = 2.0  # Maximum seconds to wait after release

OUTPUT_FOLDER = "recordings"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Initialize OpenAI client (ensure OPENAI_API_KEY is set in your environment)
client = OpenAI()

# ================================
# Audio Recording Functions
# ================================
def audio_callback(indata, frames, time_info, status):
    """Callback for sounddevice InputStream."""
    global silence_start, stop_requested, stop_request_time
    if status:
        print(Fore.RED + f"[Status] {status}")
    recorded_frames.append(indata.copy())

    if stop_requested:
        current_time = time.time()
        rms = np.sqrt(np.mean(indata**2))
        # Check if current block is silent.
        if rms < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = current_time
            elif current_time - silence_start >= MIN_SILENCE_DURATION:
                print(Fore.GREEN + "Silence detected for sufficient duration. Stopping recording.")
                threading.Thread(target=stop_recording, daemon=True).start()
        else:
            silence_start = None
        # If maximum waiting time elapsed, force stop.
        if stop_request_time is not None and (current_time - stop_request_time) >= MAX_WAIT_AFTER_STOP_REQUEST:
            print(Fore.RED + "Maximum wait time reached. Stopping recording.")
            threading.Thread(target=stop_recording, daemon=True).start()

def start_recording():
    """Starts recording audio."""
    global recording, recorded_frames, stream, stop_requested, silence_start, stop_request_time
    if recording:
        return
    print(Fore.GREEN + "Starting recording...")
    recording = True
    stop_requested = False
    silence_start = None
    stop_request_time = None
    recorded_frames = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback)
    stream.start()

def stop_recording():
    """Stops recording and processes the audio."""
    global recording, stream
    if not recording:
        return
    print(Fore.MAGENTA + "Stopping recording...")
    recording = False
    stream.stop()
    stream.close()
    process_recording()

def process_recording():
    """Saves the recorded audio to a WAV file and initiates transcription."""
    if not recorded_frames:
        print(Fore.RED + "No audio recorded.")
        return
    audio_data = np.concatenate(recorded_frames, axis=0)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = os.path.join(OUTPUT_FOLDER, f"recording_{timestamp}.wav")
    wavfile.write(wav_filename, SAMPLE_RATE, audio_data)
    print(Fore.CYAN + f"Saved recording to {wav_filename}")
    
    transcription_text = transcribe_audio(wav_filename)
    if transcription_text:
        pyperclip.copy(transcription_text)
        print(Fore.CYAN + "Transcription copied to clipboard.")
        txt_filename = os.path.join(OUTPUT_FOLDER, f"transcription_{timestamp}.txt")
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(transcription_text)
        print(Fore.GREEN + f"Transcription saved to {txt_filename}")
    else:
        print(Fore.RED + "Transcription failed.")

# ================================
# Transcription Function
# ================================
def transcribe_audio(filename):
    """
    Transcribes the given audio file using the GPT-4o transcription API.
    Uses the OpenAI client as per the provided usage example.
    """
    try:
        with open(filename, "rb") as audio_file:
            print(Fore.BLUE + "Sending audio to transcription API...")
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )
            print(Fore.GREEN + "Transcription received.")
            return transcription.text
    except Exception as e:
        print(Fore.RED + f"Transcription error: {e}")
        return None

# ================================
# Hotkey Event Handlers with Adaptive Stop
# ================================
def on_hotkey_press(e):
    """Starts recording when the hotkey is pressed."""
    if not recording:
        start_recording()

def on_hotkey_release(e):
    """Flags stop request so the audio callback can flush remaining audio."""
    global stop_requested, stop_request_time
    if recording:
        print(Fore.YELLOW + "Stop requested; initiating adaptive flush...")
        stop_requested = True
        stop_request_time = time.time()

keyboard.on_press_key(HOTKEY, on_hotkey_press)
keyboard.on_release_key(HOTKEY, on_hotkey_release)

# ================================
# System Tray Icon (Optional)
# ================================
def create_tray_image():
    """Creates a simple tray icon image."""
    width, height = 64, 64
    image = Image.new('RGB', (width, height), color="blue")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 48, 48), fill="white")
    return image

def quit_app(icon, item):
    """Quits the application."""
    icon.stop()
    os._exit(0)

def run_tray():
    """Runs the system tray icon."""
    image = create_tray_image()
    menu = pystray.Menu(pystray.MenuItem("Quit", quit_app))
    icon = pystray.Icon("PushToTalkRecorder", image, "Push-to-Talk Recorder", menu)
    icon.run()

tray_thread = threading.Thread(target=run_tray, daemon=True)
tray_thread.start()

# ================================
# Main Loop
# ================================
print(Fore.LIGHTBLUE_EX + f"Push-to-Talk Recorder running on Windows.\nHold '{HOTKEY}' to record audio. Transcriptions will be copied to the clipboard and saved in '{OUTPUT_FOLDER}' folder.")
keyboard.wait()
