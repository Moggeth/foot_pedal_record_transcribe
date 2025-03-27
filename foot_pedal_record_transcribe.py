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
from openai import OpenAI  # Updated usage per OpenAI's example

# ================================
# Configuration and Global Variables
# ================================

# Remappable hotkey – change this string to your preferred key (e.g. "F9", "ctrl+shift+p", etc.)
HOTKEY = "F9"

# Audio recording settings
SAMPLE_RATE = 44100  # in Hz
CHANNELS = 1         # mono recording

# Global flags and data storage
recording = False
recorded_frames = []  # List to store chunks of audio
stream = None         # Will hold the active InputStream

# Output folder for recordings and transcriptions
OUTPUT_FOLDER = "recordings"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Initialize the OpenAI client (make sure OPENAI_API_KEY is set in your environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_WORK"))  # This automatically uses the OPENAI_API_KEY environment variable

# ================================
# Audio Recording Functions
# ================================

def audio_callback(indata, frames, time_info, status):
    """Callback function for sounddevice InputStream."""
    if status:
        print(f"Recording status: {status}")
    recorded_frames.append(indata.copy())

def start_recording():
    """Starts the audio recording."""
    global recording, recorded_frames, stream
    if recording:
        return  # Already recording, do nothing
    print("Starting recording...")
    recording = True
    recorded_frames = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback)
    stream.start()

def stop_recording():
    """Stops the audio recording and processes the audio."""
    global recording, stream
    if not recording:
        return  # Not recording, nothing to stop
    print("Stopping recording...")
    recording = False
    stream.stop()
    stream.close()
    process_recording()

def process_recording():
    """Saves the recorded audio to a WAV file and initiates transcription."""
    if not recorded_frames:
        print("No audio recorded.")
        return
    # Combine all recorded chunks into a single array
    audio_data = np.concatenate(recorded_frames, axis=0)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = os.path.join(OUTPUT_FOLDER, f"recording_{timestamp}.wav")
    wavfile.write(wav_filename, SAMPLE_RATE, audio_data)
    print(f"Saved recording to {wav_filename}")
    
    # Transcribe the audio file using GPT-4o transcription API via OpenAI client
    transcription_text = transcribe_audio(wav_filename)
    if transcription_text:
        pyperclip.copy(transcription_text)
        print("Transcription copied to clipboard.")
        txt_filename = os.path.join(OUTPUT_FOLDER, f"transcription_{timestamp}.txt")
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(transcription_text)
        print(f"Transcription saved to {txt_filename}")
    else:
        print("Transcription failed.")

# ================================
# Transcription Function
# ================================

def transcribe_audio(filename):
    """
    Sends the audio file to the GPT-4o transcription API using OpenAI's client 
    and returns the transcription text.
    """
    try:
        with open(filename, "rb") as audio_file:
            print("Sending audio to transcription API...")
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )
            print("Transcription received.")
            return transcription.text
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

# ================================
# Hotkey Event Handlers
# ================================

def on_hotkey_press(e):
    """Called when the hotkey is pressed."""
    # Start recording only if not already recording
    if not recording:
        start_recording()

def on_hotkey_release(e):
    """Called when the hotkey is released."""
    # Stop recording only if recording
    if recording:
        stop_recording()

# Register global hotkey events.
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
    """Sets up and runs the system tray icon."""
    image = create_tray_image()
    menu = pystray.Menu(pystray.MenuItem("Quit", quit_app))
    icon = pystray.Icon("PushToTalkRecorder", image, "Push-to-Talk Recorder", menu)
    icon.run()

# Run the system tray icon in a separate thread so it doesn't block hotkey handling.
tray_thread = threading.Thread(target=run_tray, daemon=True)
tray_thread.start()

# ================================
# Main Loop
# ================================

print(f"Push-to-Talk Recorder running on Windows.\nHold '{HOTKEY}' to record audio. Transcriptions will be copied to the clipboard and saved in the '{OUTPUT_FOLDER}' folder.")
keyboard.wait()  # Wait indefinitely for hotkey events
