import whisper
import sounddevice as sd
import numpy as np
import json
import google.generativeai as genai
import scipy.io.wavfile as wav
import tempfile
import cv2
import os
import base64
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import openwakeword 
from openwakeword.model import Model
import pyttsx3
import time


#Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
recording_duration = 5 # seconds
sample_rate = 16000
camera = True
thinking_budget = 0 #0-1024 
wake_threshold = 0.2
chunk_size = 1280 # 1280 samples at 16kHz = 80ms
gemini_temp = 0.1 # low temp = more predictable, safer for robotics control

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found — check your .env file")

sample_objects= [
    "pen",
    "phone",
    "mouse",
    "keyboard",
    "cup",
]

oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
print("Wake word model ready.")


def listen_for_wake_word():
    print("Waiting for 'Hey Jarvis'...")

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=chunk_size
    ) as stream:
        while True:
            audio_chunk, _ = stream.read(chunk_size)
            audio_flat = audio_chunk.flatten()

            prediction = oww_model.predict(audio_flat)
            score = prediction.get("hey_jarvis", 0)

            # Print the score every frame so we can see what's happening
            if score > 0.01:  # only print if there's any signal at all
                print(f"Score: {score:.3f}")

            if score > wake_threshold:
                oww_model.reset()
                return

''''
def listen_for_wake_word():
    with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=chunk_size) as stream:
        while True:
            audio_chunk,_ = stream.read(chunk_size)
            audio_flat = audio_chunk.flatten()
            prediction = oww_model.predict(audio_flat)
            score = prediction.get("hey_jarvis", 0.0)

            if score>wake_threshold:
                oww_model.reset()  # prevent multiple triggers
                return
'''

#Gemini Prompt

SYSTEM_PROMPT = """
You are controlling a 5-DOF robotic arm with a multi-finger hand.
The arm has: base rotation, shoulder, elbow, wrist pitch, wrist rotation, and finger servos.
 
When given a voice command and (optionally) a camera image of the scene,
return ONLY a JSON object describing the action to take. No other text.
 
Available actions:
  pick_up    - pick up a named object (requires target)
  put_down   - place the held object down (target optional: location)
  move_to    - move toward an object or position (requires target)
  open_hand  - open fingers fully
  close_hand - close fingers fully
  wave       - wave hand as a greeting
  look_at    - turn and orient toward an object (requires target)
  stop       - halt all movement immediately
  unknown    - command unclear or object not found
 
Response format (JSON only):
{
  "action": "<action>",
  "target": "<object name or null>",
  "confidence": <0.0 to 1.0>,
  "reply": "<short spoken reply, max 10 words>",
  "notes": "<optional: spatial observations about the target>"
}
 
Safety rule: if confidence is below 0.5, always use action "unknown".
"""

whisper_model=whisper.load_model("base")
tts = pyttsx3.init()

voices = tts.getProperty('voices')
for voice in voices:
    if "samantha" in voice.name.lower():
        tts.setProperty('voice', voice.id)
        break

tts.setProperty('rate', 150)  # Set speaking rate
tts.setProperty('volume', 1.0)  # Set volume

def speak(text):
    print(f"Jarvis says: {text}")
    tts.say(text)
    tts.runAndWait()

genai.configure(api_key = GEMINI_API_KEY)
gemini = genai.GenerativeModel(
    model_name = "gemini-robotics-er-1.5-preview",
    system_instruction= SYSTEM_PROMPT
    )


def capture_frame():
    if not camera:
        return None
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("WARNING: Webcam not available!")
        return None
    
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("WARNING: Failed to capture frame")
        return None
    
    _,buffer = cv2.imencode('.jpg', frame)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return b64
    
    
def record_audio(seconds=recording_duration, fs=sample_rate):
    print("Recording audio...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Done listening")
    return audio.flatten()

def transcribe_audio(audio, fs=sample_rate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        wav.write(tmp_path, fs, (audio * 32767).astype(np.int16))
    result = whisper_model.transcribe(tmp_path)
    os.remove(tmp_path)
    return result['text'].strip()


def get_command(transcription, image_b64=None):
    content = []

    if image_b64:
        content.append({
            "mime_type": "image/jpeg",
            "data": image_b64
        })
        
        content.append(
            f'Voice command: "{transcription}"\n\n'
            f'Look at the image, identify what objects are present, '
            f'then respond to the command.'
        )
    else:
        content.append(
            f'Voice command: "{transcription}"\n\n'
            f'No camera image available. Do your best with the command alone.'
        )

    response = gemini.generate_content(
        content,
        generation_config=genai.GenerationConfig(
            temperature=gemini_temp,  # low = predictable, safe outputs
        ),
    )

    raw = response.text.strip()

    # strip markdown code fences if Gemini wraps the JSON
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1])
 
    try:
        command = json.loads(raw)
    except json.JSONDecodeError:
        command = {
            "action": "unknown",
            "target": None,
            "confidence": 0.0,
            "reply": "Sorry, I didn't understand that.",
            "notes": f"Raw response was: {raw}"
        }
 
    return command

def handle_command(command):
    print()
    print("─" * 44)
    print(f"  Action     : {command.get('action')}")
    print(f"  Target     : {command.get('target')}")
    print(f"  Confidence : {command.get('confidence')}")
    print(f"  Reply      : {command.get('reply')}")
    if command.get("notes"):
        print(f"  Notes      : {command.get('notes')}")
    print("─" * 44)
    print()
    #ser.write((json.dumps(command) + '\n').encode())


def main():
    print("=== Robot Voice Controller — Gemini Robotics-ER 1.5 ===")
    print()
    print("The AI sees a live webcam frame with each command.")
    print("It identifies objects visually — no hardcoded list needed.")
    print()
    print("Try saying:")
    print("  'pick up the red cup'")
    print("  'wave at me'")
    print("  'open your hand'")
    print("  'stop everything'")
    print()
    speak("Hello, I am Jarvis. I am ready to take your commands.")
 
    while True:
        listen_for_wake_word()

        speak("Yes? What can I do for you?")

        audio = record_audio()
        transcript = transcribe_audio(audio)

        if not transcript:
            speak("I didn't catch that, try again.")
            continue

        print(f"You said: \"{transcript}\"")
        print("thinking...")

        image = capture_frame()

        command = get_command(transcript, image_b64=image)
        handle_command(command)
        speak(command.get("reply", "Done"))

        time.sleep(0.5)
 
 
if __name__ == "__main__":
    main()
 
