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


#Configurations
GEMINI_API_KEY = ""
recording_duration = 5 # seconds
sample_rate = 16000
camera = True
thinking_budget = 0 #0-1024 

sample_objects= [
    "pen",
    "phone",
    "mouse",
    "keyboard",
    "cup",
]


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


def generate_robot_command(transcription, image_b64=None):
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
            temperature=0.1,  # low = predictable, safe outputs
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
 
    while True:
        input("Press Enter to speak (Ctrl+C to quit)...")
 
        # capture scene BEFORE recording so image is current
        frame_b64 = capture_frame()
 
        audio = record_audio()
        transcript = transcribe(audio)
 
        if not transcript:
            print("Nothing detected, try again.")
            continue
 
        print(f'You said: "{transcript}"')
        print("Sending to Gemini Robotics-ER...")
 
        command = get_command(transcript, frame_b64)
        handle_command(command)
 
 
if __name__ == "__main__":
    main()
 
