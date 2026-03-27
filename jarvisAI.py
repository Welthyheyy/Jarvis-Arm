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
from openwakeword.model import Model
import pyttsx3
import time
from ultralytics import YOLO

#source venv/bin/activate

#Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
recording_duration = 5 # seconds
sample_rate = 16000
camera = True
thinking_budget = 0 #0-1024  
wake_threshold = 0.2
chunk_size = 1280 # 1280 samples at 16kHz = 80ms
gemini_temp = 0.1 # low temp = more predictable, safer for robotics control
yolo_model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 model for object detection
yolo_confidence = 0.5  # Confidence threshold for YOLO detections

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
    blocksize=chunk_size
    ) as stream:
        while True:
            audio_chunk,_ = stream.read(chunk_size)
            audio_flat = audio_chunk.flatten()
            prediction = oww_model.predict(audio_flat)
            score = prediction.get("hey_jarvis", 0.0)

            if score>wake_threshold:
                oww_model.reset()  # prevent multiple triggers
                return

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
        return None, []
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("WARNING: Webcam not available!")
        return None, []
    
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("WARNING: Failed to capture frame")
        return None, []
    
    results = yolo_model(frame, conf=yolo_confidence,verbose=False)[0]
    detected_objects = []

    for result in results.boxes:
        label = yolo_model.names[int(result.cls)]
        conf = float(result.conf)
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        detected_objects.append({
            "label": label,
            "confidence": round(conf,2),
            "center_x":cx,
            "center_y":cy,
            "bbox": [x1, y1, x2, y2]
        })

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (173, 216, 230), 2)

    cv2.imshow("Jarvis", frame)
    cv2.waitKey(1)

    _, buffer = cv2.imencode('.jpg', frame)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    return image_b64, detected_objects

    
    
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


def get_command(transcription, image_b64=None,detected_objects=None):
    content = []

    if image_b64:
        content.append({
            "mime_type": "image/jpeg",
            "data": image_b64
        })

        if detected_objects:
            objects_info = "\n".join([
                f"- {obj['label']} (conf: {obj['confidence']:0%}"
                f"pos: {obj['center_x']},{obj['center_y']})"
                for obj in detected_objects
            ]) 
            detection_text = f"Objects detected in view:\n{objects_info}\n\n"
        else:
            detection_text = "No objects detected in view.\n\n"

        content.append(
            f'{detection_text}\n\n',
            f'Voice command: "{transcription}"\n\n'
        )
    else:
        content.append(
            f'Voice command: "{transcription}"\n\n'
            f'No camera image available. Do your best with the command alone.'
        )


    response = gemini.generate_content(
        content,
        generation_config=genai.GenerationConfig(
            temperature=gemini_temp,
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
            "notes": f"Raw response was: {raw}",
            "detections": []
        }
 
    return command

def handle_command(command):
    detections = command.get("detections", [])

    print()
    print("─" * 44)
    print(f"  Action     : {command.get('action')}")
    print(f"  Target     : {command.get('target')}")
    print(f"  Confidence : {command.get('confidence')}")
    print(f"  Reply      : {command.get('reply')}")
    if command.get("notes"):
        print(f"  Notes      : {command.get('notes')}")
    if(detections):
        print(f"  YOLO saw   : {', '.join([d['label'] for d in detections])}")
    else:  
        print(f"  YOLO saw   : nothing")
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
    try:
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

            image,detections = capture_frame()

            command = get_command(transcript, image_b64=image,detected_objects=detections)
            handle_command(command)
            speak(command.get("reply", "Done"))

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nShutting down...")
        speak("Shutting down. Goodbye.")
        cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 
