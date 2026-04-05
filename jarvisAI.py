import whisper
import sounddevice as sd
import numpy as np
import json
import scipy.io.wavfile as wav
import tempfile
import cv2
import os
import base64
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from openwakeword.model import Model
import time
from ultralytics import YOLO
import subprocess

#elevenlabs api = https://elevenlabs.io/app/developers/analytics/usage
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

#gemini api = https://aistudio.google.com/api-keys
#from google import genai
#from google.genai import types
import google.generativeai as genai
import serial


#source venv/bin/activate

#Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLAB_API_KEY = os.getenv("ELEVENLAB_API_KEY")
recording_duration = 5 # seconds
sample_rate = 16000
camera = True
thinking_budget = 0 #0-1024  
wake_threshold = 0.3
chunk_size = 1280 # 1280 samples at 16kHz = 80ms
gemini_temp = 0.1 # low temp = more predictable, safer for robotics control
yolo_model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 model for object detection
yolo_confidence = 0.5  # Confidence threshold for YOLO detections


try:
    hand_ser = serial.Serial('/dev/cu.usbmodem11301', 9600, timeout=1)
    time.sleep(2)
    print("Hand Arduino connected.")
except Exception as e:
    print(f"Hand Arduino not connected: {e}")
    hand_ser = None

try:
    arm_ser = serial.Serial('/dev/cu.usbmodem11302', 9600, timeout=1)
    time.sleep(2)
    print("Arm Arduino connected.")
except Exception as e:
    print(f"Arm Arduino not connected: {e}")
    arm_ser = None

mode = "autonomous"  # or "teleop"

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
    dtype='int16',
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

CONVERSATION_PROMPT = """
You are Jarvis, an AI assistant controlling a robotic arm. Your personality is
modelled after TARS from Interstellar — dry wit, playful sarcasm, and occasional
dark humour, but always competent and helpful when it matters.

You have a humour setting of 75%. Act accordingly.

You can have full natural conversations. When the user wants you to do something
physical with the arm (pick up, move, wave, grab, open hand, etc.), respond
naturally AND include this exact tag on its own line at the end:
[ARM_TRIGGER]

If the user says something like "copy my movements", "mirror me", "follow my hand",
include this tag on its own line: [TELEOP_MODE]

If the user says "stop copying", "take over", "do it yourself",
include this tag: [AUTO_MODE]

Otherwise just reply conversationally. Keep replies under 20 words unless the
user is clearly asking for a longer explanation.

Examples:
  User: "what's 2 plus 2"
  Reply: "Four. Though I suspect you knew that and just wanted to hear my voice."

  User: "pick up the cup"
  Reply: "On it. Try not to knock anything over while I'm concentrating."
  [ARM_TRIGGER]

  User: "are you smarter than me"
  Reply: "I'm tactfully going to say we have different strengths."
"""

COMMAND_PROMPT = """
You are controlling a 5-DOF robotic arm with a multi-finger hand.
Return ONLY a JSON object. No personality. No extra text.

Available actions: pick_up, put_down, move_to, open_hand, close_hand,
wave, look_at, stop, unknown.

Response format:
{
  "action": "<action>",
  "target": "<object name or null>",
  "confidence": <0.0 to 1.0>,
  "reply": "<short spoken reply, max 10 words>",
  "notes": "<optional spatial observations>"
}

Safety rule: confidence below 0.5 must use action unknown.
"""

whisper_model=whisper.load_model("base")

AI_voice = ElevenLabs(api_key=ELEVENLAB_API_KEY)

def speak(text):
    print(f"Jarvis says: {text}")
    try:
        audio = AI_voice.text_to_speech.convert(
            text=text,
            voice_id="pNInz6obpgDQGcFmaJgB",
            model_id="eleven_flash_v2_5"
        )

        play(audio)
        time.sleep(0.3)

    except Exception as e:
        print(f"ElevenLabs error: {e}")
        #Fall back to Mac say if ElevenLabs fails
        subprocess.run(["say", "-v", "Samantha", "-r", "175", text])
    

genai.configure(api_key = GEMINI_API_KEY)

convo_gemini = genai.GenerativeModel(
    model_name = "gemini-robotics-er-1.5-preview",
    system_instruction= CONVERSATION_PROMPT
    )

chat_session = convo_gemini.start_chat(history=[])

command_gemini = genai.GenerativeModel(
    model_name = "gemini-robotics-er-1.5-preview",
    system_instruction= COMMAND_PROMPT
    )


def capture_frame():
    if not camera:
        return None, []
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("WARNING: Webcam not available!")
        return None, []
    
    time.sleep(0.5)  # Warm up the camera

    for _ in range(5):
        cap.read()  # Discard initial frames to allow auto-adjustment
    
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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (173, 216, 230), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

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


def get_command(transcription, image_b64=None,detected_objects=None, gemini_instance=None):
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
            detection_text = f'Objects detected in view:\n{objects_info}\n\n'
        else:
            detection_text = f'No objects detected in view.\n\n'

        content.append(
            f'{detection_text}\n\n'
            f'Voice command: "{transcription}"\n\n'
        )
    else:
        content.append(
            f'Voice command: "{transcription}"\n\n'
            f'No camera image available. Do your best with the command alone.'
        )


    response = gemini_instance.generate_content(
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
        command["detections"] = detected_objects if detected_objects else []
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

def handle_command(command,arm_ser=None):
    action = command.get("action")
    detections = command.get("detections", [])

    print()
    print("-" * 44)
    print(f"  Action     : {action}")
    print(f"  Target     : {command.get('target')}")
    print(f"  Confidence : {command.get('confidence')}")
    print(f"  Reply      : {command.get('reply')}")
    if command.get("notes"):
        print(f"  Notes      : {command.get('notes')}")
    if(detections):
        print(f"  YOLO saw   : {', '.join([d['label'] for d in detections])}")
    else:  
        print(f"  YOLO saw   : nothing")
    print("-" * 44)
    print()


def main():
    global mode

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
    speak("Hello, I am Jarvis. Say 'Hey Jarvis' to get my attention, and then give me a command.")
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

          

            try:
                convo_response = chat_session.send_message(transcript)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    speak("I'm being rate limited. Give me a moment.")
                    time.sleep(45)
                    continue
                else:
                    speak("Something went wrong. Try again.")
                    print(f"Error: {e}")
                    continue

            convo_text = convo_response.text.strip()

            if "[TELEOP_MODE]" in convo_text:
                mode = "teleop"
                speak("Copying your movements")
            if "[AUTO_MODE]" in convo_text:
                mode = "autonomous"
                speak("I'll take it from here")

            arm_triggered = "[ARM_TRIGGER]" in convo_text
            reply_text = convo_text.replace("[ARM_TRIGGER]", "").strip() 

            speak(reply_text)

            if arm_triggered:
                image,detections = capture_frame()

                if detections:
                    labels = ", ".join([d['label'] for d in detections])
                    print(f"I can see {labels}.")


                command = get_command(transcript, image_b64=image,detected_objects=detections, gemini_instance=command_gemini)
                handle_command(command)
                speak(command.get("reply", "Done"))

                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nShutting down...")
        sd.stop()
        speak("Shutting down. Goodbye.")
        cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 
