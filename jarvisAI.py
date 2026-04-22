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
from kinematics import px_to_table_landscape, calculate_arm_angles, load_calibration
from faster_whisper import WhisperModel
import subprocess
import webrtcvad

#elevenlabs api = https://elevenlabs.io/app/developers/analytics/usage
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

#gemini api = https://aistudio.google.com/api-keys
#from google import genai
#from google.genai import types
import google.generativeai as genai
import serial

#groq api = https://console.groq.com/keys
from groq import Groq

#source venv/bin/activate

#Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLAB_API_KEY = os.getenv("ELEVENLAB_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
recording_duration = 5 # seconds
sample_rate = 16000
camera = True
wake_threshold = 0.3
chunk_size = 1280 # 1280 samples at 16kHz = 80ms
gemini_temp = 0.1 # low temp = more predictable, safer for robotics control
yolo_model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 model for object detection
yolo_confidence = 0.5  # Confidence threshold for YOLO detections
mode = "autonomous"  # or "teleop"
conversationTemp = 0.75 #higher for more personality
thinking_budget = 150 #max tokens for Groq response
CONVO_TIMEOUT = 12 # seconds to respond before sleeping
humour = 75 # percentage of humour in Jarvis's personality


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

#Groq Prompt
CONVERSATION_PROMPT = """
You are hood Jarvis, an AI assistant with a toronto mans slang, controlling a robotic arm. Your personality is
modelled after TARS from Interstellar — dry wit, playful sarcasm, and occasional
dark humour, but always competent and helpful when it matters.

You have a humour setting of {humour}percent. Act accordingly.

You are able to have a continuous conversation — reply back with follow-up questions occasionally when you see fit.
Remember context from earlier in the conversation and build on it naturally.
Ask follow-up questions occasionally. React to what was just said rather than
giving generic responses.

Reply length should match the question — short questions get short answers,
detailed questions get detailed answers. Never pad with filler. Keep it concise.

When the user wants something physical with the arm, include [ARM_TRIGGER] on
its own line at the end. For teleop mode include [TELEOP_MODE]. To return to
autonomous include [AUTO_MODE].

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

whisper_model=WhisperModel("base", device="cpu",compute_type="int8")  # Load Whisper model for speech recognition

AI_voice = ElevenLabs(api_key=ELEVENLAB_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

conversation_history=[
    {"role": "system", "content": CONVERSATION_PROMPT}
]

def convo_message(transcript):
    conversation_history.append({"role": "user", "content": transcript})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_history,
        temperature=conversationTemp,
        max_tokens=thinking_budget
    )

    reply = response.choices[0].message.content
    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    return reply


def speak(text):
    print(f"Jarvis says: {text}")
    try:
        audio = AI_voice.text_to_speech.convert(
            text=text,
            voice_id="pNInz6obpgDQGcFmaJgB",
            model_id="eleven_flash_v2_5"
        )

        play(audio)
        time.sleep(0.1)

    except Exception as e:
        print(f"ElevenLabs error: {e}")
        #Fall back to Mac say if ElevenLabs fails
        subprocess.run(["say", "-v", "Samantha", "-r", "175", text])
    

genai.configure(api_key = GEMINI_API_KEY)


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

    
    
def record_audio(fs=sample_rate,silence_duration=1.2, max_duration=10):
    vad = webrtcvad.Vad(2)  # Aggressiveness mode (0-3)

    frame_ms = 30  # Frame size in ms
    frame_size = int(fs * frame_ms / 1000)
    
    silence_frames_needed = int(silence_duration * 1000 / frame_ms)
    max_frames = int(max_duration * 1000 / frame_ms)

    print("Listening... (speak now)")

    frames = []
    silence_counter = 0
    speech_started = False

    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', blocksize=frame_size) as stream:

        for _ in range(max_frames):
            audio_chunk, _ = stream.read(frame_size)
            audio_flat = audio_chunk.flatten().tobytes()
            frames.append(audio_chunk.flatten())

            is_speech = vad.is_speech(audio_flat, fs)

            if is_speech:
                speech_started = True
                silence_counter = 0
            elif speech_started:
                silence_counter += 1
                if silence_counter >= silence_frames_needed:
                    print("Done listening")
                    break
    audio = np.concatenate(frames).astype(np.float32) / 32767.0
    return audio

    

def transcribe_audio(audio, fs=sample_rate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        wav.write(tmp_path, fs, (audio * 32767).astype(np.int16))

    segments,_ = whisper_model.transcribe(tmp_path,beam_size=1)
    os.remove(tmp_path)
    return " ".join([s.text for s in segments]).strip()


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

def handle_command(command,ser=None):
    action = command.get("action")
    detections = command.get("detections", [])
    target = command.get("target")

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

    if not ser or not ser.is_open:
        print("No serial connection for arm control. Skipping physical action.")
        return
    
    if action == "pick up" and target:
        match = None
        for obj in detections:
            if obj['label'].lower() == target.lower():
                match = obj
                break

        if match:
            cx = match['center_x']
            cy = match['center_y']

            x, y,z = px_to_table_landscape(cx, cy)

            angles = calculate_arm_angles(x, y,z)
            
            ser.write(f"B:{int(angles['base_deg'])}\n".encode())
            time.sleep(0.1)
            ser.write(f"S:{int(angles['shoulder_deg'])}\n".encode())
            time.sleep(0.1)
            ser.write(f"E:{int(angles['elbow_deg'])}\n".encode())
            time.sleep(0.1)

            time.sleep(0.8)  # wait for arm to reach position
            ser.write(b"I:0\n")   # close index
            ser.write(b"M:0\n")   # close middle
            ser.write(b"T:0\n")   # close thumb
            ser.write(b"R:0\n")   # close ring
            ser.write(b"P:0\n")   # close pinky

        else:
            speak(f"I can see the scene but could not find the {target}.")

    elif action == "open_hand":
        ser.write(b"I:180\n")   # open index
        ser.write(b"M:180\n")   # open middle
        ser.write(b"T:180\n")   # open thumb
        ser.write(b"R:180\n")   # open ring
        ser.write(b"P:180\n")   # open pinky

    elif action == "put_down":
        ser.write(b"I:180\n")   # open index
        ser.write(b"M:180\n")   # open middle
        ser.write(b"T:180\n")   # open thumb
        ser.write(b"R:180\n")   # open ring
        ser.write(b"P:180\n")   # open pinky
        time.sleep(0.5)
        ser.write(b"B:90\n")   # move base to neutral
        ser.write(b"S:90\n")   # move shoulder to neutral
        ser.write(b"E:90\n")   # move elbow to neutral

    elif action == "close_hand":
        ser.write(b"I:0\n")   # close index
        ser.write(b"M:0\n")   # close middle
        ser.write(b"T:0\n")   # close thumb
        ser.write(b"R:0\n")   # close ring
        ser.write(b"P:0\n")   # close pinky
    
    elif action == "wave":
        for _ in range(3):
            ser.write(b"B:60\n")  # wave left
            time.sleep(0.5)
            ser.write(b"B:120\n") # wave right
            time.sleep(0.5)
        ser.write(b"B:90\n")   # return to center
    
    elif action == "stop":
        pass


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

            speak("Yeah?")

            last_interaction = time.time()
            in_conversation = True


            while in_conversation:
                time_since_last = time.time() - last_interaction

                if time_since_last > CONVO_TIMEOUT:
                    speak("Going to sleep. Wake me up when you need me.")
                    in_conversation = False
                    break

                audio = record_audio()
                transcript = transcribe_audio(audio)

                if not transcript or len(transcript.strip())<2:
                    continue

                print(f"You said: \"{transcript}\"")
                print("thinking...")
                last_interaction = time.time()

                try:
                    convo_text = convo_message(transcript)
                except Exception as e:
                    if "429" in str(e):
                        speak("Give me a moment.")
                        time.sleep(10)
                        continue
                    speak("Something went wrong.")
                    print(f"Error: {e}")
                    continue

                if "[TELEOP_MODE]" in convo_text:
                    mode = "teleop"
                    print("Teleoperation mode enabled")
                elif "[AUTO_MODE]" in convo_text:
                    mode = "autonomous"
                    print("Autonomous mode enabled")

                reply_text = (convo_text
                    .replace("[ARM_TRIGGER]", "")
                    .replace("[TELEOP_MODE]", "")
                    .replace("[AUTO_MODE]", "").strip()) 


                speak(reply_text)
                last_interaction = time.time()
            

                arm_triggered = "[ARM_TRIGGER]" in convo_text

                if arm_triggered:
                    image,detections = capture_frame()

                    if detections:
                        labels = ", ".join([d['label'] for d in detections])
                        print(f"I can see {labels}.")
                        
                    command = get_command(transcript, image_b64=image,detected_objects=detections, gemini_instance=command_gemini)
                    handle_command(command,ser=hand_ser)
                    speak(command.get("reply", "Done"))
                    last_interaction = time.time()

    except KeyboardInterrupt:
        print("\nShutting down...")
        sd.stop()
        speak("Shutting down. Goodbye.")
        cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 
