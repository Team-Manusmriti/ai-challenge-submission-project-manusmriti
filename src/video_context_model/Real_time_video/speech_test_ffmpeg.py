import sys
import queue
import time
import sounddevice as sd
from vosk import Model, KaldiRecognizer, SetLogLevel
import json
import os
from datetime import datetime

SAMPLE_RATE = 16000
SetLogLevel(0)

MODEL_PATH = "vosk-model-small-en-in-0.4"
if not os.path.isdir(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Please unzip it.")

print(f"[INFO] Loading Vosk model from {MODEL_PATH}...")
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, SAMPLE_RATE)

q = queue.Queue()

def _callback(indata, frames, time, status):
    if status:
        print("[Mic Status]", status, file=sys.stderr)
    q.put(bytes(indata))

try:
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=_callback
    )
    stream.start()
    print("[INFO] Microphone stream started.")
except Exception as e:
    raise RuntimeError(f"Could not open mic: {e}")

def log_speech_transcription(text, is_final=True, log_file="logs/speech_log.txt"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    status = "FINAL" if is_final else "PARTIAL"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] SPEECH ({status}): {text}\n")

def listen_and_transcribe(log_enabled=True):
    if not q.empty():
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "").strip()
            if text:
                print("[Final]", text)
                if log_enabled:
                    log_speech_transcription(text, is_final=True)
            return text
        else:
            partial = json.loads(rec.PartialResult())
            text = partial.get("partial", "").strip()
            if text:
                print("[Partial]", text)
                if log_enabled:
                    log_speech_transcription(text, is_final=False)
            return text
    return None

class SpeechProcessor:
    def __init__(self, log_file="logs/speech_log.txt"):
        self.log_file = log_file
        self.last_transcription_time = 0
        self.transcription_cooldown = 0.5  
        self.command_keywords = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
            'request': ['please', 'can you', 'could you', 'help me', 'i need'],
            'question': ['what', 'when', 'where', 'why', 'how', 'who'],
            'emotion': ['happy', 'sad', 'angry', 'excited', 'tired', 'frustrated'],
            'action': ['start', 'stop', 'pause', 'continue', 'show me', 'tell me']
        }
    
    def process_speech(self):
        current_time = time.time()
        if current_time - self.last_transcription_time < self.transcription_cooldown:
            return None
            
        text = listen_and_transcribe(log_enabled=True)
        if text:
            self.last_transcription_time = current_time
            context = self.categorize_speech(text)
            self.log_speech_context(text, context)
            return {'text': text, 'context': context}
        return None
    
    def categorize_speech(self, text):
        text_lower = text.lower()
        categories = []
        
        for category, keywords in self.command_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['general']
    
    def log_speech_context(self, text, context):
        # Speechlog with contact info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] SPEECH_CONTEXT: {text} | CATEGORIES: {', '.join(context)}\n")

if __name__ == "__main__":
    print("Speak into the mic...")
    speech_processor = SpeechProcessor()
    
    while True:
        result = speech_processor.process_speech()
        if result:
            print(f"Detected: {result['text']} | Context: {result['context']}")
        time.sleep(0.1)