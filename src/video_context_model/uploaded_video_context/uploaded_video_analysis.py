import os
import cv2
import wave
import json
import subprocess
from fusion_model import FusionCompanion
from vosk import Model, KaldiRecognizer


model_path = "demo_upload_video\\vosk-model-small-en-in-0.4"

def extract_audio_from_video(video_path, audio_path):
    command = [
        'ffmpeg', '-y', '-i', video_path,  
        '-vn',                         
        '-acodec', 'pcm_s16le',       
        '-ar', '16000',               
        '-ac', '1',                   
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def recognize_speech_from_audio(audio_filepath, model_path=model_path):
    wf = wave.open(audio_filepath, "rb")
    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    transcript_parts = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcript_parts.append(result.get('text', ''))
    final_result = json.loads(rec.FinalResult())
    transcript_parts.append(final_result.get('text', ''))

    wf.close()
    transcript = ' '.join(filter(None, transcript_parts))
    return transcript

def process_uploaded_video(video_path):
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    extract_audio_from_video(video_path, audio_path)
    speech_text = recognize_speech_from_audio(audio_path, model_path=model_path)
    print("Speech recognized:", speech_text)

    speech_text = recognize_speech_from_audio(audio_path, model_path=model_path)



    cap = cv2.VideoCapture(video_path)
    companion = FusionCompanion()  

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = companion.process_frame(frame, external_speech_text=speech_text)
        companion.logger.log_fusion_result(
            results['objects'], results['action'], results['speech'], 
            scene_context="N/A", confidence_scores={}
        )

        
        annotated = companion.draw_annotations(frame, results)
        cv2.imshow("Analysis with Audio", annotated)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

uploaded_video_path = input("Enter path to the uploaded video file: ")
process_uploaded_video(uploaded_video_path)
