import os
import cv2
import time
import torch
import json as js
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from speech_test_ffmpeg import SpeechProcessor
from action_model import CNN_GRU, load_action_model, predict_action, preprocess_frames, log_action_prediction
from object_detection_v8s import extract_objects_info, analyze_scene_objects

YOLO_MODEL = "yolov8s-oiv7.pt"   
IMG_SZ = 128  
CLIP_LEN = 16

class FusionLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.fusion_log = os.path.join(log_dir, "fusion_complete_log.txt")
        self.session_log = os.path.join(log_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        self.session_data = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "duration_sec": None,
            "entries": []
        }
        self.log_session_start()
    
    def log_session_start(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.session_log, 'w', encoding='utf-8') as f:
            f.write(f"=== ListenIQ SESSION STARTED ===\n")
            f.write(f"Start Time: {timestamp}\n")
            f.write(f"{'='*50}\n\n")
    
    def log_fusion_result(self, objects, action, speech, scene_context, confidence_scores):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        objects_str = "None"
        if objects:
            objects_str = " | ".join([f"{obj['label']}({obj['conf']:.2f})" for obj in objects])
        

        action_text = action['text']
        if isinstance(action_text, dict):
             action_text = action_text.get('action', str(action_text))

        action_str = f"{action_text}({action['confidence']:.2f})" if action_text else "None"
        speech_str = speech['text'] if speech['text'] else "None"
        speech_context = ", ".join(speech['context']) if speech['context'] else "None"
        
        log_entry = (
            f"[{timestamp}]\n"
            f"  OBJECTS: {objects_str}\n"
            f"  ACTION: {action_str}\n"
            f"  SPEECH: {speech_str}\n"
            f"  SPEECH_CONTEXT: {speech_context}\n"
            f"  SCENE_UNDERSTANDING: {scene_context}\n"
            f"  CONFIDENCE_SCORES: {confidence_scores}\n"
            f"  {'-'*50}\n"
        )
        
        with open(self.fusion_log, 'a', encoding='utf-8') as f:
            f.write(log_entry)

        with open(self.session_log, 'a', encoding='utf-8') as f:
            f.write(log_entry)

        self.session_data["entries"].append({
            "timestamp": timestamp,
            "objects": objects,
            "action": action,
            "speech": speech,
            "scene_context": scene_context,
            "confidence_scores": confidence_scores
        })
    
    def log_system_event(self, event_type, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {event_type.upper()}: {message}\n"
        
        with open(self.session_log, 'a', encoding='utf-8') as f:
            f.write(log_entry)

    def save_session_summary(self):
        self.session_data["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start = datetime.strptime(self.session_data["start_time"], "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(self.session_data["end_time"], "%Y-%m-%d %H:%M:%S")
        self.session_data["duration_sec"] = (end - start).total_seconds()

        json_path = os.path.join(self.log_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_full.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            js.dump(self.session_data, f, indent=4)

        self.log_system_event("INFO", f"Full session summary saved to {json_path}")

class SceneInterpreter:
    def __init__(self, logger):
        self.logger = logger
        self.context_history = deque(maxlen=15)
        
        self.activity_patterns = {
            'eating': {
                'objects': ['cup', 'bottle', 'food', 'apple', 'sandwich', 'plate', 'spoon', 'fork'],
                'actions': ['eating', 'drinking'],
                'speech_keywords': ['hungry', 'thirsty', 'delicious', 'food', 'eat', 'drink']
            },
            'working': {
                'objects': ['laptop', 'computer', 'keyboard', 'mouse', 'book', 'paper'],
                'actions': ['typing', 'writing', 'reading'],
                'speech_keywords': ['work', 'project', 'meeting', 'deadline', 'busy', 'computer']
            },
            'exercising': {
                'objects': ['sports', 'ball', 'equipment'],
                'actions': ['punch', 'swing', 'jump', 'run'],
                'speech_keywords': ['exercise', 'workout', 'tired', 'training', 'fitness']
            },
            'entertainment': {
                'objects': ['tv', 'phone', 'remote'],
                'actions': ['playing', 'watching'],
                'speech_keywords': ['fun', 'game', 'movie', 'music', 'entertainment']
            },
            'communication': {
                'objects': ['phone', 'computer'],
                'actions': ['talking', 'typing'],
                'speech_keywords': ['call', 'message', 'talk', 'hello', 'conversation']
            }
        }
    
    def interpret_scene(self, objects, action, speech):
        object_labels = [obj['label'].lower() for obj in objects if obj['conf'] > 0.4]
        action_text = action['text'].lower() if action['text'] else ""
        speech_text = speech['text'].lower() if speech['text'] else ""
        
        activity_scores = {}
        for activity, patterns in self.activity_patterns.items():
            score = 0
            
            object_matches = sum(1 for obj in object_labels if any(pattern in obj for pattern in patterns['objects']))
            score += object_matches * 0.4
            
            action_matches = sum(1 for pattern in patterns['actions'] if pattern in action_text)
            score += action_matches * 0.3
            
            speech_matches = sum(1 for keyword in patterns['speech_keywords'] if keyword in speech_text)
            score += speech_matches * 0.3
            
            if score > 0:
                activity_scores[activity] = score
        
        context_parts = []
        
        if activity_scores:
            primary_activity = max(activity_scores, key=activity_scores.get)
            confidence = min(activity_scores[primary_activity], 1.0)
            context_parts.append(f"Activity: {primary_activity} (conf: {confidence:.2f})")
        
        if object_labels:
            high_conf_objects = [obj['label'] for obj in objects if obj['conf'] > 0.7]
            if high_conf_objects:
                context_parts.append(f"Key objects: {', '.join(high_conf_objects[:3])}")
        
        if action['text'] and action['confidence'] > 0.5:
            context_parts.append(f"Action: {action['text']}")
        
        if speech['text'] and speech['context']:
            context_parts.append(f"Speech intent: {', '.join(speech['context'])}")
        
        if context_parts:
            scene_context = " | ".join(context_parts)
        else:
            scene_context = "Observing environment - no clear activity detected"
        
        return scene_context, activity_scores

class FusionCompanion:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = FusionLogger()
        self.scene_interpreter = SceneInterpreter(self.logger)
        
        self.setup_models()
        
        self.frame_buffer = deque(maxlen=CLIP_LEN)
        self.frame_count = 0
        self.last_log_time = 0
        self.log_interval = 2.0  
        
    def setup_models(self):
        self.logger.log_system_event("INIT", "Starting model initialization...")
        try:
            self.det_model = YOLO(YOLO_MODEL)
            self.logger.log_system_event("INIT", f"Object detection model loaded: {YOLO_MODEL}")
            
            self.action_model = load_action_model("best_model.pt", device=self.device, num_classes=5, hidden_size=128)
            if self.action_model:
                self.logger.log_system_event("INIT", "Action recognition model loaded successfully")
            else:
                self.logger.log_system_event("ERROR", "Failed to load action recognition model")
            
            self.speech_processor = SpeechProcessor()
            self.logger.log_system_event("INIT", "Speech processing initialized")
            
            self.logger.log_system_event("INIT", "All models initialized successfully")
        except Exception as e:
            self.logger.log_system_event("ERROR", f"Model initialization failed: {str(e)}")
            raise
    
    def process_frame(self, frame):
        self.frame_count += 1
        results = {
            'objects': [],
            'action': {'text': None, 'confidence': 0.0},
            'speech': {'text': None, 'context': []}
        }

        
        try:
            det_results = self.det_model(frame, imgsz=IMG_SZ, conf=0.25, verbose=False)
            results['objects'] = extract_objects_info(det_results[0])
        except Exception as e:
            self.logger.log_system_event("ERROR", f"Object detection failed: {str(e)}")
        
        self.frame_buffer.append(frame)
        if self.action_model and len(self.frame_buffer) == CLIP_LEN:
            try:
                clip_frames = list(self.frame_buffer)
                clip_tensor = preprocess_frames(clip_frames, seq_len=CLIP_LEN, resize=(112, 112))
                action_result = predict_action(self.action_model, clip_tensor, device=self.device)

                if isinstance(action_result, tuple):
                    results['action']['text'], results['action']['confidence'] = action_result
                else:
                    if isinstance(action_result, dict):
                        results['action']['text'] = action_result.get("action", None)
                        results['action']['confidence'] = action_result.get("confidence", 0.0)
                    else:
                        if isinstance(action_result, tuple):
                            results['action']['text'], results['action']['confidence'] = action_result
                        else:
                            results['action']['text'] = action_result
                            results['action']['confidence'] = 0.0


                if results['action']['text'] and results['action']['text'] != "Model not loaded":
                    log_action_prediction(results['action']['text'], results['action']['confidence'])

            except Exception as e:
                self.logger.log_system_event("ERROR", f"Action recognition failed: {str(e)}")
        
        try:
            speech_result = self.speech_processor.process_speech()
            if speech_result:
                results['speech'] = speech_result
        except Exception as e:
            self.logger.log_system_event("ERROR", f"Speech processing failed: {str(e)}")

        print(f"[DEBUG] Final results: {results}")
        return results
    
    def draw_annotations(self, frame, results):
        for obj in results['objects']:
            x1, y1, x2, y2 = map(int, obj['xyxy'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{obj['label']} {obj['conf']:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if results['action']['text']:
            color = (255, 0, 0) if results['action']['confidence'] > 0.5 else (0, 0, 255)
            cv2.putText(frame, f"ACTION: {results['action']['text']} ({results['action']['confidence']:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if results['speech']['text']:
            cv2.putText(frame, f"SPEECH: {results['speech']['text'][:50]}...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            if results['speech']['context']:
                cv2.putText(frame, f"CONTEXT: {', '.join(results['speech']['context'])}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        return frame
    
    def log_fusion_data(self, results):
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            scene_context, activity_scores = self.scene_interpreter.interpret_scene(
                results['objects'], results['action'], results['speech']
            )
            confidence_scores = {
                'objects': sum(obj['conf'] for obj in results['objects']) / max(len(results['objects']), 1),
                'action': results['action']['confidence'],
                'speech': 1.0 if results['speech']['text'] else 0.0,
                'scene': max(activity_scores.values()) if activity_scores else 0.0
            }
            self.logger.log_fusion_result(
                results['objects'], results['action'], results['speech'],
                scene_context, confidence_scores
            )
            self.last_log_time = current_time
            return scene_context
        return None
    
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.log_system_event("ERROR", "Could not open webcam")
            raise RuntimeError("Could not open webcam")
        
        self.logger.log_system_event("START", "ListenIQ started - Press 'q' to quit")
        print("[INFO] ListenIQ started. Press 'q' to quit.")
        
        start_time = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if time.time() - start_time > 60:
                    self.logger.log_system_event("STOP", "60 seconds elapsed, stopping detection.")
                    print("[INFO] 60 seconds elapsed, stopping detection.")
                    break
                
                results = self.process_frame(frame)
                annotated_frame = self.draw_annotations(frame, results)
                scene_context = self.log_fusion_data(results)
                
                if scene_context:
                    words = scene_context.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line + word + " ") <= 60:
                            current_line += word + " "
                        else:
                            lines.append(current_line.strip())
                            current_line = word + " "
                    if current_line:
                        lines.append(current_line.strip())
                    
                    y_offset = frame.shape[0] - 80
                    for i, line in enumerate(lines[-3:]):
                        cv2.putText(annotated_frame, line, (10, y_offset + i * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.imshow("ListenIQ", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            self.logger.log_system_event("STOP", "User interrupted execution")
        except Exception as e:
            self.logger.log_system_event("ERROR", f"Runtime error: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.log_system_event("STOP", "ListenIQ session ended")
            self.logger.save_session_summary()
    
    def run_on_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.process_frame(frame)
                annotated = self.draw_annotations(frame, results)
                cv2.imshow("Fusion Test", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    companion = FusionCompanion()
    companion.run()