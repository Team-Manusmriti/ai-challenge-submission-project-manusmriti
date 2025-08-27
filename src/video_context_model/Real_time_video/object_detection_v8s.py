import os
import cv2
import time
import torch
from collections import deque
from ultralytics import YOLO
from datetime import datetime
from action_model import load_action_model, preprocess_frames, predict_action, log_action_prediction

YOLO_MODEL = "yolov8s-oiv7.pt"   
model = YOLO("yolov8s-oiv7.pt")  
model.export(format='onnx', imgsz=112)
IMG_SZ = 112
CLIP_LEN = 16

det_model = YOLO(YOLO_MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
action_model = load_action_model("best_model.pt", device=device)

frame_buffer = deque(maxlen=CLIP_LEN)

def log_object_detection(objects, log_file="logs/object_detection_log.txt"):
    #Object Detection Log
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    with open(log_file, 'a', encoding='utf-8') as f:
        if objects:
            objects_str = " | ".join([f"{obj['label']}({obj['conf']:.2f})" for obj in objects])
            f.write(f"[{timestamp}] OBJECTS: {objects_str}\n")
        else:
            f.write(f"[{timestamp}] OBJECTS: None detected\n")

def extract_objects_info(detection_result):
    objects = []
    for box in detection_result.boxes:
        x1, y1, x2, y2 = box.xyxy.tolist()[0]
        conf = float(box.conf)
        cls = int(box.cls)
        label = detection_result.names[cls]
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        objects.append({
            'label': label, 
            'xyxy': (x1, y1, x2, y2), 
            'conf': conf,
            'area': area,
            'center': (center_x, center_y),
            'size': (width, height)
        })
    
    return objects

def analyze_scene_objects(objects):
    if not objects:
        return {"scene_type": "empty", "object_count": 0}
    
    # Categorize objects
    categories = {
        'person': ['person', 'man', 'woman', 'child'],
        'furniture': ['chair', 'table', 'sofa', 'bed', 'desk'],
        'electronics': ['laptop', 'computer', 'phone', 'tv', 'monitor'],
        'food_drink': ['cup', 'bottle', 'food', 'apple', 'sandwich', 'plate'],
        'vehicles': ['car', 'bicycle', 'motorcycle', 'truck', 'bus'],
        'animals': ['cat', 'dog', 'bird', 'horse']
    }
    
    detected_categories = []
    for obj in objects:
        for category, items in categories.items():
            if any(item in obj['label'].lower() for item in items):
                if category not in detected_categories:
                    detected_categories.append(category)
    
    # Determine scene 
    scene_type = "general"
    if 'person' in detected_categories and 'electronics' in detected_categories:
        scene_type = "work_environment"
    elif 'person' in detected_categories and 'food_drink' in detected_categories:
        scene_type = "dining_scene"
    elif 'vehicles' in detected_categories:
        scene_type = "transportation"
    elif 'furniture' in detected_categories:
        scene_type = "indoor_living"
    
    return {
        "scene_type": scene_type,
        "object_count": len(objects),
        "categories": detected_categories,
        "high_confidence_objects": [obj['label'] for obj in objects if obj['conf'] > 0.7]
    }

def log_scene_analysis(scene_info, log_file="logs/scene_analysis_log.txt"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] SCENE: {scene_info['scene_type']} | ")
        f.write(f"OBJECTS: {scene_info['object_count']} | ")
        f.write(f"CATEGORIES: {', '.join(scene_info['categories'])} | ")
        f.write(f"HIGH_CONF: {', '.join(scene_info['high_confidence_objects'])}\n")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("[INFO] Enhanced Object + Action Detection Started")
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if time.time() - start_time > 60:  
            print("[INFO] 1 minute elapsed, stopping...")
            break
        
        # Object Detection
        results = det_model(frame, imgsz=IMG_SZ, conf=0.25, verbose=False)
        r = results[0]
        objects = extract_objects_info(r)
        
        if frame_count % 30 == 0:
            log_object_detection(objects)
            scene_info = analyze_scene_objects(objects)
            log_scene_analysis(scene_info)

        for obj in objects:
            x1, y1, x2, y2 = map(int, obj['xyxy'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{obj['label']} {obj['conf']:.2f}",
                       (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 1)

        # Action Recognition
        frame_buffer.append(frame)
        action_label = None
        action_confidence = 0.0

        if len(frame_buffer) == CLIP_LEN and action_model:
            clip_frames = list(frame_buffer)
            clip_tensor = preprocess_frames(clip_frames, seq_len=CLIP_LEN, resize=(112, 112))
            action_result = predict_action(action_model, clip_tensor, device=device)
            
            if isinstance(action_result, tuple):
                action_label, action_confidence = action_result
            else:
                action_label = action_result
                action_confidence = 0.0
            
            if action_label and action_label != "Model not loaded":
                log_action_prediction(action_label, action_confidence)

        if action_label:
            color = (255, 0, 0) if action_confidence > 0.5 else (0, 0, 255)
            cv2.putText(frame, f"ACTION: {action_label} ({action_confidence:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if frame_count % 30 == 0 and objects:
            scene_info = analyze_scene_objects(objects)
            cv2.putText(frame, f"SCENE: {scene_info['scene_type']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Enhanced Object + Action Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()