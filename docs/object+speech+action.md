# Overview
This model fuses three AI modalities—object detection, speech transcription, and action recognition—to achieve enriched scene understanding. By combining visual cues from object detection and action recognition with auditory information from speech processing, it generates a contextual interpretation of the environment and activities.

# Components

## Object Detection
- Utilizes YOLOv8s with custom weights (yolov8s-oiv7.pt) for fast and accurate real-time detection.  
- Detects multiple objects per frame, providing bounding boxes, labels, and confidence scores.  
- Extracts object features such as size and position for scene analysis.

## Action Recognition
- Employs a custom CNN-GRU deep neural network with MobileNetv2 backbone and GRU temporal layers.  
- Processes clips of 16 consecutive frames resized to 112x112.  
- Predicts one of five predefined activities derived from the UCF101 dataset.  
- Outputs action label with confidence scores and top predictions.

## Speech Processing
- Uses the Vosk speech recognition toolkit with a pre-trained compact English model.  
- Captures live microphone audio, decoding speech into text.  
- Classifies spoken content into contextual categories such as greetings, requests, questions, emotions, and commands.  
- Logs partial and final transcriptions for analysis.

# Fusion Strategy
- Buffers video frames for temporal action recognition input.  
- Runs object detection on individual frames concurrently.  
- Continuously processes streaming speech input for transcription and context extraction.  
- Combines modality outputs via a rule-based SceneInterpreter that scores matching objects, actions, and speech keywords against predefined activity patterns (e.g., eating, working, exercising).  
- Generates a high-level scene context string with confidence measures indicating activity likelihood.

# Model Pipeline / Workflow

The overall processing pipeline consists of three parallel modalities—object detection, action recognition, and speech processing—which are fused to generate a comprehensive scene understanding. The key stages in the model workflow are as follows:

1. **Input Capture**  
   - Video frames are continuously captured from the camera stream.  
   - Audio is captured live from a microphone in parallel.

2. **Object Detection (Frame-level)**  
   - Each video frame is preprocessed and fed into the YOLOv8s object detection model with custom weights.  
   - Multiple objects per frame are detected along with bounding boxes, labels, confidence scores, and spatial features (size, position).

3. **Action Recognition (Clip-level)**  
   - Frames are buffered into clips of 16 consecutive frames and resized (112x112) for input to the CNN-GRU action recognition model.  
   - The model predicts one of the five predefined action classes for each clip, outputting action labels with confidence scores.

4. **Speech Processing (Stream-level)**  
   - The Vosk speech recognition toolkit continuously transcribes live microphone audio.  
   - The transcribed speech is classified into contextual categories such as greetings, requests, emotions, and commands.  
   - Partial and final transcriptions are logged alongside timestamps.

5. **Fusion Stage (Scene Interpreter)**  
   - The model buffers and matches outputs from the object detection, action recognition, and speech transcription modules.  
   - A rule-based SceneInterpreter evaluates matching object labels, actions, and speech keywords against predefined activity patterns (e.g., eating, working).  
   - It generates a high-level scene context description with associated confidence metrics indicating activity likelihood.

6. **Logging and Visualization**  
   - Detailed fusion results with timestamps, detected objects, actions, transcriptions, context, and confidence values are logged for offline analysis.  
   - Live video frames are annotated with object bounding boxes, action labels, speech snippets, and inferred scene context to provide real-time visual feedback.  
   - System events and errors are logged to support debugging and maintenance.

# Model Pipeline Diagram

flowchart TD

    A [Video Stream] -->|Frame-by-frame| B[Object Detection (YOLOv8s-oiv7)]
    A -->|Clip buffer of 16 frames| C[Action Recognition (CNN-GRU)]
    D[Audio Stream (Microphone)] --> E[Speech Processing (Vosk)]

    B --> F[Fusion-SceneInterpreter]
    C --> F
    E --> F

    F --> G[Scene Context Output]
    F --> H[Logging & Visualization]



# Logging and Visualization
- Logs detailed timestamped fusion results including detected objects, predicted actions, transcribed speech, scene context, and confidence scores into persistent session files.  
- Annotates live video frames with bounding boxes, action labels, speech snippets, and inferred scene context.  
- Provides periodic system event and error logging for troubleshooting.

# Applications
- Real-time assistive interfaces monitoring human activities and interactions.  
- Context-aware smart environments or robotics requiring multimodal perception.  
- Enhanced video analytics combining visual and audio cues.

![Built with ❤ for Samsung EnnovateX 2025 AI Challenge](./docs/assets/logo.png)

