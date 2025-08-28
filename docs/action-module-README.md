# Video Context Detection Module 

## Overview

The Video Context Detection Module integrates three core AI modalities—object detection, speech transcription, and action recognition—to deliver a comprehensive, contextual interpretation of dynamic environments. By fusing visual data from object detection and action recognition with auditory insights from speech processing, it enables real-time scene analysis that goes beyond single-modality limitations. This fusion supports applications in assistive technologies, smart environments, and advanced video analytics, providing a holistic understanding of activities, interactions, and contexts.

The model is designed for efficiency, running on standard hardware with real-time capabilities, and is particularly suited for edge deployment in mobile or IoT scenarios. It processes live video and audio streams, generating interpretable outputs like scene descriptions with confidence scores. Below, each component, the fusion strategy, workflow, logging, visualization, and applications are elaborated in greater depth.

---

## Architecture

```
Video/Camera Input → YoloV8s-oiv7 Detection/Audio Extraction (for pre-recorded videos,using ffmpeg)/Live Audio detection(for real-time) → MobileNet v2 Processing/Audio Processing[Vosk library] → Action/Object Classification → Context Generation → 
Text File Creation → Embedding Storage → Vector Database
```

---
# Core Components

## Object Detection
The object detection module forms the visual foundation of the model, focusing on identifying and localizing objects in real-time video frames. It employs **YoloV8s-oiv7**, a state-of-the-art model from the YOLO (You Only Look Once) family, known for its balance of speed and accuracy in detecting multiple objects simultaneously.

- **Model Architecture and Weights**: YoloV8s-oiv7 is a compact variant of YoloV8s, featuring a scaled-down backbone and head for efficient inference. It uses custom weights from the `YoloV8s-oiv7.pt` file, which are pre-trained on the Open Images V7 dataset. This customization enhances detection for a wide range of everyday objects, improving generalization across diverse scenes.

- **Detection Process**: For each incoming frame, the model performs single-pass inference, outputting bounding boxes (rectangular coordinates defining object locations), class labels (e.g., "person", "chair", "apple"), and confidence scores (probabilities indicating detection reliability, typically thresholded at 0.5 or higher). It also extracts additional features like object size (calculated from bounding box dimensions) and position (relative coordinates within the frame, such as center x/y or quadrant placement).

- **Performance Characteristics**: YoloV8s-oiv7 achieves real-time speeds of around 30–60 FPS on mid-range GPUs, with mean average precision (mAP) scores often exceeding 50% on benchmarks like COCO. The custom `oiv7` weights optimize for open-vocabulary detection, allowing recognition of thousands of object classes without exhaustive retraining.
- **Integration Benefits**: These outputs provide spatial and semantic context, enabling the model to infer relationships like "a person holding a phone" by analyzing proximity and overlap of bounding boxes.

## Action Recognition
To capture temporal dynamics, the action recognition component analyzes sequences of frames to identify ongoing activities, complementing the static object detection with motion-based insights.

- **Model Architecture**: This uses a hybrid CNN-GRU network, where MobileNetV2 serves as the convolutional neural network (CNN) backbone for feature extraction, followed by gated recurrent unit (GRU) layers for temporal modeling. MobileNetV2 is lightweight and efficient, employing depthwise separable convolutions to reduce parameters while maintaining accuracy. The GRU layers process sequential data, capturing dependencies across frames with fewer parameters than LSTMs.

![WhatsApp Image 2025-08-27 at 21 41 47_a94dbeda](https://github.com/user-attachments/assets/a982bcd9-ba05-4dd0-b8a4-edebdbad30af)


- **Input and Processing**: The system buffers video into clips of 16 consecutive frames, each resized to 112×112 pixels for computational efficiency. This clip length balances temporal context with real-time constraints, allowing the model to detect short-duration actions. The CNN extracts spatial features from each frame, while GRUs aggregate them into temporal representations.

- **Output and Classes**: The model predicts from five predefined action classes, adapted from the UCF101 dataset (a benchmark with 101 action categories, including sports, daily activities, and interactions). Examples might include "walking", "eating", "typing", "exercising", or "talking". Outputs include the top action label, confidence scores (softmax probabilities), and ranked predictions for uncertainty handling.

### Demo Videos

#### Action Detection
https://github.com/user-attachments/assets/c8576f47-eb0f-46e9-8845-df22fbd40cd7

#### Object Detection
https://github.com/user-attachments/assets/153b1920-ca40-4de9-b22a-7a6feb507a21

- **Performance and Training**: Trained on subsets of UCF101, which contains over 13,000 clips, the model achieves accuracies above 90% on similar tasks. The MobileNetV2 backbone ensures mobile-friendly inference, with GRU adding temporal robustness for activities spanning multiple frames.

## Speech Processing
The auditory modality handles real-time speech capture and interpretation, adding linguistic and emotional context to the visual data.

- **Toolkit and Model**: It leverages Vosk, an open-source speech recognition toolkit based on Kaldi, supporting offline, lightweight recognition. The compact English model is used, which is optimized for low-resource devices with a small footprint (under 50 MB) and vocabulary focused on general speech.

- **Processing Pipeline**: Audio is streamed from a microphone at a standard sample rate (e.g., 16 kHz). Vosk decodes it in real-time, producing partial transcriptions (incremental results during speech) and final transcriptions (complete utterances). The system then classifies the text into categories like greetings ("hello"), requests ("can you help"), questions ("what time is it"), emotions (detected via keywords like "happy" or "frustrated"), and commands ("turn on the light").

- **Features and Logging**: Timestamps are attached to transcriptions for synchronization with video. The model handles noise robustness through beam search decoding, achieving word error rates (WER) below 10% in quiet environments.

- **Contextual Analysis**: Beyond transcription, rule-based or simple ML classifiers map keywords to intents, enhancing fusion by linking speech to actions (e.g., "I'm hungry" with eating detection).

---

## Modes of Operation

### Real-time Audio Processing

  - Uses sounddevice.RawInputStream to continuously capture microphone audio.

  - Streams audio chunks to Vosk’s KaldiRecognizer.

  - Produces partial and final speech transcriptions live.

  - SpeechProcessor class logs transcriptions and categorizes speech intent.

---

### Recorded Audio Processing

  - Extracts audio from uploaded video files using ffmpeg.

  - Processes WAV audio offline with Vosk’s KaldiRecognizer.

  - Generates full speech transcription from recorded audio.

  - Transcriptions are combined with video frame analysis in the fusion stage.

## Fusion Integration

- Both live and recorded speech transcriptions are fed into the fusion pipeline.

- Fusion combines audio cues with object detection and action recognition for scene understanding.

---

# Fusion Strategy
Fusion is the core innovation, combining multimodal outputs into a unified scene interpretation without requiring complex end-to-end training.

- **Buffering and Synchronization**: Video frames are buffered for action clips, while object detection runs per-frame. Speech is processed continuously, with all outputs timestamp-aligned in a shared buffer.

- **Rule-Based SceneInterpreter**: This module scores combinations against predefined patterns. For instance, detecting "fork" (object), "eating" (action), and "dinner time" (speech) matches an "eating" activity with high confidence. Scoring uses weighted sums: object matches (40%), action confidence (30%), speech keywords (30%). Thresholds filter low-confidence fusions.

- **Output Generation**: Produces a descriptive string like "Person eating at a table (confidence: 85%)", incorporating likelihood metrics. This rule-based approach is interpretable and adaptable, unlike black-box fusion models.

# Model Pipeline / Workflow
The pipeline operates in parallel threads for efficiency, ensuring real-time performance.

1. **Input Capture**: Video from camera (e.g., OpenCV) at 30 FPS; audio via Wave and Vosk.

2. **Object Detection (Frame-level)**: Preprocess (resize, normalize), infer with YoloV8s-oiv7, extract features.

3. **Action Recognition (Clip-level)**: Buffer frames, preprocess clips, infer with CNN-GRU.

4. **Speech Processing (Stream-level)**: Stream audio, recognize with Vosk, classify text.

5. **Fusion Stage (Scene Interpreter)**: Aggregate buffered data, apply rules, compute scores.

6. **Logging and Visualization**: Save outputs, annotate frames.

The diagram illustrates this flow, emphasizing parallelism.

# Model Pipeline Diagram
```
Video Stream
├── Frame-by-frame ──► Object Detection (YoloV8s-oiv7)
│
└── Clip buffer of 16 frames ──► Action Recognition (CNN-GRU)

Audio Stream (Microphone) ──► Speech Processing (Vosk)

(All above converge to)
Fusion-SceneInterpreter
├──► Scene Context Output
└──► Logging & Visualization

``` 

# Logging and Visualization

- **Detailed Logging**: Timestamps all data (e.g., `2025-08-27 21:16: Object: person@0.92, Action: walking@0.85, Speech: 'hello'@greeting, Context: greeting someone@0.90`) into JSON or CSV files for analysis.

- **Real-Time Visualization**: Uses OpenCV to draw bounding boxes, overlay action labels, display speech text, and show context (e.g., green text for high confidence).

- **Error Handling**: Logs issues like low audio quality or model failures, with periodic system stats (e.g., FPS, memory usage).

# Applications

- **Assistive Interfaces**: Monitors elderly activities, alerting for falls or distress by fusing actions with cries for help.

- **Smart Environments**: In homes or robots, enables context-aware responses like adjusting lights during "working" detection.

- **Video Analytics**: Enhances surveillance with audio-visual insights, detecting anomalies like arguments combined with aggressive actions.

---

## For more details and related modules, Please refer:

- [RAG Module](/docs/rag-module-README.md)

- [Speech To Text Module](/docs/speech-module-README.md)

---

*Part of ListenIQ - Built with ❤️ for Samsung EnnovateX 2025*
---
