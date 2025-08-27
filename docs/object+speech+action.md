## Overview
This model fuses three AI modalities—object detection, speech transcription, and action recognition—to achieve enriched scene understanding. By combining visual cues from object detection and action recognition with auditory information from speech processing, it generates a contextual interpretation of the environment and activities.

## Components
## Object Detection
-Utilizes YOLOv8s with custom weights (yolov8s-oiv7.pt) for fast and accurate real-time detection.

-Detects multiple objects per frame, providing bounding boxes, labels, and confidence scores.

-Extracts object features such as size and position for scene analysis.

## Action Recognition
-Employs a custom CNN-GRU deep neural network with MobileNetv2 backbone and GRU temporal layers.

-Processes clips of 16 consecutive frames resized to 112x112.

-Predicts one of five predefined activities derived from the UCF101 dataset.

-Outputs action label with confidence scores and top predictions.

## Speech Processing
-Uses the Vosk speech recognition toolkit with a pre-trained compact English model.

-Captures live microphone audio, decoding speech into text.

-Classifies spoken content into contextual categories such as greetings, requests, questions, emotions, and commands.

-Logs partial and final transcriptions for analysis.

## Fusion Strategy
-Buffers video frames for temporal action recognition input.

-Runs object detection on individual frames concurrently.

-Continuously processes streaming speech input for transcription and context extraction.

-Combines modality outputs via a rule-based SceneInterpreter that scores matching objects, actions, and speech keywords against predefined activity patterns (e.g., eating, working, exercising).

-Generates a high-level scene context string with confidence measures indicating activity likelihood.

## Logging and Visualization
-Logs detailed timestamped fusion results including detected objects, predicted actions, transcribed speech, scene context, and confidence scores into persistent session files.

-Annotates live video frames with bounding boxes, action labels, speech snippets, and inferred scene context.

-Provides periodic system event and error logging for troubleshooting.

## Applications
-Real-time assistive interfaces monitoring human activities and interactions.

-Context-aware smart environments or robotics requiring multimodal perception.

-Enhanced video analytics combining visual and audio cues.


![Built with ❤ for Samsung EnnovateX 2025 AI Challenge](docs/assets/logo.png)
