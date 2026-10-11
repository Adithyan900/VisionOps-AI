# VisionOps AI

VisionOps AI is a real-time touchless visual inspection prototype combining computer vision, hand-gesture interaction, object detection, inspection routing, and a custom PPE detection model.

## Core Workflow

Camera  
→ Hand Tracking  
→ Point at Object  
→ Pinch to Select  
→ General YOLO Object Detection  
→ Inspection Router  
→ Specialist PPE Analysis  
→ Human Verification

## Features

- Real-time webcam computer vision
- MediaPipe hand tracking
- Pointing gesture recognition
- Pinch-to-select interaction
- YOLO object detection
- Object confidence scoring
- Selected-object recognition
- Inspection routing
- Custom PPE-trained YOLO detector
- Person ROI analysis
- Helmet, gloves, boots, goggles and vest analysis
- Missing-PPE detection where supported by the dataset
- Low-light warning
- Human-in-the-loop verification
- Reset and quit controls

## Controls

Q — Quit VisionOps

R — Reset current selection

## Run

```bash
source .venv/bin/activate
python3 main.py