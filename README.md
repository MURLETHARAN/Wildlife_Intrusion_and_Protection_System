# Wildlife_Intrusion_and_Protection_System
Wildlife Monitoring & Alert System using AI + IoT
Overview

This project is an AI-based real-time wildlife monitoring system that detects animals using a webcam and machine learning. When an animal is detected, it triggers an alert system with sound + live web dashboard updates.

It is designed for applications like:

Farmland protection 
Wildlife intrusion detection 
Smart surveillance systems 

Features
Real-time webcam monitoring
AI-based classification using MobileNetV2
Human detection filtering (reduces false alarms)
Motion-based triggering system
Audio alert on detection
Live Flask web dashboard
Auto-updating UI (every 1 second)
Cooldown system to avoid repeated alerts
System Architecture
Camera Input
     ↓
Motion Detection (OpenCV)
     ↓
Face Detection (Haar Cascade)
     ↓
AI Classification (MobileNetV2)
     ↓
Decision:
   ├── Human → Ignore / Show status
   └── Animal → Trigger Alert
                     ↓
        Sound Alert + Web Update
