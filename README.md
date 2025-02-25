## Real-Time Emotion Detection Using DeepFace & OpenCV  

This project implements a **real-time emotion detection system** using **DeepFace** and **OpenCV**. It captures live video from a webcam, detects faces, analyzes emotions, and provides response statements based on detected emotions.  

### Features  
- **Real-time Face Detection** – Uses OpenCV’s Haar cascade classifier to detect faces in live video.  
- **Emotion Recognition** – Analyzes facial expressions using DeepFace and identifies dominant emotions.  
- **Dynamic Response Generation** – Provides predefined responses based on detected emotions.  
- **Emotion Statistics & Reports** – Counts occurrences of different emotions and identifies the most frequent one.  

### Technologies Used  
- **Python**  
- **OpenCV**  
- **DeepFace**  
- **Streamlit (Optional for UI)**  

### How It Works  
1. Captures video using OpenCV.  
2. Detects faces in the frame using Haar cascades.  
3. Extracts and analyzes emotions using DeepFace.  
4. Displays detected emotions on the video frame.  
5. Logs and reports detected emotions, showing the most frequent one.  

### Installation  
```bash
pip install opencv-python deepface numpy
```

### Usage  
```bash
python emotion_detection.py
```
Press **'q'** to exit the program.  

### Future Enhancements  
- **Enhanced Accuracy** – Incorporate deep learning-based face detection (e.g., MTCNN).  
- **GUI Interface** – Develop a Streamlit-based interactive dashboard.  
- **Multiple Face Detection** – Support for detecting emotions of multiple people simultaneously.  

This project provides an interactive way to analyze emotions in real time, useful for applications in mental health monitoring, user engagement analysis, and AI-driven interactions. 🚀
