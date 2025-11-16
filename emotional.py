#!/usr/bin/env python3
from deepface import DeepFace
import cv2
import subprocess
import time
import os
import random
import argparse

parser = argparse.ArgumentParser(description='Emotion implies a song')
parser.add_argument('--cam-id', type=int, default=0, 
                    help='ID of the webcam to use (default: 0)')
args = parser.parse_args()

music_base_path = "music"
scan_wait_time = 1

#cap = cv2.VideoCapture(args.cam_id)
cap = cv2.VideoCapture(2)
current_emotion = "neutral"
emotion_start_time = time.time()
is_playing = False
mpv_process = None
current_song = "" 
face_detected = False
cv2.namedWindow("Capture", cv2.WINDOW_GUI_NORMAL)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    if is_playing:
        if mpv_process and mpv_process.poll() is not None:
            is_playing = False
            mpv_process = None
            emotion_start_time = time.time()
        song_name = os.path.splitext(current_song)[0]
        text = f"{current_emotion}: {song_name}"
    else:
        try:
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=True)
            detected_emotion = result[0]["dominant_emotion"]
            if 'region' in result[0]:
                face_region = result[0]['region']
                x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_detected = True
            else:
                face_detected = False
                pass
        except:
            detected_emotion = current_emotion
            face_detected = False
        if detected_emotion != current_emotion:
            current_emotion = detected_emotion
            emotion_start_time = time.time()
        duration = time.time() - emotion_start_time
        if duration > scan_wait_time and face_detected:
            folder = os.path.join(music_base_path, current_emotion)
            if os.path.isdir(folder):
                songs = [f for f in os.listdir(folder) if f.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a'))]
                if songs:
                    current_song = random.choice(songs)
                    song_path = os.path.join(folder, current_song)
                    mpv_process = subprocess.Popen(["mpv", "--osc", "--script-opts=osc-visibility=always","--force-window=yes" , song_path])
                    is_playing = True
            emotion_start_time = time.time()
        if face_detected:
            text = f"{current_emotion}"
        else:
            text = ""


    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Capture", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        if mpv_process:
            mpv_process.terminate()
        break

cap.release()
cv2.destroyAllWindows()
