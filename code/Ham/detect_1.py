# 필요한 라이브러리 임포트
import torch
import cv2
import time
from collections import defaultdict
from pathlib import Path
import argparse
import sys
import pathlib

pathlib.PosixPath = pathlib.WindowsPath
# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/82104/Desktop/best.pt')

# 비디오 파일 경로 설정
video_path = 'C:/Users/82104/Desktop/layVideo.mp4'
cap = cv2.VideoCapture(video_path)

# 비디오 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 모델을 사용하여 객체 감지
    results = model(frame)
    
    # 결과 추출
    labels = results.xyxyn[0][:, -1].numpy()
    names = results.names
    
    # 'lay' 객체 감지 여부 확인
    for label in labels:
        if names[int(label)] == 'layPeople':
            print('정지')
            break
    
    # 감지된 객체 표시 (원할 경우)
    #results.show()
    
    # cv2를 사용하여 프레임 표시 (원할 경우)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

# 비디오 해제
cap.release()
cv2.destroyAllWindows()
