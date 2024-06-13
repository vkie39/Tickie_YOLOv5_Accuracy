import cv2
import torch
import time
import sys
from pathlib import Path

# YOLOv5 모델 로드
model = torch.hub.load('D:\대외 활동 및 프로젝트\YOLOv5 정확도 향상 (졸업작품)\git_clone\Tickie_YOLOv5_Accuracy/yolov5', 'custom', path='D:\대외 활동 및 프로젝트\YOLOv5 정확도 향상 (졸업작품)\git_clone\Tickie_YOLOv5_Accuracy\알고리즘 개발\best.pt')  # 학습된 모델 경로

# 동영상 캡처 (웹캠 또는 비디오 파일)
cap = cv2.VideoCapture(0)  # 웹캠 사용 시
# cap = cv2.VideoCapture('path/to/video.mp4')  # 비디오 파일 사용 시

fps = 24
frame_count = 0
detected_frames = 0
distance_threshold = 100  # 거리 임계값 설정 (픽셀 단위, 튜닝 필요)
detection_duration = 1  # 감지 지속 시간 (초)
required_detections = 20  # 최소 감지 프레임 수
slow_down_threshold = 0.1  # 감속 판단을 위한 변화율 임계값
observation_duration = 3  # 3초 동안 측정
previous_detections = []

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 객체 감지 수행
    results = model(frame)
    detections = results.xyxy[0].numpy()

    close_object_detected = False
    max_size = 0

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        width = x2 - x1
        height = y2 - y1
        size = width * height

        if size > max_size:
            max_size = size

        # 거리 추정 (간단히 객체 크기로 거리 추정)
        if width > distance_threshold or height > distance_threshold:
            close_object_detected = True

    previous_detections.append(max_size)
    if len(previous_detections) > fps * observation_duration:
        previous_detections.pop(0)

    if close_object_detected:
        detected_frames += 1
    else:
        detected_frames = max(0, detected_frames - 1)  # 연속 감지 유지를 위해 감지되지 않을 때도 조금씩 감소

    # 변화율 계산
    if len(previous_detections) >= fps * observation_duration:
        initial_size = previous_detections[0]
        current_size = previous_detections[-1]
        size_change_rate = (current_size - initial_size) / initial_size

        if size_change_rate > slow_down_threshold:
            print("감속")
        else:
            if detected_frames >= required_detections:
                print("가까이 있는 객체 감지됨 - 멈춤")
                # 멈춤 동작 수행
                detected_frames = 0  # 멈춘 후 초기화
            else:
                print("정상 주행")

    # 결과 영상 출력 (디버깅 용도)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()