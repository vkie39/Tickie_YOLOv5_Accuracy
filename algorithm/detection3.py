import cv2
import torch
import time
from collections import defaultdict
from pathlib import Path
import argparse
import sys
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
# Yolov5 디렉토리를 시스템 경로에 추가
yolov5_path = "G:\내 드라이브\yolo_merge_layPeople2\yolov5"
sys.path.append(yolov5_path)
from models.experimental import attempt_load
from utils.torch_utils import select_device

# YOLOv5의 detect.py와 유사한 인자 파서 설정
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default="C:/Users/Min/Downloads/B-Bad Chillin - Kickboard Wheelie_Manual & Sturz (1).mp4", help='file/dir/URL/glob, 0 for webcam')
parser.add_argument('--weights', type=str, default='G:/내 드라이브/yolov5_1/runs/train/exp9/weights/best.pt', help='weights path')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
opt = parser.parse_args()

# 디바이스 설정
device = select_device(opt.device)

# 모델 로드
model = attempt_load(opt.weights)

# 클래스 이름 리스트
class_names = model.module.names if hasattr(model, 'module') else model.names

# 비디오 캡처
cap = cv2.VideoCapture(opt.source)

# 타임스탬프 초기화
start_time = time.time()

# 프레임 간격 설정 (초당 4번 체크)
frame_interval = 0.1  # 1초 간격

# 객체 카운트 초기화
kickboard_count = 0
lying_person_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 크기 조정 (32의 배수)
    img_size = 640# 모델에 따라 다를 수 있음
    frame_resized = cv2.resize(frame, (img_size, img_size))
    

    # 객체 감지
    img = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize
    img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():  # 역전파를 비활성화하여 메모리를 절약
        pred = model(img)[0]

    # 결과 처리
    pred = pred[pred[:, :, 4] > 0.25]  # confidence threshold
    current_count = defaultdict(int)

    for *xyxy, conf, cls in pred:
        class_id = int(cls)
        class_name = class_names[class_id]
        current_count[class_name] += 1

    # 현재 시간
    current_time = time.time()
  
    if class_name == "사람이 탄 킥보드":  # 클래스 이름이 "사람이 탄 킥보드"일 경우
        print("사람이 탄 킥보드 인식!")
                   
    elif class_name == "누워있는 사람":  # 클래스 이름이 "누워있는 사람"일 경우   
        print("누워있는 사람 발견!")
                   

    # 결과 프레임 표시 (선택 사항)
    for *xyxy, conf, cls in pred:
        label = f'{class_names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('YOLOv5 Detection', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 눌러 종료
        break

cap.release()
cv2.destroyAllWindows()
