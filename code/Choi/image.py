import cv2
import torch
from collections import defaultdict
import argparse
import sys
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

# Yolov5 디렉토리를 시스템 경로에 추가
yolov5_path = "G:/내 드라이브/yolo_merge_kickboard/yolov5"
sys.path.append(yolov5_path)
from models.experimental import attempt_load
from utils.torch_utils import select_device

# YOLOv5의 detect.py와 유사한 인자 파서 설정
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default="D:/대외 활동 및 프로젝트/YOLO를 활용한 자율주행/git_clone/Tickie_YOLOv5_Accuracy/최서현_이미지_정리/킥보드/1.JPG", help='file/dir/URL/glob')
parser.add_argument('--weights', type=str, default="D:/대외 활동 및 프로젝트/YOLO를 활용한 자율주행/git_clone/Tickie_YOLOv5_Accuracy/code/Choi/model/best.pt", help='weights path')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
opt = parser.parse_args()

# 디바이스 설정
device = select_device(opt.device)

# 모델 로드
model = attempt_load(opt.weights)

# 클래스 이름 리스트
class_names = model.module.names if hasattr(model, 'module') else model.names

# 이미지 파일 읽기
img_path = opt.source
img = cv2.imread(img_path)
assert img is not None, f"Image Not Found {img_path}"

# 이미지 크기 조정 (32의 배수)
img_size = 416  # 모델에 따라 다를 수 있음
h, w = img.shape[:2]
img_resized = cv2.resize(img, (img_size, img_size))

# 객체 감지
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize
img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device

with torch.no_grad():  # 역전파를 비활성화하여 메모리를 절약
    pred = model(img_tensor)[0]

# 결과 처리
pred = pred[0]  # 첫 번째 배치 선택
mask = pred[:, 4] > 0.25  # confidence threshold
pred = pred[mask]  # 마스크를 사용하여 필터링

current_count = defaultdict(int)

for *xyxy, conf, cls in pred:
    class_id = int(cls)
    class_name = class_names[class_id]
    current_count[class_name] += 1

for *xyxy, conf, cls in pred:
    # 원본 이미지에 대한 좌표로 변환
    xyxy = [int(xyxy[0] * w / img_size),
            int(xyxy[1] * h / img_size),
            int(xyxy[2] * w / img_size),
            int(xyxy[3] * h / img_size)]

    # 바운딩 박스를 확장하기 위해 좌표 조정
    bbox_extension = 120  # 확장할 크기 (픽셀 단위)
    xyxy[0] = max(xyxy[0] - bbox_extension, 0)  # x_min
    xyxy[1] = max(xyxy[1] - bbox_extension, 0)  # y_min
    xyxy[2] = min(xyxy[2] + bbox_extension, w)  # x_max
    xyxy[3] = min(xyxy[3] + bbox_extension, h)  # y_max

    label = f'{class_names[int(cls)]} {conf:.2f}'
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
    cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2.imshow('YOLOv5 Detection', img)
cv2.waitKey(0)  # 키 입력을 기다림
cv2.destroyAllWindows()
