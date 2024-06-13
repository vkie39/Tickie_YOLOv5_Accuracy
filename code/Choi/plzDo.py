import cv2
import torch
import matplotlib.pyplot as plt

# 모델 로드
model_load = torch.load('G:/내 드라이브/yolo_merge_kickboard/yolov5', 'custom', path='G:/내 드라이브/yolo_merge_kickboard/yolov5/runs/train/yolo_cutPaste_kickboard2/weights/best.pt')

image_file = "G:/내 드라이브/example/kickboard.jpg"

img = cv2.imread(image_file)  # 이미지 파일 로드
if img is not None:
    results = model_load(img)  # 모델을 사용하여 이미지 분석
    results = results.pandas().xyxy[0][['name', 'xmin', 'ymin', 'xmax', 'ymax']]

    for num, i in enumerate(results.values):
        cv2.putText(img, i[0], (int(i[1]), int(i[2])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.rectangle(img, (int(i[1]), int(i[2])), (int(i[3]), int(i[4])), (0, 0, 255), 3)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(imgRGB)
    plt.show()
else:
    print("Can't open image file.")  # 이미지 파일을 열 수 없음
