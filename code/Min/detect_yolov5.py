import sys
#yolov5의 사용을 위해서 내 컴퓨터 안에 있ㄴ
yolov5_path = "G:/내 드라이브/yolov5"
#model = torch.hub.load('ulralytics/yolov5','custom',path='모델')
sys.path.append(yolov5_path)
import cv2
import argparse
from utils.general import non_max_suppression
from models.common import DetectMultiBackend
from pathlib import Path
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import torch


from utils.torch_utils import select_device, smart_inference_mode


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="C:/Users/Min/Desktop/best.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default="0", help="0 for webcam, or video path, or stream URL")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size (height, width)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45,help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def main(opt):
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device)
    print("비디오 소스를 열고 있습니다:", opt.source)
    cap = cv2.VideoCapture(int(opt.source))
    if not cap.isOpened():
        print("비디오 캡처를 열 수 없습니다:", opt.source)
        return

    print("비디오 캡처가 성공적으로 열렸습니다.")
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        #print(ret)
        if not ret:
            break
        img = cv2.resize(frame, (opt.imgsz[1], opt.imgsz[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = 640  # 모델에 따라 다를 수 있음
        frame_resized = cv2.resize(frame, (img_size, img_size))
    
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
        pred = model(img)

        # 결과 처리
        pred = pred[pred[:, :, 4] > 0.25]  # confidence threshold
        current_count = defaultdict(int)

        for *xyxy, conf, cls in pred:
            class_id = int(cls)
            class_name = class_names[class_id]
            current_count[class_name] += 1
            
        if True: # opt.view_img:
            for result in results:
                for *xyxy, conf, cls in result:
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    cv2.putText(frame, str(cls), (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    print("Image shape:", frame.shape)
            
            cv2.imshow("YOLOv5 Detection", frame)
            cv2.waitKey(1)  # 이미지가 표시될 때까지 대기  
            print("Displaying frame")  # Debugging statemen 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)