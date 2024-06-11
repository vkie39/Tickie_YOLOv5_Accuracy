import argparse
import sys
yolov5_path = "G:/내 드라이브/yolov5"
sys.path.append(yolov5_path)
import sys
import cv2
import argparse
from utils.general import non_max_suppression
from models.common import DetectMultiBackend
from pathlib import Path
import pathlib
pathlib.PosixPath = pathlib.WindowsPath


from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
    weights= "C:/Users/Min/Desktop/best.pt" ,#학습시킨 모델의 경로
    source=0, #현재는 웹캠 사용
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    vid_stride=1,  # video frame-rate stride
    dnn=False,
    data="G:/내 드라이브/project/Tickie_YOLOv5_Accuracy/code/Min/lay_not_lay.v1i.yolov5pytorch/data.yaml",  # dataset.yaml path
    half=False,
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    name="exp",  # save results to project/name
):
    source = str(source)
    # YOLOv5 모델 및 관련 설정
    
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    
    # 웹캠으로부터 프레임을 읽어오기
    cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 나타냄

    while True:
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            break

        # YOLOv5 모델을 사용하여 프레임에서 객체 감지root

        pred = model( conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, augment=augment, visualize=visualize)


        # Non-maximum suppression 적용
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # 결과를 프레임에 표시
        if pred[0] is not None:
            for det in pred[0]:
                tl, br, c, conf = det[:4].astype(int), det[4].astype(int), int(det[5]), det[6]
                cv2.rectangle(frame, tuple(tl), tuple(br), (0, 255, 0), 2)
                label = f"{model.names[c]} {conf:.2f}"
                cv2.putText(frame, label, (tl[0], tl[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 프레임 표시
        cv2.imshow("YOLOv5 Real-time Object Detection", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 작업 완료 후 해제
    cap.release()
    cv2.destroyAllWindows()

def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

if __name__ == "__main__":
    run()
