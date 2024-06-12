import cv2
import os

def extract_frames(video_path, output_folder):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 파일이 열렸는지 확인
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    # 프레임 수와 프레임 속도 가져오기
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 저장할 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 프레임을 캡쳐하고 저장
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 파일 이름 생성 (cap01.jpg, cap02.jpg, ...)
        file_name = f"cap{frame_count + 1:02d}.jpg"
        frame_path = os.path.join(output_folder, file_name)
        
        # 프레임 저장
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    # 비디오 파일 닫기
    cap.release()
    
    print(f"Frames extracted: {frame_count}/{total_frames}")

# 비디오 파일 경로 및 출력 폴더 지정
video_path = "your_video.mp4"
output_folder = "frames"

# 프레임 추출 함수 호출
extract_frames(video_path, output_folder)
