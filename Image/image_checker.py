import os
from collections import defaultdict

def find_duplicate_images(directory):
    # 디렉토리에서 모든 파일 목록 가져오기
    files = os.listdir(directory)
    
    # 파일 확장자가 이미지인지 확인
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]
    
    # 이미지 파일들의 해시값을 저장할 딕셔너리 생성
    hash_dict = defaultdict(list)
    
    # 중복된 이미지 찾기
    for image_file in image_files:
        with open(os.path.join(directory, image_file), 'rb') as f:
            file_hash = hash(f.read())
            hash_dict[file_hash].append(image_file)
    
    # 중복된 이미지 파일들을 출력
    duplicate_images = [image_list for image_list in hash_dict.values() if len(image_list) > 1]
    if not duplicate_images:
        print("중복된 이미지가 없습니다.")
    else:
        print("중복된 이미지 파일들:")
        for image_list in duplicate_images:
            print(image_list)

# 디렉토리 경로 설정
directory_path = 'C:/Users/Min/Desktop/image/image_dep'

# 중복된 이미지 찾기 함수 호출
find_duplicate_images(directory_path)
