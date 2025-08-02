import os
import random
import shutil
from tqdm import tqdm # 진행 상황을 시각적으로 보여주는 라이브러리

# --- 설정 (이 부분을 자신의 환경에 맞게 수정하세요) ---

# 원본 이미지와 레이블이 모두 들어있는 폴더 경로
# (예: '01_data/01_raw')
SOURCE_DIR = 'path/to/your/source_data_folder'

# 분할된 데이터셋을 저장할 최상위 폴더 경로
# (예: '01_data')
OUTPUT_DIR = 'path/to/your/output_folder'

# 분할 비율 (전체 합이 1.0이 되어야 함)
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

# 재현성을 위한 랜덤 시드 고정
RANDOM_SEED = 42

# ----------------------------------------------------

def split_dataset():
    """
    주어진 소스 폴더의 이미지와 레이블을 훈련, 검증, 시험 세트로 분할합니다.
    이미지와 이름이 같은 .xml, .txt 레이블 파일을 함께 복사합니다.
    """
    
    # 1. 소스 폴더에서 모든 이미지 파일 목록 가져오기
    try:
        image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"오류: '{SOURCE_DIR}' 폴더에 이미지 파일이 없습니다.")
            return
    except FileNotFoundError:
        print(f"오류: '{SOURCE_DIR}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # 2. 재현성을 위해 파일 목록을 무작위로 섞기
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)
    
    # 3. 비율에 따라 파일 목록 분할
    total_files = len(image_files)
    train_end = int(total_files * TRAIN_RATIO)
    valid_end = int(total_files * (TRAIN_RATIO + VALID_RATIO))
    
    train_files = image_files[:train_end]
    valid_files = image_files[train_end:valid_end]
    test_files = image_files[valid_end:]
    
    # 4. 분할된 파일들을 저장할 폴더 구조 생성
    sets = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for set_name in sets.keys():
        os.makedirs(os.path.join(OUTPUT_DIR, set_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, set_name, 'labels'), exist_ok=True)

    # 5. 파일 복사
    print("데이터셋 분할을 시작합니다...")
    for set_name, file_list in sets.items():
        print(f"\n'{set_name}' 세트 복사 중...")
        for filename in tqdm(file_list, desc=f"Processing {set_name}"):
            basename = os.path.splitext(filename)[0]
            
            # 이미지 파일 복사
            shutil.copy(os.path.join(SOURCE_DIR, filename), 
                        os.path.join(OUTPUT_DIR, set_name, 'images', filename))
            
            # 해당 이미지에 맞는 레이블 파일(.xml, .txt) 복사
            for label_ext in ['.xml', '.txt']:
                label_filename = basename + label_ext
                source_label_path = os.path.join(SOURCE_DIR, label_filename)
                if os.path.exists(source_label_path):
                    shutil.copy(source_label_path, 
                                os.path.join(OUTPUT_DIR, set_name, 'labels', label_filename))

    print("\n" + "="*30)
    print("데이터셋 분할이 성공적으로 완료되었습니다!")
    print(f"총 {total_files}개의 이미지 파일이 분할되었습니다.")
    print(f"  - 훈련 세트 (Train): {len(train_files)}개")
    print(f"  - 검증 세트 (Valid): {len(valid_files)}개")
    print(f"  - 시험 세트 (Test):  {len(test_files)}개")
    print(f"결과는 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")
    print("="*30)


if __name__ == '__main__':
    # tqdm 라이브러리가 없다면 설치해야 합니다.
    # 터미널에서: pip install tqdm
    split_dataset()
