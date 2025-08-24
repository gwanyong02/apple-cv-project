import os
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm

# --- 설정 ---
# 원본 이미지와 XML 파일이 있는 폴더 경로
DATA_DIR = '01_data/01_raw' 

# 수정 모드 설정
# 'scan': 오류가 있는 파일만 찾아서 출력합니다. (기본값)
# 'fix': 오류가 있는 파일을 찾아서 자동으로 수정하고 덮어씁니다.
MODE = 'scan' # 'scan' 또는 'fix'로 설정
# ----------------

def validate_and_fix_annotations():
    """
    데이터 폴더의 모든 XML 어노테이션을 검증하고, 필요시 수정합니다.
    """
    problematic_files = []
    
    xml_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.xml')]
    
    print(f"총 {len(xml_files)}개의 XML 파일을 검사합니다. (모드: {MODE})")

    for xml_file in tqdm(xml_files, desc="Validating annotations"):
        basename = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(DATA_DIR, xml_file)
        
        # 이미지 경로 찾기
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_img_path = os.path.join(DATA_DIR, basename + ext)
            if os.path.exists(potential_img_path):
                img_path = potential_img_path
                break
        
        if not img_path:
            continue

        # 이미지 크기 읽기
        try:
            image = cv2.imread(img_path)
            img_height, img_width = image.shape[:2]
        except Exception:
            print(f"경고: {img_path} 이미지 파일을 읽을 수 없습니다.")
            continue
            
        # XML 파싱
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        is_fixed = False
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            # 좌표값 검증
            original_coords = (xmin, ymin, xmax, ymax)
            
            # 좌표값이 이미지 경계를 벗어나는지 확인하고 수정 (Clamping)
            fixed_xmin = max(0, xmin)
            fixed_ymin = max(0, ymin)
            fixed_xmax = min(img_width, xmax)
            fixed_ymax = min(img_height, ymax)
            
            fixed_coords = (fixed_xmin, fixed_ymin, fixed_xmax, fixed_ymax)

            # 좌표가 수정되었는지 확인
            if original_coords != fixed_coords:
                if xml_file not in problematic_files:
                    problematic_files.append(xml_file)
                
                if MODE == 'fix':
                    bndbox.find('xmin').text = str(fixed_xmin)
                    bndbox.find('ymin').text = str(fixed_ymin)
                    bndbox.find('xmax').text = str(fixed_xmax)
                    bndbox.find('ymax').text = str(fixed_ymax)
                    is_fixed = True

        # 'fix' 모드이고 파일이 수정되었다면, 덮어쓰기
        if is_fixed:
            tree.write(xml_path)

    print("\n--- 검사 완료 ---")
    if not problematic_files:
        print("모든 어노테이션이 유효합니다! 🎉")
    else:
        print(f"총 {len(problematic_files)}개의 파일에서 좌표 오류가 발견되었습니다:")
        for fname in problematic_files:
            print(f" - {fname}")
        
        if MODE == 'fix':
            print("\n모든 오류 파일이 자동으로 수정되어 덮어쓰기 되었습니다.")
            print("이제 'voc_to_yolo_converter.py'를 다시 실행하여 .txt 파일을 재생성하세요.")
        else:
            print("\n오류를 수정하려면 MODE를 'fix'로 변경하고 스크립트를 다시 실행하거나,")
            print("LabelImg 같은 툴을 사용하여 위 파일들을 직접 수정해주세요.")


if __name__ == '__main__':
    validate_and_fix_annotations()
