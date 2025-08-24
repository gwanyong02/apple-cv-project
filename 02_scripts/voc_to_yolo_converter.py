import xml.etree.ElementTree as ET
import os
import cv2
from tqdm import tqdm

# --- 설정 (이 부분을 자신의 환경에 맞게 수정하세요) ---

# 클래스 이름 정의 (사과만 있으므로 'apple' 하나만 정의)
CLASSES = ['apple']

# 원본 이미지와 XML 파일이 함께 들어있는 폴더 경로
# (예: '01_data/01_raw')
INPUT_DIR = 'path/to/your/source_data_folder' 

# 변환된 YOLO 형식(.txt) 라벨을 저장할 폴더 경로
# 이 폴더는 미리 만들어 두거나, 스크립트가 자동으로 생성합니다.
# 원본 폴더와 같은 곳에 저장해도 무방합니다.
OUTPUT_DIR = 'path/to/your/source_data_folder'

# ----------------------------------------------------

def convert_voc_to_yolo(img_width, img_height, box):
    """
    PASCAL VOC (xmin, xmax, ymin, ymax) 좌표를 YOLO 형식으로 변환합니다.
    YOLO 형식: (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
    모든 좌표 값은 0과 1 사이로 정규화됩니다.
    """
    # 박스의 중심점과 너비, 높이 계산
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    width = box[1] - box[0]
    height = box[3] - box[2]
    
    # 이미지의 전체 너비와 높이로 정규화
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    
    x_center_norm = x_center * dw
    y_center_norm = y_center * dh
    width_norm = width * dw
    height_norm = height * dh
    
    return (x_center_norm, y_center_norm, width_norm, height_norm)

def process_voc_to_yolo_conversion():
    """데이터셋 폴더를 순회하며 모든 XML을 YOLO 형식으로 변환합니다."""
    
    # 출력 폴더가 없으면 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"'{OUTPUT_DIR}' 폴더를 생성했습니다.")

    # 입력 폴더에서 .xml 파일 목록을 가져옴
    xml_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.xml')]
    
    print(f"총 {len(xml_files)}개의 XML 파일을 YOLO 형식으로 변환합니다...")
    
    for xml_file in tqdm(xml_files, desc="Converting XML to YOLO"):
        xml_path = os.path.join(INPUT_DIR, xml_file)
        file_basename = os.path.splitext(xml_file)[0]
        
        # 대응하는 이미지 파일 경로 탐색
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_img_path = os.path.join(INPUT_DIR, file_basename + ext)
            if os.path.exists(potential_img_path):
                img_path = potential_img_path
                break
        
        if img_path is None:
            print(f"경고: {xml_file}에 해당하는 이미지 파일이 없습니다. 건너뜁니다.")
            continue

        output_path = os.path.join(OUTPUT_DIR, file_basename + '.txt')
        
        yolo_labels = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            image = cv2.imread(img_path)
            if image is None:
                print(f"오류: {img_path} 이미지 파일을 읽을 수 없습니다. 건너뜁니다.")
                continue
            img_height, img_width = image.shape[:2]

            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in CLASSES:
                    continue
                class_id = CLASSES.index(class_name)
                
                bndbox = obj.find('bndbox')
                box = (float(bndbox.find('xmin').text), 
                       float(bndbox.find('xmax').text), 
                       float(bndbox.find('ymin').text), 
                       float(bndbox.find('ymax').text))
                
                yolo_box = convert_voc_to_yolo(img_width, img_height, box)
                
                yolo_labels.append(f"{class_id} {' '.join(map(str, yolo_box))}\n")
                
            with open(output_path, 'w') as f:
                f.writelines(yolo_labels)

        except Exception as e:
            print(f"오류: {xml_file} 처리 중 예외 발생: {e}")

    print("\n변환 완료!")

if __name__ == '__main__':
    process_voc_to_yolo_conversion()
