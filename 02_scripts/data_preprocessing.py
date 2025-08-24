import torch
from torch.utils.data import Dataset
import cv2
import os
import xml.etree.ElementTree as ET
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- [공통 설정 영역] ---
# 모든 팀원이 동일하게 사용하는 설정입니다.
# 클래스 정의 (사과는 0번 클래스)
CLASSES = ['apple']

# 데이터 증강 파이프라인 정의
# Albumentations 라이브러리를 사용하면 이미지와 바운딩 박스를 함께 변환해줘서 편리합니다.
def get_transforms(is_train=True):
    if is_train:
        # 훈련용 데이터 증강
        transform = A.Compose([
            A.HorizontalFlip(p=0.5), # 50% 확률로 좌우 반전
            A.Rotate(limit=15, p=0.5), # -15도 ~ +15도 사이에서 50% 확률로 회전
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8), # 80% 확률로 광학적 변환
            A.Resize(640, 640), # 이미지 크기를 640x640으로 통일
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 정규화
            ToTensorV2() # PyTorch 텐서로 변환
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        # 검증 및 시험용 (데이터 증강 최소화)
        transform = A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return transform

# --- [공통 클래스 영역] ---
class AppleDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.transforms = get_transforms(is_train)
        
        self.image_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        
        # 이미지 파일 목록을 정렬하여 순서 보장
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. 이미지 불러오기
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB 변환

        # 2. 레이블 불러오기
        label_name = os.path.splitext(img_name)[0] + '.xml'
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        labels = []

        if os.path.exists(label_path):
            tree = ET.parse(label_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in CLASSES:
                    class_id = CLASSES.index(class_name)
                    
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)

        # --- [개별 코드 영역 시작] ---
        # Faster R-CNN 담당자: 아래 target을 그대로 사용하면 됩니다.
        # YOLO 담당자: 이 target을 모델에 맞게 변환하거나, 
        #             이전에 변환해둔 .txt 파일을 직접 읽어오는 코드를 이 부분에 추가할 수 있습니다.
        #             하지만 대부분의 YOLO 프레임워크는 데이터 경로만 지정해주면 되므로,
        #             이 클래스 전체가 필요 없을 수도 있습니다. 그 경우 이 파일은
        #             Faster R-CNN 담당자와 공통 코드의 예시로 참고하면 됩니다.
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        # --- [개별 코드 영역 끝] ---

        # 3. 데이터 증강 적용
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=target['boxes'], class_labels=target['labels'])
            image = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            # albumentations가 box가 없는 경우 빈 텐서를 올바르게 처리하지 못할 수 있으므로, 아래 코드 추가
            if len(target['boxes']) == 0:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        
        return image, target


if __name__ == '__main__':
    # 이 스크립트를 직접 실행하면, 잘 작동하는지 테스트해볼 수 있습니다.
    
    # Albumentations 라이브러리가 없다면 설치해야 합니다.
    # 터미널에서: pip install albumentations
    
    # 테스트를 위한 데이터셋 경로 (예시)
    # train, valid, test 폴더가 있는 상위 폴더를 지정
    DATA_PATH = '01_data' 
    
    # 훈련 데이터셋 로더 테스트
    train_dataset = AppleDataset(data_dir=os.path.join(DATA_PATH, 'train'), is_train=True)
    
    if len(train_dataset) > 0:
        # 첫 번째 데이터 샘플 가져오기
        image, target = train_dataset[0]
        
        print("--- 데이터셋 테스트 ---")
        print(f"이미지 텐서 모양: {image.shape}")
        print(f"타겟(레이블) 정보: {target}")
        print("테스트 성공! 이 클래스를 학습 코드에서 사용하세요.")
    else:
        print("오류: 훈련 데이터셋을 찾을 수 없거나 비어있습니다. 경로를 확인해주세요.")

