import torch
from torch.utils.data import Dataset
import cv2
import os
import xml.etree.ElementTree as ET
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- [공통 설정 및 클래스 영역] ---
# 이 스크립트는 팀 전체의 데이터 로딩 및 증강 표준을 정의합니다.

# 클래스 이름 정의 (XML 파일 내부의 클래스 이름과 정확히 일치해야 함)
CLASSES = ['Apple']

# 데이터 증강 파이프라인 정의
def get_transforms(is_train=True):
    if is_train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        transform = A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    return transform

class AppleDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.transforms = get_transforms(is_train)
        self.image_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
                    class_id = CLASSES.index(class_name) + 1
                    bndbox = obj.find('bndbox')
                    xmin, ymin, xmax, ymax = [float(bndbox.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id)

        # --- [오류 수정 최종 버전] ---
        # 데이터 증강 적용
        transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
        
        image = transformed['image']
        transformed_boxes = transformed['bboxes']
        transformed_labels = transformed['class_labels']

        # 최종 target 딕셔너리 생성
        # 객체가 있든 없든, 항상 동일한 형식과 타입의 텐서를 반환하도록 보장
        target = {}
        target['boxes'] = torch.as_tensor(transformed_boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(transformed_labels, dtype=torch.int64)

        # 만약 증강 후 박스가 모두 사라졌다면, 빈 텐서의 모양을 (0, 4)로 맞춰줌
        if target['boxes'].shape[0] == 0:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)

        return image, target
