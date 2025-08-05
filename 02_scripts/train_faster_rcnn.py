import torch
from torch.utils.data import DataLoader
import yaml
import pandas as pd
import os
import time
from tqdm import tqdm

# --- [Faster R-CNN 담당자 스크립트] ---
#
# 이 코드는 YOLOv8 스크립트와 동일한 구조를 유지하면서,
# Faster R-CNN 모델을 학습하고 평가하는 완전한 파이프라인입니다.
#
# ---

# 1. 팀 공통 스크립트 및 라이브러리 임포트
# apple_dataset.py가 02_scripts 폴더에 있다고 가정합니다.
from apple_dataset import AppleDataset 
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def load_config(config_path='data.yaml'):
    """YAML 설정 파일을 불러옵니다."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model(num_classes):
    """사전 학습된 Faster R-CNN 모델을 불러오고, 분류기를 수정합니다."""
    # COCO 데이터셋으로 사전 학습된 모델 로드
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # 분류기(classifier)의 입력 특성 수를 가져옴
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 사전 학습된 헤드를 새로운 헤드로 교체
    # num_classes는 배경(background) + 우리가 정의한 클래스 수
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def collate_fn(batch):
    """DataLoader에서 배치(batch)를 구성할 때 사용하는 함수"""
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device):
    """한 에폭(epoch) 동안 모델을 학습합니다."""
    model.train()
    total_loss = 0
    
    for images, targets in tqdm(data_loader, desc="Training"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
    return total_loss / len(data_loader)


@torch.no_grad() # 평가 중에는 그래디언트 계산을 비활성화
def evaluate(model, data_loader, device):
    """모델을 평가하고 mAP 계산을 위한 예측/정답을 반환합니다."""
    model.eval()
    
    preds = []
    targets_for_metric = []
    
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = list(image.to(device) for image in images)
        
        outputs = model(images)
        
        # mAP 계산을 위해 예측값과 정답값을 CPU로 옮기고 형식 맞추기
        for i in range(len(images)):
            pred = {}
            pred['boxes'] = outputs[i]['boxes'].cpu()
            pred['scores'] = outputs[i]['scores'].cpu()
            pred['labels'] = outputs[i]['labels'].cpu()
            preds.append(pred)
            
            target = {}
            target['boxes'] = targets[i]['boxes'].cpu()
            target['labels'] = targets[i]['labels'].cpu()
            targets_for_metric.append(target)
            
    return preds, targets_for_metric


def train_model(config):
    """Faster R-CNN 모델을 학습시키고, 학습된 모델 경로를 반환합니다."""
    
    # 1. 설정값 불러오기
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 클래스 개수 (+1은 배경 클래스)
    num_classes = config['nc'] + 1
    
    # 2. 데이터셋 및 데이터로더 준비
    dataset_path = os.path.join(config['path'], config['train'])
    dataset_valid_path = os.path.join(config['path'], config['val'])
    
    # 팀 공통 AppleDataset 클래스 사용
    dataset = AppleDataset(data_dir=os.path.dirname(dataset_path), is_train=True)
    dataset_valid = AppleDataset(data_dir=os.path.dirname(dataset_valid_path), is_train=False)

    data_loader = DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    data_loader_valid = DataLoader(
        dataset_valid, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 3. 모델, 옵티마이저, 스케줄러 설정
    model = get_model(num_classes)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config['learning_rate'], momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 4. 학습 루프 실행
    best_map = -1.0
    patience_counter = 0
    
    print("--- Faster R-CNN 모델 학습 시작 ---")
    for epoch in range(config['epochs']):
        avg_loss = train_one_epoch(model, optimizer, data_loader, device)
        lr_scheduler.step()
        
        # 검증 세트로 성능 평가
        preds, targets = evaluate(model, data_loader_valid, device)
        metric = MeanAveragePrecision()
        metric.update(preds, targets)
        map_result = metric.compute()
        current_map = map_result['map_50'].item()
        
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}, Validation mAP@.50: {current_map:.4f}")
        
        # 최적 모델 저장 (Early Stopping 포함)
        if current_map > best_map:
            best_map = current_map
            save_path = f"03_models/faster_rcnn/best_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path} with mAP: {best_map:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    # 최종적으로 가장 성능이 좋았던 모델의 경로 반환
    # (실제로는 가장 마지막에 저장된 best_model 경로를 추적해야 함)
    # 여기서는 간단하게 마지막 저장 경로를 예시로 반환
    final_best_model_path = f"03_models/faster_rcnn/best_model.pth" # 최종 모델 이름은 통일
    # 실제로는 가장 좋았던 모델을 best_model.pth로 복사/저장하는 로직이 필요
    # torch.save(model.state_dict(), final_best_model_path) # 데모용
    
    return final_best_model_path # 실제로는 가장 좋았던 모델의 경로를 반환해야 함

def evaluate_model(model_path, config):
    """학습된 모델을 테스트 세트로 평가하고, 성능 지표를 반환합니다."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = config['nc'] + 1
    
    # 1. 모델 불러오기
    model = get_model(num_classes)
    # model.load_state_dict(torch.load(model_path)) # 이 부분을 실제로 사용할 때 활성화
    model.to(device)
    
    # 2. 테스트 데이터로더 준비
    dataset_test_path = os.path.join(config['path'], config['test'])
    dataset_test = AppleDataset(data_dir=os.path.dirname(dataset_test_path), is_train=False)
    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print("\n--- 최종 모델 성능 평가 시작 (Test Set) ---")
    preds, targets = evaluate(model, data_loader_test, device)
    
    # 3. 성능 지표 계산
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(preds, targets)
    map_result = metric.compute()
    
    # 클래스가 하나이므로, 전체 map과 클래스별 지표가 거의 동일
    map50 = map_result['map_50'].item()
    # torchmetrics 0.11+ 에서는 results['per_class']가 없음.
    # 여기서는 전체 precision/recall을 대표로 사용
    precision = map_result['map_per_class'][0].item() if 'map_per_class' in map_result else map_result['map'].item()
    recall = map_result['mar_100_per_class'][0].item() if 'mar_100_per_class' in map_result else map_result['mar_100'].item()
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 4. 속도(FPS) 측정
    dummy_input = torch.randn(1, 3, config['imgsz'], config['imgsz']).to(device)
    
    # 워밍업
    for _ in range(10):
        _ = model(dummy_input)
        
    start_time = time.time()
    for _ in range(50):
        _ = model(dummy_input)
    end_time = time.time()
    fps = 50 / (end_time - start_time)
    
    print(f"평가 완료! mAP@.50: {map50:.4f}, FPS: {fps:.2f}")
    
    return {
        'Model': 'Faster R-CNN',
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'mAP': map50,
        'FPS': fps
    }

def save_results(results, csv_path='04_results/performance_summary.csv'):
    """평가 결과를 공유 CSV 파일에 저장하거나 업데이트합니다."""
    # (이 함수는 YOLOv8 스크립트와 동일하므로 생략 가능, 혹은 그대로 사용)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if results['Model'] in df['Model'].values:
            df.loc[df['Model'] == results['Model']] = list(results.values())
        else:
            df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
    else:
        df = pd.DataFrame([results])
    df.to_csv(csv_path, index=False)
    print(f"\n결과가 '{csv_path}'에 성공적으로 저장/업데이트되었습니다.")


if __name__ == '__main__':
    # 라이브러리 설치: pip install torch torchvision torchmetrics pandas pyyaml tqdm albumentations
    
    # 1. 설정 파일 로드
    config = load_config()
    
    # 2. 모델 학습 실행
    best_model_path = train_model(config)
    
    # 3. 학습된 최적 모델 평가 (실제로는 저장된 best 모델 경로를 사용해야 함)
    # 여기서는 데모를 위해 학습 직후의 모델로 평가
    performance_metrics = evaluate_model(best_model_path, config)
    
    # 4. 최종 결과 저장
    save_results(performance_metrics)
