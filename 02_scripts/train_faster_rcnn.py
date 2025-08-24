import torch
from torch.utils.data import DataLoader
import yaml
import pandas as pd
import os
import time
from tqdm import tqdm
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import sys
import argparse #<-- 인자값 처리를 위한 라이브러리 추가

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from apple_dataset import AppleDataset 

def load_config(config_path='data.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training")
    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
        progress_bar.set_postfix(loss=losses.item())
    return total_loss / len(data_loader)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    preds, targets_for_metric = [], []
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = list(image.to(device) for image in images)
        outputs = model(images)
        for i in range(len(images)):
            preds.append({'boxes': outputs[i]['boxes'].cpu(), 'scores': outputs[i]['scores'].cpu(), 'labels': outputs[i]['labels'].cpu()})
            targets_for_metric.append({'boxes': targets[i]['boxes'].cpu(), 'labels': targets[i]['labels'].cpu()})
    return preds, targets_for_metric

def train_model(config, run_number): # <-- run_number 인자 추가
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Run #{run_number}: Faster R-CNN 모델 학습 시작 ---")
    print(f"Using device: {device}")
    
    num_classes = config['nc'] + 1
    
    train_data_dir = os.path.join(config['path'], 'train')
    valid_data_dir = os.path.join(config['path'], 'valid')
    
    dataset = AppleDataset(data_dir=train_data_dir, is_train=True)
    dataset_valid = AppleDataset(data_dir=valid_data_dir, is_train=False)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    data_loader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    model = get_model(num_classes)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config['learning_rate'], momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_map = -1.0
    patience_counter = 0
    best_model_path_during_training = ""

    # --- [수정된 부분] ---
    # 실행 횟수에 따라 모델 저장 폴더를 다르게 설정
    model_save_dir = f'03_models/faster_rcnn/run_{run_number}'
    os.makedirs(model_save_dir, exist_ok=True)
    # --- [수정 끝] ---
    
    for epoch in range(config['epochs']):
        avg_loss = train_one_epoch(model, optimizer, data_loader, device)
        preds, targets = evaluate(model, data_loader_valid, device)
        metric = MeanAveragePrecision()
        metric.update(preds, targets)
        map_result = metric.compute()
        current_map = map_result['map_50'].item()
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}, Validation mAP@.50: {current_map:.4f}")
        lr_scheduler.step()
        
        if current_map > best_map:
            best_map = current_map
            if best_model_path_during_training and os.path.exists(best_model_path_during_training):
                os.remove(best_model_path_during_training)
            best_model_path_during_training = os.path.join(model_save_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path_during_training)
            print(f"Best model saved to {best_model_path_during_training} with mAP: {best_map:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    final_best_model_path = os.path.join(model_save_dir, "best_model.pth")
    if best_model_path_during_training:
        if os.path.exists(final_best_model_path):
             os.remove(final_best_model_path)
        os.rename(best_model_path_during_training, final_best_model_path)
        print(f"Final best model is saved as {final_best_model_path}")
    else:
        print("Warning: No best model was saved during training.")
        return None
    return final_best_model_path

def evaluate_model(model_path, config):
    # (이 함수 내용은 수정할 필요 없음)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = config['nc'] + 1
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    test_data_dir = os.path.join(config['path'], 'test')
    dataset_test = AppleDataset(data_dir=test_data_dir, is_train=False)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print("\n--- 최종 모델 성능 평가 시작 (Test Set) ---")
    preds, targets = evaluate(model, data_loader_test, device)
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(preds, targets)
    map_result = metric.compute()
    map50, precision, recall = map_result['map_50'].item(), map_result['map'].item(), map_result['mar_100'].item()
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    dummy_input = torch.randn(1, 3, config['imgsz'], config['imgsz']).to(device)
    for _ in range(10): _ = model(dummy_input)
    start_time = time.time()
    for _ in range(50): _ = model(dummy_input)
    end_time = time.time()
    fps = 50 / (end_time - start_time)
    print(f"평가 완료! mAP@.50: {map50:.4f}, FPS: {fps:.2f}")
    return {'Model': 'Faster R-CNN', 'Precision': precision, 'Recall': recall, 'F1-Score': f1_score, 'mAP': map50, 'FPS': fps}

def save_results(results, run_number): # <-- run_number 인자 추가
    # --- [수정된 부분] ---
    # 실행 횟수에 따라 CSV 파일 이름을 다르게 설정
    csv_path = f'04_results/performance_summary_run_{run_number}.csv'
    # --- [수정 끝] ---
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv(csv_path, index=False)
    print(f"\n결과가 '{csv_path}'에 성공적으로 저장되었습니다.")

if __name__ == '__main__':
    # --- [수정된 부분] ---
    # 터미널에서 실행 번호를 입력받도록 설정
    parser = argparse.ArgumentParser(description='Train Faster R-CNN model.')
    parser.add_argument('--run', type=int, required=True, help='The run number for this experiment (e.g., 1, 2, 3).')
    args = parser.parse_args()
    # --- [수정 끝] ---
    
    config = load_config()
    best_model_path = train_model(config, args.run) # <-- run_number 전달
    
    if best_model_path:
        performance_metrics = evaluate_model(best_model_path, config)
        save_results(performance_metrics, args.run) # <-- run_number 전달
