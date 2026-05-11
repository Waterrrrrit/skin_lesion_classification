import random
import numpy as np
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from model import build_efficientnet_b3
from loss import FocalLoss
from preprocess import ReinhardNormalizer
from lesion_dataset import LesionClassificationDataset, LESION_CLASSES_13

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Multi-GPU 사용 시
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_transforms(input_size: int):
    reinhard = ReinhardNormalizer()
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        reinhard,
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transform = transforms.Compose([
        reinhard,
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    return train_transform, val_transform

def safe_collate_fn(batch):
    """Dataset에서 반환된 None(에러 발생 데이터)을 제거하고 배치를 구성합니다."""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0) # 배치가 모두 에러면 빈 텐서 반환
    return default_collate(batch)

def main():
    set_seed(42)
    default_dataset_root = Path(__file__).resolve().parent / "dataset"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, 
                        default=Path(os.environ.get("SM_CHANNEL_TRAINING", "./dataset")))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=Path, 
                        default=Path(os.environ.get("SM_MODEL_DIR", "./outputs")))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    class_names_path = args.save_dir / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(LESION_CLASSES_13, f, ensure_ascii=False, indent=4)
    print(f"클래스 정보 저장 완료: {class_names_path}")

    model = build_efficientnet_b3(num_classes=len(LESION_CLASSES_13), device=device)
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = Adam(model.parameters(), lr=args.lr)

    train_transform, val_transform = build_transforms(input_size=300)

    train_img_dir = args.dataset_root / "Training" / "images"
    train_json_dir = args.dataset_root / "Training" / "jsons"
    val_img_dir = args.dataset_root / "Validation" / "images"
    val_json_dir = args.dataset_root / "Validation" / "jsons"

    if not train_img_dir.exists():
        print(f"[오류] 데이터셋 경로를 찾을 수 없습니다: {train_img_dir}")
        return

    train_dataset = LesionClassificationDataset(image_dir=str(train_img_dir), json_dir=str(train_json_dir), transform=train_transform)
    val_dataset = LesionClassificationDataset(image_dir=str(val_img_dir), json_dir=str(val_json_dir), transform=val_transform)

    # collate_fn=safe_collate_fn 적용
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=safe_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=safe_collate_fn)

    best_f1 = 0.0
    metrics_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        for images, labels in train_loader:
            if len(images) == 0: # 스킵되어 빈 배치인 경우 패스
                continue
                
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
        # 수정됨: 중복된 계산 및 변수 삭제로 올바른 Loss 산출
        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                if len(images) == 0:
                    continue
                    
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                except Exception as e:
                    print(f"\n[Warning] Validation 배치 처리 중 에러 발생 스킵: {e}")
                    continue

        if len(all_labels) > 0:
            val_acc = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
            
            print(f"Epoch [{epoch}/{args.epochs}] - Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {f1:.4f}")

            metrics_history.append({
                "epoch": epoch, "train_loss": float(epoch_loss), "val_accuracy": float(val_acc),
                "precision_macro": float(precision), "recall_macro": float(recall), "f1_macro": float(f1)
            })

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), args.save_dir / "best_model.pth")
                
                # 수정됨: 들여쓰기 교정 및 누락되었던 Confusion Matrix 복구
                print(f"  [!] 최고 성능 모델 갱신 (F1: {best_f1:.4f})")
                
                cm = confusion_matrix(all_labels, all_preds, labels=range(len(LESION_CLASSES_13)))
                with open(args.save_dir / "confusion_matrix.csv", "w", encoding="utf-8") as f:
                    f.write("," + ",".join(LESION_CLASSES_13) + "\n")
                    for i, row in enumerate(cm):
                        f.write(f"{LESION_CLASSES_13[i]}," + ",".join(map(str, row)) + "\n")
        else:
            print(f"Epoch [{epoch}/{args.epochs}] - 검증할 수 있는 유효한 데이터가 없습니다.")

    with open(args.save_dir / "metrics_history.json", "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=4, ensure_ascii=False)
    print("학습이 완료되었습니다.")

if __name__ == "__main__":
    main()
