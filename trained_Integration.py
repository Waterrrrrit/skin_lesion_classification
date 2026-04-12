import os
import torch
import logging
import kagglehub
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from classifier_head import model
from focal_Loss import FocalLoss
from dataset import AdvancedLesionDataset
from kaggle_to_13_mapper import KaggleTo13Mapper

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("SkinLesion")

LOCAL_ROOT = Path(r"C:\Users\dlrkd\Desktop\face_dis_synthetic")
TARGET_STATS = {
    'l_mean': 150.0, 'l_std': 20.0, 
    'a_mean': 128.0, 'a_std': 10.0, 
    'b_mean': 128.0, 'b_std': 10.0
}

def main():
    print("Initialize training process...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")
    
    if device.type == 'cpu':
        print("Warning: CUDA not available. Falling back to CPU.")

    valid_datasets = []

    if LOCAL_ROOT.exists():
        ds = AdvancedLesionDataset(LOCAL_ROOT, "Training", TARGET_STATS)
        if len(ds) > 0:
            valid_datasets.append(ds)
            print(f"Local dataset loaded: {len(ds)} samples")
    
    try:
        path = Path(kagglehub.dataset_download("nayanchaure/acne-dataset"))
        k_ds = KaggleTo13Mapper(path, TARGET_STATS)
        if len(k_ds) > 0:
            valid_datasets.append(k_ds)
            print(f"Kaggle dataset loaded: {len(k_ds)} samples")
    except Exception as e:
        print(f"Kaggle dataset skipped. Reason: {e}")

    if not valid_datasets:
        raise RuntimeError("No valid datasets found. Check directory paths.")

    transform = transforms.Compose([
        transforms.Resize((300, 300)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    for ds in valid_datasets:
        ds.transform = transform
    
    loader = DataLoader(
        ConcatDataset(valid_datasets), 
        batch_size=32, 
        shuffle=True, 
        pin_memory=True
    )
    
    model_gpu = model.to(device)
    criterion = FocalLoss().to(device)
    optimizer = torch.optim.AdamW(model_gpu.parameters(), lr=1e-4)

    print("Training started...")
    model_gpu.train()
    
    epochs = 1
    total_steps = len(loader)
    
    for epoch in range(epochs):
        for step, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model_gpu(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 학습 진행률 계산 및 출력 로직
            progress_pct = (step / total_steps) * 100
            print(f"Epoch [{epoch}/{epochs}] Step [{step}/{total_steps}] ({progress_pct:.1f}%) Loss: {loss.item():.4f}", end="\r")

    torch.save(model_gpu.state_dict(), "skin_final_gpu.pth")
    print("\nTraining complete. Model weights saved.")

if __name__ == "__main__":
    main()