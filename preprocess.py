import numpy as np
import cv2
from PIL import Image

class ReinhardNormalizer:
    def __init__(self, target_l=150, target_a=128, target_b=128, target_std_l=40, target_std_a=5, target_std_b=5):
        self.target_means = np.array([target_l, target_a, target_b])
        # 타겟 이미지의 평균적인 표준편차를 설정 (의료/피부 데이터에 맞게 조정 필요)
        self.target_stds = np.array([target_std_l, target_std_a, target_std_b])

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img.convert('RGB'))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        for i in range(3):
            means = np.mean(lab[:, :, i])
            stds = np.std(lab[:, :, i]) + 1e-6 # 0으로 나누기 방지
            
            # 평균과 표준편차 모두 보정
            lab[:, :, i] = ((lab[:, :, i] - means) * (self.target_stds[i] / stds)) + self.target_means[i]
            
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)
