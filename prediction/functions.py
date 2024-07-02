import sys, os
sys.path.append(os.pardir)
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from torchviz import make_dot
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import time


# 전처리 클래스
class BaseTransform() :
    def __init__(self , resize, mean, std) :
        self.base_transform = transforms.Compose([
            transforms.Resize(resize), # 짧은 변의 길이 기준으로 resize
            transforms.CenterCrop(resize), # 화상 중앙을 resize * resize 로 자름
            transforms.ToTensor(), # 토치 텐서로 변환
            transforms.Normalize(mean, std) # 색상 정보 표준화
        ])
        
    def __call__(self, img) :
        return self.base_transform(img)
    

# Step 5: Define the dataset class
class ImageNetDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
    

# Step 6: Define the prediction function
def pred_model(model, dataset, batch_size, num_workers, device):
    y_pred_all = torch.tensor([], dtype=torch.float32, device=device)

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast():
                outputs = model(inputs)
                y_pred_all = torch.cat((y_pred_all, outputs), dim=0)

            if (batch_idx + 1) % 20 == 0:
                elapsed_time = time.time() - start_time
                print(f'Batch {batch_idx + 1}: Time taken for last 20 batches: {elapsed_time:.2f} seconds')
                start_time = time.time()  # Reset the start time for the next set of 20 batches

    y_pred_all_cpu = y_pred_all.cpu().numpy()
    y_pred_all_list = y_pred_all_cpu.tolist()

    return y_pred_all_list