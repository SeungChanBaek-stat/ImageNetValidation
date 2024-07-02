import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.models import alexnet
from torchvision import models
import time
from tqdm import tqdm
from torch.cuda.amp import autocast
import pandas as pd
import numpy as np
import os



def enable_dropout(model):
    """ Enable dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def check_dropout(model):
    """ Check if dropout layers are enabled """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            print(m)
            print(m.training)
            if not m.training:
                return False
    return True


if __name__ == "__main__":
    # Seed for reproducibility
    torch.manual_seed(77)

    # Seed 고정됬는지 확인
    print(torch.initial_seed())


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         normalize,
         ])

    batch_size = 128
    num_workers = 2
    T = 25  # Number of MC dropout iterations

    test_set = torchvision.datasets.ImageNet(root="c:\\Users\\AAA\\ImageNetValidation\\data", transform=transform, split='val')
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device)
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    # model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
    # model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    # model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1).to(device)
    # model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1).to(device)
    # model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).to(device) # 이건 쓸 수 없음
    # model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    print(model)
    enable_dropout(model)
    check_dropout(model)
    
    # # print model name
    # print(model.__class__.__name__)
    model_name = f'{model.__class__.__name__}16'
    print(model_name)
    
    # 모델 스크립팅
    scripted_model = torch.jit.script(model)


    print(f"Performing MCdropout with {T} iterations...")
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    start_time = time.time()

    results = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            labels = labels.to(device)

            all_top1_preds = []
            all_top5_preds = []

            for t in range(T):
                with autocast():
                    outputs = scripted_model(images)
                top1_preds = outputs.argmax(dim=1)
                top5_preds = outputs.topk(5, dim=1).indices
                all_top1_preds.append(top1_preds.cpu().numpy())
                all_top5_preds.append(top5_preds.cpu().numpy())

            all_top1_preds = np.array(all_top1_preds)  # Shape: (T, batch_size)
            all_top5_preds = np.array(all_top5_preds)  # Shape: (T, batch_size, 5)

            batch_start_idx = batch_idx * batch_size
            for i in range(images.size(0)):
                img_idx = batch_start_idx + i
                img_path, _ = test_set.imgs[img_idx]
                result = {
                    'filename': os.path.basename(img_path),
                    'true_label': labels[i].item()
                }
                for t in range(T):
                    result[f'top1_preds_dropout{t+1}'] = all_top1_preds[t, i].item()
                    result[f'top5_preds_dropout{t+1}'] = all_top5_preds[t, i].tolist()
                results.append(result)

            if (batch_idx + 1) % 40 == 0:
                elapsed_time = time.time() - start_time
                print(f"Step : {batch_idx + 1} / {len(test_loader)}")
                print(f"Batch {batch_idx + 1}: Time taken for last 40 batches: {elapsed_time:.2f} seconds")
                start_time = time.time()
    print(f"MCdropout with {T} iterations complete")

    print("csv 파일 저장중...")
    start_time = time.time()
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(f'mc_dropout{T}_{model_name}_results.csv', index=False)
    end_time = time.time()
    print(f"CSV 파일 저장 완료. 소요시간: {end_time - start_time:.2f} seconds")