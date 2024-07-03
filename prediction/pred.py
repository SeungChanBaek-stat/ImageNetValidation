import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

def main():
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

    test_set = torchvision.datasets.ImageNet(root="c:\\Users\\AAA\\ImageNetValidation\\data", transform=transform, split='val')
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2)

    # model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)
    # model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1).to(device)
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1).to(device)
    # model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    # model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
    # model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1).to(device)
    # model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1).to(device)
    # model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).to(device)
    # model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    model_name = f'{model.__class__.__name__}1_1'
    print(model_name)
    print(model)
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    all_labels = []
    all_preds = []
    all_filenames = []

    start_time = time.time()
    total_time = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader, leave=False)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (pred == labels).sum().item()

            _, rank5 = outputs.topk(5, 1, True, True)
            rank5 = rank5.t()
            correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))

            for k in range(6):
                correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)

            correct_top5 += correct_k.item()

            all_labels.extend(labels)
            all_preds.extend(pred)
            all_filenames.extend([os.path.basename(test_set.imgs[idx * batch_size + i][0]) for i in range(images.size(0))])

            if idx % 40 == 0:
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                print(f"Step: {idx + 1} / {len(test_loader)}")
                print(f"Top-1 accuracy: {correct_top1 / total * 100:.2f}%")
                print(f"Top-5 accuracy: {correct_top5 / total * 100:.2f}%")
                print(f"Time taken for last 40 batches: {elapsed_time:.2f} seconds")
                start_time = time.time()

    print(f"Final Top-1 accuracy: {correct_top1 / total * 100:.2f}%")
    print(f"Final Top-5 accuracy: {correct_top5 / total * 100:.2f}%")
    print(f"Total time taken: {total_time:.2f} seconds")

    # Convert all collected data to CPU at once
    all_labels = torch.stack(all_labels).cpu().numpy()
    all_preds = torch.stack(all_preds).cpu().numpy()

    results = [{'filename': fn, 'true_label': tl, 'pred_label': pl} for fn, tl, pl in zip(all_filenames, all_labels, all_preds)]

    print("Saving results to CSV...")
    start_time = time.time()
    df = pd.DataFrame(results)
    df.to_csv(f'{model_name}_results.csv', index=False)
    end_time = time.time()
    print(f"CSV file saved. Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()