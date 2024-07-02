import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models
from torch.cuda.amp import autocast
import time
from tqdm import tqdm
import pandas as pd
import os
import cProfile
import pstats
import io
import numpy as np

class DropoutDataset(data.Dataset):
    def __init__(self, dataset, T):
        self.dataset = dataset
        self.T = T

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, self.T

class DropoutDataLoader(data.DataLoader):
    def __init__(self, dataset, T, **kwargs):
        super().__init__(dataset, **kwargs)
        self.T = T

    def __iter__(self):
        for batch in super().__iter__():
            images, labels = batch
            yield images, labels, self.T

def enable_dropout(model):
    """ Enable dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def check_dropout(model):
    """ Check if dropout layers are enabled """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            if not m.training:
                return False
    return True

def preds_torch_parallel(images, T, scripted_model):
    # Repeat images T times along a new dimension and flatten to create a new batch dimension
    expanded_images = images.repeat(T, 1, 1, 1)
    batch_size = images.size(0)
    num_classes = 1000  # Assuming ImageNet
    
    # Initialize tensors to store predictions
    all_top1_preds = torch.zeros(T, batch_size, dtype=torch.int64, device='cuda')
    all_top5_preds = torch.zeros(T, batch_size, 5, dtype=torch.int64, device='cuda')

    # Perform inference with dropout enabled, in parallel for T iterations
    with autocast():
        outputs = scripted_model(expanded_images)

    # Reshape outputs to separate the T iterations
    outputs = outputs.view(T, batch_size, -1)

    # Collect predictions for each iteration
    for t in range(T):
        all_top1_preds[t] = outputs[t].argmax(dim=1)
        all_top5_preds[t] = outputs[t].topk(5, dim=1).indices

    return all_top1_preds, all_top5_preds

def preds_torch_parallel_optimized(images, T, scripted_model):
    expanded_images = images.repeat(T, 1, 1, 1)
    batch_size = images.size(0)
    num_classes = 1000  # Assuming ImageNet

    with autocast():
        outputs = scripted_model(expanded_images)
    
    outputs = outputs.view(T, batch_size, num_classes)
    all_top1_preds = outputs.argmax(dim=2)
    all_top5_preds = outputs.topk(5, dim=2).indices

    return all_top1_preds, all_top5_preds

def batch_organize(T, labels, images, all_top1_preds, all_top5_preds, batch_start_idx, test_set):
    batch_results = [{
        'filename': os.path.basename(test_set.imgs[batch_start_idx + i][0]),
        'true_label': labels[i].item(),
        **{f'top1_preds_dropout{t+1}': all_top1_preds[t, i].item() for t in range(T)},
        **{f'top5_preds_dropout{t+1}': all_top5_preds[t, i].tolist() for t in range(T)}
    } for i in range(images.size(0))]
    return batch_results


def batch_organize_optimized(T, labels, all_top1_preds, all_top5_preds):
    batch_size = labels.size(0)

    # Use tensor operations to count frequencies of top-1 and top-5 predictions
    top1_counts_tensor = torch.zeros((batch_size, 1000), dtype=torch.int64, device='cuda')
    top5_counts_tensor = torch.zeros((batch_size, 1000), dtype=torch.int64, device='cuda')

    for t in range(T):
        top1_preds_t = all_top1_preds[t]
        top5_preds_t = all_top5_preds[t]

        # Count top-1 predictions
        top1_counts_tensor.scatter_add_(1, top1_preds_t.unsqueeze(1), torch.ones_like(top1_preds_t.unsqueeze(1)))

        # Flatten the last dimension of top-5 predictions and create a count tensor
        top5_preds_t_flat = top5_preds_t.view(batch_size, -1)
        top5_counts = torch.ones_like(top5_preds_t_flat)

        # Use scatter_add to update the counts for top-5 predictions
        top5_counts_tensor.scatter_add_(1, top5_preds_t_flat, top5_counts)

    return top1_counts_tensor, top5_counts_tensor, labels

def main():
    pr = cProfile.Profile()
    pr.enable()
    torch.manual_seed(77)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    batch_size = 4  # Adjust based on hardware
    num_workers = 2  # Adjust based on hardware
    T = 50  # Number of MC dropout iterations

    test_set = torchvision.datasets.ImageNet(root="c:\\Users\\AAA\\ImageNetValidation\\data", transform=transform, split='val')
    test_loader = DropoutDataLoader(test_set, T, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2)

    # model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)
    # model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1).to(device)
    # model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    # model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
    # model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device) # 이건 쓸 수 없음
    # model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device) # 이건 쓸 수 없음
    # model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).to(device) # 이건 쓸 수 없음  
    model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1).to(device)
    # model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1).to(device)
    # model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).to(device) # 이건 쓸 수 없음
    # model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    print(model)
    model_name = f'{model.__class__.__name__}16_bn'
    print(model_name)

    enable_dropout(model)
    if not check_dropout(model):
        raise RuntimeError("Dropout layers are not enabled.")

    # 모델 스크립팅
    scripted_model = torch.jit.script(model)

    print(f"Performing MC dropout with {T} iterations...")

    all_top1_counts = []
    all_top5_counts = []
    all_labels = []
    all_filenames = []

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels, T) in enumerate(tqdm(test_loader)):
            # if batch_idx >= 20:  # Only process the first 20 batches for testing
            #     break

            images = images.to(device)
            labels = labels.to(device)

            # all_top1_preds, all_top5_preds = preds_torch_parallel(images, T, scripted_model)
            all_top1_preds, all_top5_preds = preds_torch_parallel_optimized(images, T, scripted_model)

            top1_counts, top5_counts, batch_labels = batch_organize_optimized(T, labels, all_top1_preds, all_top5_preds)

            all_top1_counts.append(top1_counts)
            all_top5_counts.append(top5_counts)
            all_labels.append(batch_labels)
            all_filenames.extend([os.path.basename(test_set.imgs[batch_idx * batch_size + i][0]) for i in range(batch_size)])

            if (batch_idx + 1) % 40 == 0:
                elapsed_time = time.time() - start_time
                print(f"Step: {batch_idx + 1} / {len(test_loader)}")
                print(f"Time taken for last 40 batches: {elapsed_time:.2f} seconds")
                start_time = time.time()

    print(f"MC dropout with {T} iterations complete")

    # Concatenate all results
    all_top1_counts = torch.cat(all_top1_counts, dim=0)
    all_top5_counts = torch.cat(all_top5_counts, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print("Transferring results to CPU and converting to dictionaries...")

    labels_np = all_labels.cpu().numpy()
    top1_counts_np = all_top1_counts.cpu().numpy()
    top5_counts_np = all_top5_counts.cpu().numpy()
    
    final_results = []
    for i in range(len(all_filenames)):
        top1_nonzero_indices = np.where(top1_counts_np[i] > 0)
        top5_nonzero_indices = np.where(top5_counts_np[i] > 0)
        
        final_result = {
            'filename': all_filenames[i],
            'true_label': labels_np[i],
            'top1_preds_dropout': {int(k): int(v) for k, v in zip(top1_nonzero_indices[0], top1_counts_np[i][top1_nonzero_indices[0]])},
            'top5_preds_dropout': {int(k): int(v) for k, v in zip(top5_nonzero_indices[0], top5_counts_np[i][top5_nonzero_indices[0]])}
        }
        final_results.append(final_result)

    end_time = time.time()
    print(f"Conversion complete. Time taken: {end_time - start_time:.2f} seconds")


    print("csv 파일 저장중...")
    start_time = time.time()
    # Save results to CSV
    df = pd.DataFrame(final_results)
    df.to_csv(f'mc_dropout{T}_{model_name}_results.csv', index=False)
    end_time = time.time()
    print(f"CSV 파일 저장 완료. 소요시간: {end_time - start_time:.2f} seconds")

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open(f"{model_name}_profile_results.txt", "w") as f:
        f.write(s.getvalue())

if __name__ == "__main__":
    main()
