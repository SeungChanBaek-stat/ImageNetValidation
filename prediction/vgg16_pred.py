import sys, os
sys.path.append(os.pardir)
from functions import BaseTransform, ImageNetDataset, pred_model
import torch
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from torchviz import make_dot
from tqdm import tqdm
import pandas as pd

# # vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
# weights = models.VGG16_Weights.IMAGENET1K_V1
# vgg16 = models.vgg16(weights=weights)
# vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
# alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)



def main():
    # Step 1: Set up the environment
    SEED = 77
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dtype = torch.float32


    # Step 2: Load ground truth and meta data
    ground_truth_path = "c:\\Users\\AAA\\Downloads\\ILSVRC2012_devkit_t12\\data\\ILSVRC2012_validation_ground_truth.txt"
    meta_data_path = "c:\\Users\\AAA\\Downloads\\ILSVRC2012_devkit_t12\\data\\meta.mat"

    # Load ground truth labels
    with open(ground_truth_path, 'r') as f:
        ground_truth = [int(line.strip()) for line in f.readlines()]
    print(f'Number of ground truth labels: {len(ground_truth)}')
    print("Ground truth labels loaded")






    # Step 3: Prepare the model
    weights = models.VGG16_Weights.IMAGENET1K_V1
    vgg16 = models.vgg16(weights=weights).to(device)
    vgg16.eval()
    print(vgg16)



    # 전처리 클래스    
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, .224, 0.225)
    transform = BaseTransform(resize, mean, std)





    # Step 7: Prepare the dataset and perform inference
    val_images_path = "C:\\Users\\AAA\\Downloads\\val_images"
    val_image_paths = [os.path.join(val_images_path, f'ILSVRC2012_val_{i:08d}.JPEG') for i in range(1, 50001)]
    val_labels = ground_truth

    # Create the dataset
    val_dataset = ImageNetDataset(val_image_paths, val_labels, transform=transform)




    # Perform inference
    batch_size = 32
    num_workers = 2  # Adjust based on your system capabilities
    y_pred_all_list = pred_model(vgg16, val_dataset, batch_size, num_workers, device)

    # Step 8: Calculate accuracy
    correct = 0
    for i, prediction in enumerate(y_pred_all_list):
        if np.argmax(prediction) == val_labels[i]:
            correct += 1

    accuracy = correct / len(val_labels) * 100
    print(f'Accuracy: {accuracy:.2f}%')





if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()