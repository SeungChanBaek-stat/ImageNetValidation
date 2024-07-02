# mc_dropout_Alexnet_results.csv 읽기
import sys, os
sys.path.append(os.pardir)
import pandas as pd
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
from typing import Dict, List, Optional, Tuple
import pickle

data_path = os.path.join('c:\\', 'Users', 'AAA', 'ImageNetValidation', 'data')
imagenet_folder_path = os.path.join(data_path, 'val')
meta_file_path = os.path.join(data_path, 'meta.bin')


META_FILE = 'meta.bin'

# Define the load_meta_file function
def check_integrity(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0

def load_meta_file(root: str, file: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file)
    else:
        msg = ("The meta file {} is not present in the root directory or is corrupted. "
               "This file is automatically created by the ImageNet dataset.")
        raise RuntimeError(msg.format(file, root))

# Load the meta.bin file
meta = load_meta_file(data_path, META_FILE)

# # print(meta[0])



# # Extract synsets and create a mapping from synset ID to the first class name
# synset_to_class = {key: values[0] for key, values in meta[0].items()}



# Extract synsets and create a mapping from synset ID to the first or second class name based on the condition
synset_to_class = {}
for key, values in meta[0].items():
    if key == 'n03710721':
        synset_to_class[key] = values[1]
    elif key == 'n02012849':
        synset_to_class[key] = 'crane bird'
    else:
        synset_to_class[key] = values[0]

# Print a few entries to verify
print({k: synset_to_class[k] for k in list(synset_to_class)[:5]})








# Define transformations for the dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# ImageFolder를 사용하여 데이터셋 로드
imagenet_folder_dataset = ImageFolder(root=imagenet_folder_path, transform=transform)

# Invert the class_to_idx dictionary to get idx_to_class
idx_to_class_by_folder_dataset = {v: k for k, v in imagenet_folder_dataset.class_to_idx.items()}

# Create a new dictionary to map from class index to actual class name using synset_to_class
idx_to_class_name = {idx: synset_to_class[synset_id] for idx, synset_id in idx_to_class_by_folder_dataset.items()}

# # Print the idx_to_class_name dictionary to verify
# print(idx_to_class_name)


# # idx_to_class_by_folder_dataset key 갯수 세기
# print(len(idx_to_class_name))
# print({k: idx_to_class_name[k] for k in list(idx_to_class_name)[:10]})


# idx_to_class_name 에서 value값이 중복되는 게 있으면 그 key값 출력
for key, value in idx_to_class_name.items():
    if list(idx_to_class_name.values()).count(value) > 1:
        print(key, value)
    else:
        continue
# print("Done")




print("idx_to_class_name 저장중...")
# Save the dictionary to a pickle file
with open('idx_to_class_name.pkl', 'wb') as f:
    pickle.dump(idx_to_class_name, f)
print("idx_to_class_name 저장끝...")
# # Load the dictionary from the pickle file
# with open('idx_to_class_name.pkl', 'rb') as f:
#     idx_to_class_name = pickle.load(f)



# # Check the class names for indices 134 and 517
# class_134 = idx_to_class_by_folder_dataset.get(134, 'Unknown')
# class_517 = idx_to_class_by_folder_dataset.get(517, 'Unknown')

# print(f"Class index 134 corresponds to folder: {class_134}")
# print(f"Class index 517 corresponds to folder: {class_517}")