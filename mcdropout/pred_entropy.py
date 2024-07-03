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
import pickle
import numpy as np
from scipy.stats import ttest_ind


# dropout iterations
T = 50

# Define the path to the meta.bin file
data_path = os.path.join('c:\\', 'Users', 'AAA', 'ImageNetValidation', 'data')
meta_file_path = os.path.join(data_path, 'meta.bin')
idx_to_class_name_path = os.path.join(data_path, 'idx_to_class_name.pkl')

# Define transformations for the dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Load the ImageNet dataset to get class_to_idx
imagenet_dataset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transform)




# Load the dictionary from the pickle file
with open(idx_to_class_name_path, 'rb') as f:
    idx_to_class_name = pickle.load(f)

print(idx_to_class_name[0])

# print("idx_to_class_name:", idx_to_class_name)









# Read the CSV file
df = pd.read_csv("c:\\Users\\AAA\\ImageNetValidation\\mcdropout\\mcdropout_50_csv\\mc_dropout50_AlexNet_results.csv")
# print(df.head())



# predictive entropy calculator, using numpy
def predictive_entropy(probs_dict):
    probs = [v for k, v in probs_dict.items()]
    # print("probs:", probs)
    entropy = -np.sum(probs * np.log(probs))
        
    return entropy

# Convert the string representation of dictionary to a dictionary
df['top1_preds_dropout'] = df['top1_preds_dropout'].apply(lambda x: eval(x))
df['top5_preds_dropout'] = df['top5_preds_dropout'].apply(lambda x: eval(x))

df['top1_preds_distribution'] = df['top1_preds_dropout'].apply(lambda x: {k: v / T for k, v in x.items()})

# 'top1_preds_distribution'열의 각 행에 대해서 predictive_entropy 함수를 적용하여 'top1_preds_entropy'열 생성
df['top1_preds_entropy'] = df['top1_preds_distribution'].apply(predictive_entropy)

# 'top1_preds_distribution' 에서 가장 value 값이 큰 값을 'top1_preds'에 저장
df['top1_preds'] = df['top1_preds_distribution'].apply(lambda x: max(x, key=x.get))


# top1_preds_ditribution에서 true_label에 해당하는 value값을 'true_label_prob'에 저장, 없으면 0
df['true_label_prob'] = df.apply(lambda x: x['top1_preds_distribution'].get(x['true_label'], 0), axis=1)



# true_label과 top1_preds가 같은 경우에 대해서 'correct'열 생성
df['correct'] = (df['true_label'] == df['top1_preds']).astype(int)

# 'correct'열에 대해서 누적합을 계산하고 'cumulative_correct'열 생성
df['cumulative_correct'] = df['correct'].cumsum()

# 'cumulative_correct'열을 이용하여 'cumulative_accuracy'열 생성
df['cumulative_accuracy'] = df['cumulative_correct'] / (df.index + 1)

# 'cumulative_accuracy'열 시각화
plt.figure(figsize=(10, 5))
plt.plot(df['cumulative_accuracy'])
plt.xlabel('Number of samples')
plt.ylabel('Accuracy')
plt.title('Cumulative accuracy')
plt.show()

# 최종 정확도 계산
final_accuracy = df['correct'].mean()
print(f'Final accuracy: {final_accuracy:.4f}')



# top1_preds_entropy 에 대해서 각 행별로 누적평균을 계산하고 시각화
df['top1_preds_entropy_cumavg'] = df['top1_preds_entropy'].expanding().mean()

# top1_preds_entropy_cumavg 시각화
plt.figure(figsize=(10, 5))
plt.plot(df['top1_preds_entropy_cumavg'])
plt.xlabel('Number of samples')
plt.ylabel('Predictive entropy')
plt.title('Cumulative average predictive entropy')
plt.show()


# top1_preds_entropy 히스토그램 시각화
plt.figure(figsize=(10, 5))
plt.hist(df['top1_preds_entropy'], bins=50)
plt.xlabel('Predictive entropy')
plt.ylabel('Frequency')
plt.title('Predictive entropy histogram')
plt.show()





# correlation analysis : top1_preds_entropy 와 true_label_prob 사이의 상관관계
correlation = df['top1_preds_entropy'].corr(df['true_label_prob'])
print(f'Correlation between predictive entropy and true label probability: {correlation:.4f}')





# Assuming the 'correct' column is already computed as in your previous code
# 'correct' = 1 for correct predictions and 0 for incorrect predictions

# Separate the data into two groups
correct_predictions = df[df['correct'] == 1]
incorrect_predictions = df[df['correct'] == 0]

# Calculate mean entropy for each group
mean_entropy_correct = correct_predictions['top1_preds_entropy'].mean()
mean_entropy_incorrect = incorrect_predictions['top1_preds_entropy'].mean()

print(f"Mean entropy for correct predictions: {mean_entropy_correct}")
print(f"Mean entropy for incorrect predictions: {mean_entropy_incorrect}")

# Perform t-test
t_stat, p_value = ttest_ind(correct_predictions['top1_preds_entropy'], incorrect_predictions['top1_preds_entropy'])

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Interpret the results
if p_value < 0.05:
    print("The difference in mean entropy between correct and incorrect predictions is statistically significant.")
else:
    print("The difference in mean entropy between correct and incorrect predictions is not statistically significant.")






print(df.head())