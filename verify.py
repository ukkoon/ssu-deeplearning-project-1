import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision import models

csv_file = './datasets/G1020/G1020_test.csv'
image_dir = './datasets/G1020/images'

df = pd.read_csv(csv_file)

class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.dataframe.iloc[idx, 1]
        return image, label

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = CustomDataset(dataframe=df, root_dir=image_dir, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

device = torch.device("mps")

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model.load_state_dict(torch.load('glaucoma_classification_model.pth'))
model = model.to(device)

model.eval()

def evaluate_model(model, dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            
            total += labels.size(0)
            correct += torch.sum(preds == labels.data)

    accuracy = correct.float() / total
    print(f'Accuracy: {accuracy:.4f}')

evaluate_model(model, dataloader)