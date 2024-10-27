import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models
import multiprocessing
multiprocessing.set_start_method('fork')

csv_file = './datasets/G1020/G1020_train.csv'

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
        label = self.dataframe.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

image_dir = './datasets/G1020/images'
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = CustomDataset(dataframe=df, root_dir=image_dir, transform=data_transforms)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

device = torch.device("mps")

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 1)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, criterion, optimizer, dataloader, dataset_size, num_epochs=20):
    train_loss_history = []
    train_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
        
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                preds = torch.sigmoid(outputs)
                preds = preds > 0.5
                loss = criterion(outputs, labels.unsqueeze(1))
                
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.unsqueeze(1).data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.float() / dataset_size

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print('Training complete')

    plt.figure(figsize=(12, 5))
        
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_history, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Epoch vs. Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc_history, label="Train Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Epoch vs. Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

train_model(model, criterion, optimizer, dataloader, len(dataset), num_epochs=20)

torch.save(model.state_dict(), 'glaucoma_classification_model.pth')
