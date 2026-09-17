import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from sklearn.model_selection import train_test_split

# img = Image.open('train/feab9f01-f056-478e-9b3c-ac0567231df6.png')
# print(img.size)   # (width, height)
# print(img.mode)   # RGB, L, etc.

# arr = np.array(img)
# print(arr.shape)           # (128, 55, 4)
# print(arr[:,:,3].min(), arr[:,:,3].max())  # alpha channels values

# print("PyTorch version:", torch.__version__)
# print("GPU disponibil:", torch.cuda.is_available())


class SignalDataset(Dataset):
    def __init__(self, df, image_directory, transform=None, is_test=False): 
        self.df = df
        self.image_directory = image_directory
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        image_path = f"{self.image_directory}/{self.df['id'].iloc[idx]}"
        image = Image.open(image_path).convert('RGB') 
        image = self.transform(image)
        if self.is_test:
            return image
        label = self.df['label'].iloc[idx] - 1  # 1-5 -> 0-4
        return image, label
    
# data augmentation     
train_transform = transforms.Compose([ 
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomVerticalFlip(), 
    transforms.RandomErasing(p=0.25) 
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_df = pd.read_csv('/kaggle/input/datasets/iuliamariapopescu/signal-object-detection/train.csv')
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])
# train_df — 90% images
# val_df — 10% iamges

train_dataset = SignalDataset(train_df,'/kaggle/input/datasets/iuliamariapopescu/signal-object-detection/train', transform=train_transform)
val_dataset = SignalDataset(val_df,'/kaggle/input/datasets/iuliamariapopescu/signal-object-detection/train', transform=val_transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  pin_memory=True)
val_loader = DataLoader(val_dataset,   batch_size=32, shuffle=False, pin_memory=True)

# print(f"Train batches: {len(train_loader)}")
# print(f"Val batches: {len(val_loader)}")

# model class
class ImprovedResidualCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ImprovedResidualCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
     
        identity = self.shortcut(x)
        
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        
        out += identity
        out = F.relu(out)
        
        out = self.adaptive_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
    
# function for training the model
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')  
            

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# model
model = ImprovedResidualCNN(num_classes=5).to(device)

# hyperparameters
num_epochs = 20
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

print("Start training...")
train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)            


def generate_submission(model_path, test_csv, test_dir, transform, output_csv='sample_submission.csv'):
    print("Submission file loading...")
    
    # load the best model 
    model = ImprovedResidualCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # read test data
    test_df = pd.read_csv(test_csv)
    test_dataset = SignalDataset(test_df, test_dir, transform=transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())
            
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    })
    
    submission_df.to_csv(output_csv, index=False)
    print("sample_submission has been successfully created!")


generate_submission(model_path='best_model.pth', test_csv='/kaggle/input/datasets/iuliamariapopescu/signal-object-detection/test.csv', 
    test_dir='/kaggle/input/datasets/iuliamariapopescu/signal-object-detection/test', transform=val_transform, output_csv='sample_submission.csv')