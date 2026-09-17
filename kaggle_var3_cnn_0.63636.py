import torch
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split


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


#data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),          
    transforms.RandomVerticalFlip(),             
    transforms.RandomRotation(15),               
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.3),            
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# BASE = '/kaggle/input/datasets/iuliamariapopescu/signal-object-detection'

train_df = pd.read_csv(f'train.csv')
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])
print(f"Train: {len(train_df)} images | Validation: {len(val_df)} images")

train_dataset = SignalDataset(train_df, f'train', transform=train_transform)
val_dataset = SignalDataset(val_df,   f'train', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  pin_memory=True, num_workers=0)
val_loader = DataLoader(val_dataset,   batch_size=64, shuffle=False, pin_memory=True, num_workers=0)


# model class
# using silu instead of relu
#bigger dropout
class ImprovedResidualCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # 3 -> 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2, 2)

        # 32 -> 64 channels
        self.conv2    = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2      = nn.BatchNorm2d(64)
        self.conv3    = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3      = nn.BatchNorm2d(64)
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        # 64 -> 128 channels
        self.conv4    = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4      = nn.BatchNorm2d(128)
        self.conv5    = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5      = nn.BatchNorm2d(128)
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128)
        )

        # final classification
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.6)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 1
        x = self.pool(F.silu(self.bn1(self.conv1(x))))

        # 2
        identity = self.shortcut1(x)
        out = F.silu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        out = F.silu(out + identity)
        out = self.pool(out)

        # 3
        identity = self.shortcut2(out)
        out2 = F.silu(self.bn4(self.conv4(out)))
        out2 = self.bn5(self.conv5(out2))
        out2 = F.silu(out2 + identity)

        # clasification
        out2 = self.adaptive_pool(out2)
        out2 = torch.flatten(out2, 1)
        out2 = self.dropout(out2)
        out2 = F.silu(self.fc1(out2))
        out2 = self.dropout(out2)
        out2 = self.fc2(out2)

        return out2

# function for training the model
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
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

            # gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = 100 * train_correct/train_total

        # validation
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
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total

        scheduler.step()

        print(f'Epoch {epoch+1}/{num_epochs} | 'f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | 'f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        #saving the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model nou salvat! (best val acc: {best_val_acc:.2f}%)')

    print(f'\nAntrenament terminat. Cel mai bun Val Acc: {best_val_acc:.2f}%')


# start training
if torch.backends.mps.is_available():
    device = torch.device("mps") 
elif torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")  
print(f"Using device {device}")


model = ImprovedResidualCNN(num_classes=5).to(device)

num_epochs = 60                        
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-3) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

print("Start training...")
train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)



def generate_submission(model_path, test_csv, test_dir, transform, device, output_csv='sample_submission.csv'):
    print("Submission file loading...")

    model = ImprovedResidualCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    test_df = pd.read_csv(test_csv)
    test_dataset = SignalDataset(test_df, test_dir, transform=transform, is_test=True)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted + 1 
            predictions.extend(predicted.cpu().numpy())

    submission_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    submission_df.to_csv(output_csv, index=False)
    print(f"File {output_csv} has been successfully created!")


generate_submission(model_path='best_model.pth', test_csv=f'test.csv', test_dir=f'test', transform=val_transform, device=device, output_csv='sample_submission.csv')