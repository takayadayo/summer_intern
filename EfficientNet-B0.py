import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from utils import new_folder, fix_seed
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# シードの固定
fix_seed(SEED)

# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {device} device')

# データ前処理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),  # EfficientNetは3チャンネル入力を期待
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomAutocontrast(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# データセットの読み込み
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
valid_dataset = datasets.ImageFolder(root='data/valid', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# モデルの定義（EfficientNet-B0を使用）
model = models.efficientnet_b0(pretrained=False)
# 出力層の変更
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)
model = model.to(device)

# モデルの概要
summary(model, (3, 64, 64))

# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 学習率スケジューラの設定
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 学習ループ
model.train()
iteration = 0
while iteration < ITERS:
    for images, labels in train_loader:
        iteration += 1
        if iteration > ITERS:
            break
        images, labels = images.to(device), labels.to(device)

        # フォワードプロパゲーション
        outputs = model(images)
        loss = criterion(outputs, labels)

        # バックプロパゲーションと最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 学習率のステップ
        scheduler.step()

        # 検証
        if iteration % VAL_ITERS == 0:
            model.eval()
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images_val, labels_val in valid_loader:
                    images_val, labels_val = images_val.to(device), labels_val.to(device)
                    outputs_val = model(images_val)
                    _, predicted = torch.max(outputs_val.data, 1)
                    total += labels_val.size(0)
                    correct += (predicted == labels_val).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels_val.cpu().numpy())
            accuracy = 100 * correct / total
            print(f'Iteration [{iteration}/{ITERS}], Validation Accuracy: {accuracy:.2f}%')

            # 分類レポートと混同行列の生成
            print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(6,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(f'confusion_matrix_{iteration}.png')
            plt.close()

            model.train()

# モデルの保存
torch.save(model.state_dict(), 'model.pth')
print('Model saved to model.pth')
