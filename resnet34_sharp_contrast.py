import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from utils import new_folder, fix_seed
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# シードの固定
fix_seed(SEED)

# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {device} device')

# データ前処理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# データセットの読み込み
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
valid_dataset = datasets.ImageFolder(root='data/valid', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# モデルの定義
model = models.resnet34(pretrained=False)
# グレースケール画像に対応
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# 出力層の変更
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# モデルの概要
summary(model, (1, 64, 64))

# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

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

        # 検証
        if iteration % VAL_ITERS == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images_val, labels_val in valid_loader:
                    images_val, labels_val = images_val.to(device), labels_val.to(device)
                    outputs_val = model(images_val)
                    _, predicted = torch.max(outputs_val.data, 1)
                    total += labels_val.size(0)
                    correct += (predicted == labels_val).sum().item()
            accuracy = 100 * correct / total
            print(f'Iteration [{iteration}/{ITERS}], Validation Accuracy: {accuracy:.2f}%')
            model.train()

# モデルの保存
torch.save(model.state_dict(), 'model.pth')
print('Model saved to model.pth')
