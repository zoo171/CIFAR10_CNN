#pip install torch torchvision scikit-learn matplotlib
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. ������ ��ó��
# CIFAR-10 �����ͼ� �ε� �� ������ ���� ����
# - ���� ũ��, �¿� ����, ����ȭ�� ����

# ������ ���� ����
# �̹����� ȸ�� ��Ű�� ������ ��Ȯ���� ������
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # �̹����� �������� �ڸ��� 4�ȼ� �е� �߰� �κ�
    transforms.RandomHorizontalFlip(),    # �̹����� �¿� ���� �κ�
    transforms.ToTensor(),                # �̹����� Tensor�� ��ȯ
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # ��� 0, ǥ������ 1�� ����ȭ ����
])


# �׽�Ʈ �����ͼ¿� ���� ��ȯ ���� �κ�
transform_test = transforms.Compose([
    transforms.ToTensor(),               # �̹����� Tensor�� ��ȯ
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 �����ͼ� �ε�
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)    # �н� �����ͼ�
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)     # �׽�Ʈ �����ͼ�

# ������ �δ� ����
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)   # �н� ������ �δ�
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)    # �׽�Ʈ ������ �δ�

# 2. �� ����
# CNN �� ����
# - 3���� ������� ���� Max Pooling, Fully Connected ������ ����
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # ù ��° ������� ��: �Է� ä�� 3, ��� ä�� 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # �� ��° ������� ��: �Է� ä�� 32, ��� ä�� 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # �� ��° ������� ��: �Է� ä�� 64, ��� ä�� 128
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)         # Max Pooling ��: 2x2 ũ��� �ٿ���ø�

        # Fully Connected ��: �Է� ���� 128*4*4, ��� ���� 256
        self.fc1 = nn.Linear(128 * 4 * 4, 256)                   # ù ��° Fully Connected ��
        self.fc2 = nn.Linear(256, 10)                            # �� ��° Fully Connected ��
        # ������ ������ ���� ��Ӿƿ�
        self.dropout = nn.Dropout(0.5)                          # ��Ӿƿ� ��

    def forward(self, x):
        # Forward propagation ����
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pooling
        x = self.pool(torch.relu(self.conv3(x)))  # Conv3 -> ReLU -> Pooling
        x = x.view(-1, 128 * 4 * 4)               # ����ȭ ���� �κ�
        x = torch.relu(self.fc1(x))              # Fully Connected Layer 1��°
        x = self.dropout(x)                      # ��� �ƿ� ���� �κ�
        x = self.fc2(x)                          # Fully Connected Layer 2��°
        return x

# �� �ʱ�ȭ
model = CNN()

# GPU ��� ���� Ȯ�� �� �� �κ�
# GPU ������ CPU�� ����
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# 3. �н��κ�
# �ս� �Լ��� ��Ƽ������ ����
criterion = nn.CrossEntropyLoss()  # ũ�ν� ��Ʈ���� �ս� �Լ�
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam ��Ƽ������

# �н� �� ����
num_epochs = 42      # �н��� ���� ��
train_losses = []    # �н� �ս� ���
val_losses = []      # ���� �ս� ���

for epoch in range(num_epochs):
    # �н� �ܰ� ���� �ڵ�
    model.train()                   # ���� �н� ���� ����
    running_train_loss = 0.0        # �������� �ս� �ʱ�ȭ�ϴ� �κ�

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        print(f"Processing batch {batch_idx + 1}/{len(train_loader)}")

        optimizer.zero_grad()              # ��� �ʱ�ȭ
        outputs = model(images)            # �� ��� ���
        loss = criterion(outputs, labels)  # �ս� ���
        loss.backward()                    # ������
        optimizer.step()                   # ����ġ ������Ʈ

        running_train_loss += loss.item()   # �ս� ����

    # ��� �ս� ���
    train_losses.append(running_train_loss / len(train_loader))
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_train_loss / len(train_loader):.4f}")

    # ���� �ܰ� ���� �ڵ�
    model.eval()                    # ���� �� ���� ����
    running_val_loss = 0.0          # ���� �ս� �ʱ�ȭ
    with torch.no_grad():           
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)             # �� ��� ���
            loss = criterion(outputs, labels)   # �ս� ���
            running_val_loss += loss.item()     # �ս� ����

    # ��� ���� �ս� ���
    val_losses.append(running_val_loss / len(test_loader))
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {running_val_loss / len(test_loader):.4f}")

# 4. ��
# �׽�Ʈ �����Ϳ��� �� ���� ��
model.eval()        # ���� �� ���� ����
y_true = []         # ���� ���̺�
y_pred = []         # ���� ���̺�

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)        # ���� ���� �������� ���� Ŭ������ ����

        y_true.extend(labels.cpu().numpy())         # ���� ���̺� ����
        y_pred.extend(predicted.cpu().numpy())      # ���� ���̺� ����

accuracy = accuracy_score(y_true, y_pred)                           # ��Ȯ��
precision = precision_score(y_true, y_pred, average='weighted')     # ���е�
recall = recall_score(y_true, y_pred, average='weighted')           # ������
f1 = f1_score(y_true, y_pred, average='weighted')                   # F1 ����


# ��� ��� �κ�
print("----" * 10)
print(f"Accuracy: {accuracy:.4f}")      # ��Ȯ�� ���
print(f"Precision: {precision:.4f}")    # ���е� ���
print(f"Recall: {recall:.4f}")          # ������ ���
print(f"F1-Score: {f1:.4f}")            # F1 ���� ���

# �� ����
torch.save(model.state_dict(), "cnn_cifar10.pth")       # �� �Ķ���� �����ڵ� = torch.save
print("Model saved!")

# ��� �ð�ȭ
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')           # �н� �ս� �׷���
plt.plot(val_losses, label='Validation Loss')           # ���� �ս� �׷���
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()
