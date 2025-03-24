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

# 1. 데이터 전처리
# CIFAR-10 데이터셋 로드 및 데이터 증강 수행
# - 랜덤 크롭, 좌우 반전, 정규화를 적용

# 데이터 증강 정의
# 이미지를 회전 시키는 과정은 정확도를 낮췄음
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 이미지를 랜덤으로 자르고 4픽셀 패딩 추가 부분
    transforms.RandomHorizontalFlip(),    # 이미지를 좌우 반전 부분
    transforms.ToTensor(),                # 이미지를 Tensor로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 평균 0, 표준편차 1로 정규화 진행
])


# 테스트 데이터셋에 대한 변환 정의 부분
transform_test = transforms.Compose([
    transforms.ToTensor(),               # 이미지를 Tensor로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 데이터셋 로드
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)    # 학습 데이터셋
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)     # 테스트 데이터셋

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)   # 학습 데이터 로더
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)    # 테스트 데이터 로더

# 2. 모델 구현
# CNN 모델 정의
# - 3개의 컨볼루션 층과 Max Pooling, Fully Connected 층으로 구성
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 첫 번째 컨볼루션 층: 입력 채널 3, 출력 채널 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 두 번째 컨볼루션 층: 입력 채널 32, 출력 채널 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 세 번째 컨볼루션 층: 입력 채널 64, 출력 채널 128
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)         # Max Pooling 층: 2x2 크기로 다운샘플링

        # Fully Connected 층: 입력 차원 128*4*4, 출력 차원 256
        self.fc1 = nn.Linear(128 * 4 * 4, 256)                   # 첫 번째 Fully Connected 층
        self.fc2 = nn.Linear(256, 10)                            # 두 번째 Fully Connected 층
        # 과적합 방지를 위한 드롭아웃
        self.dropout = nn.Dropout(0.5)                          # 드롭아웃 층

    def forward(self, x):
        # Forward propagation 정의
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pooling
        x = self.pool(torch.relu(self.conv3(x)))  # Conv3 -> ReLU -> Pooling
        x = x.view(-1, 128 * 4 * 4)               # 평탕화 진행 부분
        x = torch.relu(self.fc1(x))              # Fully Connected Layer 1번째
        x = self.dropout(x)                      # 드롭 아웃 진행 부분
        x = self.fc2(x)                          # Fully Connected Layer 2번째
        return x

# 모델 초기화
model = CNN()

# GPU 사용 여부 확인 및 모델 부분
# GPU 없으면 CPU로 진행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# 3. 학습부분
# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()  # 크로스 엔트로피 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저

# 학습 및 검증
num_epochs = 42      # 학습할 에폭 수
train_losses = []    # 학습 손실 기록
val_losses = []      # 검증 손실 기록

for epoch in range(num_epochs):
    # 학습 단계 진행 코드
    model.train()                   # 모델을 학습 모드로 설정
    running_train_loss = 0.0        # 에폭마다 손실 초기화하는 부분

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        print(f"Processing batch {batch_idx + 1}/{len(train_loader)}")

        optimizer.zero_grad()              # 경사 초기화
        outputs = model(images)            # 모델 출력 계산
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()                    # 역전파
        optimizer.step()                   # 가중치 업데이트

        running_train_loss += loss.item()   # 손실 누적

    # 평균 손실 기록
    train_losses.append(running_train_loss / len(train_loader))
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_train_loss / len(train_loader):.4f}")

    # 검증 단계 진행 코드
    model.eval()                    # 모델을 평가 모드로 설정
    running_val_loss = 0.0          # 검증 손실 초기화
    with torch.no_grad():           
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)             # 모델 출력 계산
            loss = criterion(outputs, labels)   # 손실 계산
            running_val_loss += loss.item()     # 손실 누적

    # 평균 검증 손실 기록
    val_losses.append(running_val_loss / len(test_loader))
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {running_val_loss / len(test_loader):.4f}")

# 4. 평가
# 테스트 데이터에서 모델 성능 평가
model.eval()        # 모델을 평가 모드로 설정
y_true = []         # 실제 레이블
y_pred = []         # 예측 레이블

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)        # 가장 높은 예측값을 가진 클래스를 선택

        y_true.extend(labels.cpu().numpy())         # 실제 레이블 저장
        y_pred.extend(predicted.cpu().numpy())      # 예측 레이블 저장

accuracy = accuracy_score(y_true, y_pred)                           # 정확도
precision = precision_score(y_true, y_pred, average='weighted')     # 정밀도
recall = recall_score(y_true, y_pred, average='weighted')           # 재현율
f1 = f1_score(y_true, y_pred, average='weighted')                   # F1 점수


# 결과 출력 부분
print("----" * 10)
print(f"Accuracy: {accuracy:.4f}")      # 정확도 출력
print(f"Precision: {precision:.4f}")    # 정밀도 출력
print(f"Recall: {recall:.4f}")          # 재현율 출력
print(f"F1-Score: {f1:.4f}")            # F1 점수 출력

# 모델 저장
torch.save(model.state_dict(), "cnn_cifar10.pth")       # 모델 파라미터 저장코드 = torch.save
print("Model saved!")

# 결과 시각화
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')           # 학습 손실 그래프
plt.plot(val_losses, label='Validation Loss')           # 검증 손실 그래프
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()
