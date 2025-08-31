import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(20050202)

# 1. 하이퍼파라미터
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균/표준편차
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 클래스 분포 확인
def check_class_distribution():
    class_counts = torch.zeros(10)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    print("클래스별 데이터 수:", class_counts)
    return class_counts

print("MNIST 클래스 분포 확인:")
check_class_distribution()

# 3. 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 → 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 → 7x7
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN().to(DEVICE)

# 4. 손실함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. 학습 함수
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

# 6. 평가 함수
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 7. 학습 및 평가 루프
best_acc = 0.0
patience = 5  # patience 증가
counter = 0

for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader, criterion, optimizer, DEVICE)
    test_acc = evaluate(model, test_loader, DEVICE)
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Test Acc={test_acc:.4f}")
    
    # 각 클래스별 예측 성능 확인 (디버깅용)
    if epoch % 5 == 0:
        model.eval()
        class_correct = torch.zeros(10)
        class_total = torch.zeros(10)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                for i in range(10):
                    class_mask = (labels == i)
                    class_correct[i] += (predicted[class_mask] == labels[class_mask]).sum().item()
                    class_total[i] += class_mask.sum().item()
        
        print("클래스별 정확도:")
        for i in range(10):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                print(f"  클래스 {i}: {acc:.4f} ({int(class_correct[i])}/{int(class_total[i])})")

    if test_acc > best_acc:
        best_acc = test_acc
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"새로운 최고 성능! 정확도: {best_acc:.4f}")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break


# 테스트마다 모델 이름 다르게 해주기
model_name = "test_cnn.pth"

# 8. 모델 저장
torch.save(model.state_dict(), model_name)
print(f"모델이 {model_name}로 저장되었습니다.")