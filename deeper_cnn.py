import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 개선된 CNN 모델
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # 첫 번째 블록
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 두 번째 블록
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # 세 번째 블록
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.dropout_conv = nn.Dropout2d(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # 28->14->7->3 (반올림)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        # 첫 번째 블록: 28x28 -> 14x14
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # 두 번째 블록: 14x14 -> 7x7
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # 세 번째 블록: 7x7 -> 3x3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (3, 3))  # 정확히 3x3로 만들기
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# 데이터 변환 정의 (데이터 증강 포함)
transform_train = transforms.Compose([
    transforms.RandomRotation(10),  # ±10도 회전
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.1, 0.1),  # 최대 10% 이동
        scale=(0.9, 1.1),      # 90%~110% 크기 변화
        shear=5                # 최대 5도 기울임
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # 랜덤하게 일부 지우기
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

# 데이터셋 로드
print("데이터셋 로딩 중...")
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transform_train, 
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transform_test
)

# 데이터로더
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

device = torch.device('cpu')
print(f"사용 디바이스: {device}")

# 모델, 손실함수, 옵티마이저 정의
model = ImprovedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 학습률 스케줄러
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)

print("모델 구조:")
print(model)
print(f"\n총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# 테스트 함수
def test_model():
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f'테스트 정확도: {accuracy:.2f}% ({correct}/{total})')
    print(f'테스트 손실: {avg_loss:.6f}')
    
    return accuracy, avg_loss

# 학습 함수
def train_model(num_epochs=15):
    print(f"\n=== {num_epochs} 에포크 학습 시작 ===")
    
    # 결과 저장용
    train_losses = []
    test_accuracies = []
    test_losses = []
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        print(f"\n에포크 {epoch + 1}/{num_epochs}")
        print(f"현재 학습률: {optimizer.param_groups[0]['lr']:.6f}")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 순전파
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 통계
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()
            total_train += target.size(0)
            
            # 진행상황 출력
            if batch_idx % 100 == 0:
                print(f'  배치 {batch_idx:3d}/{len(train_loader)}, '
                      f'손실: {loss.item():.6f}, '
                      f'훈련 정확도: {100 * correct_train / total_train:.2f}%')
        
        # 에포크 결과
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        
        print(f'\n에포크 {epoch + 1} 훈련 완료:')
        print(f'  평균 훈련 손실: {avg_train_loss:.6f}')
        print(f'  훈련 정확도: {train_accuracy:.2f}%')
        
        # 테스트 평가
        test_acc, test_loss = test_model()
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
        # 최고 성능 모델 저장
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_mnist_model.pth')
            print(f'  *** 최고 성능 갱신! 모델 저장됨 (정확도: {best_accuracy:.2f}%) ***')
        
        # 학습률 조정
        scheduler.step()
        
        print("-" * 60)
    
    print(f"\n🎉 학습 완료!")
    print(f"최고 테스트 정확도: {best_accuracy:.2f}%")
    
    # 최종 모델도 저장
    torch.save(model.state_dict(), 'final_mnist_model.pth')
    
    return train_losses, test_accuracies, test_losses

# 결과 시각화 함수
def plot_results(train_losses, test_accuracies, test_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 손실 그래프
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='훈련 손실', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='테스트 손실', linewidth=2)
    ax1.set_xlabel('에포크')
    ax1.set_ylabel('손실')
    ax1.set_title('훈련/테스트 손실')
    ax1.legend()
    ax1.grid(True)
    
    # 정확도 그래프
    ax2.plot(epochs, test_accuracies, 'g-', label='테스트 정확도', linewidth=2)
    ax2.set_xlabel('에포크')
    ax2.set_ylabel('정확도 (%)')
    ax2.set_title('테스트 정확도')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"최고 정확도: {max(test_accuracies):.2f}%")
    print(f"최고 정확도 달성 에포크: {test_accuracies.index(max(test_accuracies)) + 1}")

# 학습 실행
if __name__ == "__main__":
    # 초기 테스트
    print("초기 모델 성능:")
    test_model()
    
    # 학습 시작
    train_losses, test_accuracies, test_losses = train_model(num_epochs=15)
    
    # 결과 시각화
    plot_results(train_losses, test_accuracies, test_losses)
    
    print("\n모델 파일이 저장되었습니다:")
    print("- best_mnist_model.pth: 최고 성능 모델")
    print("- final_mnist_model.pth: 최종 모델")
