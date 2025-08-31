# MNIST 손글씨 인식 웹앱 - 학습 과정과 시행착오

## 프로젝트 개요
PyTorch CNN으로 MNIST 손글씨 인식 모델을 학습하고, Flask 웹앱으로 실시간 예측 서비스를 구현

## 주요 시행착오와 해결 과정

### 1. 치명적인 색상 반전 버그 🚨
**문제**: 모델의 실제 예측값이 너무 터무니 없었음  
**원인**: 
```python
# 🚨 문제의 코드
image_array = 255 - image_array  # 불필요한 색상 반전!
```
- 사용자가 검은 캔버스에 흰 글씨로 그림 (이미 MNIST 형태)
- 앱에서 또 색상 반전 → 완전히 다른 데이터 형태가 됨

**해결**: 색상 반전 제거로 즉시 해결

**결론**: 모델을 단순하게도 학습시켜보고, early stopping도 해봤는데 결국 이 단계에서 문제가 있었다니 사소한 fault에서 큰 오류가 나왔네요!

### 2. 오버피팅의 무서움
**문제**: MNIST에서 99.19% 정확도에도 실제 손글씨에서 실패
**원인**: 
- 머신러닝 관련 책에서는 그렇게 죽어라 읽어도 와닿지 않았던 내용인데, 실제로 간단한 데이터셋에서 학습을 해보니 많이 와닿았습니다.
- 모델을 단순하게도 학습시켜보고, early stopping도 해봤는데 결국 전처리 단계에서 문제가 있었다니 사소한 fault에서 큰 오류가 나왔네요!

**더 복잡한 모델도 시도해봤지만...**
mnist는 이 정도의 깊이를 가진 모델은 필요가 없다고 하네요 .. 😭
```python
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
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
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
        x = F.adaptive_avg_pool2d(x, (3, 3))
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
```

**교훈**: Test accuracy만으로는 실제 성능을 보장할 수 없음

### 3. 체계적 디버깅의 중요성
**적용한 디버깅 전략:**
- 단계별 이미지 저장 (`debug_original.png`, `debug_final.png`)
- Raw logits 출력으로 모델 내부 상태 확인
- 정규화 전후 통계 모니터링

## 핵심 배운점

### 1. 로깅 설계의 중요성
- 데이터 클래스별 분포 확인
- 전처리 각 단계별 값 범위 체크
- 모델 출력 분석 (logits, probabilities)

### 2. 오버피팅 감지 신호
- Train/Test accuracy 격차 모니터링
- Early stopping 적절한 patience 설정
- 클래스별 성능 불균형 확인

### 3. 실제 배포시 고려사항
- 학습 데이터와 실제 입력 데이터의 분포 차이
- 전처리 파이프라인의 일관성 보장
- 실시간 성능 모니터링의 필요성

## 프로젝트 구조
```
mnist/
├── dev.py              # 모델 학습 코드
├── app.py              # Flask 웹 서버
├── templates/index.html # 웹 인터페이스
├── best_model.pth      # 최고 성능 모델
└── debug_*.png         # 디버깅용 이미지들
```

## 실행 방법
```bash
# 모델 학습
python dev.py

# 웹앱 실행
python app.py
```

## 기술 스택
- **ML**: PyTorch, torchvision
- **웹**: Flask, Flask-CORS
- **이미지 처리**: PIL, NumPy
- **프론트엔드**: HTML5 Canvas, JavaScript

## 주요 개선사항
1. 체계적인 디버깅 로그 추가
2. 클래스별 성능 분석 기능
3. 이미지 전처리 파이프라인 최적화
4. 모델 파일 관리 체계화
