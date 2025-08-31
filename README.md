# MNIST 손글씨 인식 웹앱 - 학습 과정과 시행착오

## 프로젝트 개요
PyTorch CNN으로 MNIST 손글씨 인식 모델을 학습하고, Flask 웹앱으로 실시간 예측 서비스를 구현

## 주요 시행착오와 해결 과정

### 1. 치명적인 색상 반전 버그 🚨
**문제**: 모델이 항상 0만 예측  
**원인**: 
```python
# 🚨 문제의 코드
image_array = 255 - image_array  # 불필요한 색상 반전!
```
- 사용자가 검은 캔버스에 흰 글씨로 그림 (이미 MNIST 형태)
- 앱에서 또 색상 반전 → 완전히 다른 데이터 형태가 됨

**해결**: 색상 반전 제거로 즉시 해결

### 2. 오버피팅의 무서움
- MNIST에서 99.19% 정확도에도 실제 손글씨에서 실패
- **교훈**: Test accuracy만으로는 실제 성능을 보장할 수 없음

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
