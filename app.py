from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64

from PIL import Image

app = Flask(__name__)
CORS(app)  # 프론트엔드와 통신을 위해

# 개선된 CNN 모델 클래스 (학습할 때와 동일해야 함)
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

# 모델 로드
model = SimpleCNN()
model.load_state_dict(torch.load('/Users/ldrugsnw/mnist/best_model.pth'))
model.eval()

print("모델이 성공적으로 로드되었습니다!")

def preprocess_image(image_data):
      """Base64 이미지 데이터를 모델 입력 형태로 변환"""
      # Base64 디코딩
      image_data = image_data.split(',')[1]  # data:image/png;base64, 제거
      image_bytes = base64.b64decode(image_data)

      # PIL 이미지로 변환
      image = Image.open(io.BytesIO(image_bytes)).convert('L')  # 흑백 변환

      # 원본 이미지 저장 (디버그용)
      image.save("debug_original.png")

      # 배경을 검정, 글자를 흰색으로 (MNIST와 동일하게)
      image_array = np.array(image)
      print(f"원본 이미지 범위: {image_array.min()} ~ {image_array.max()}")

      # 이진화 (Thresholding) - 회색 제거
      image_array[image_array < 128] = 0
      image_array[image_array >= 128] = 255
      print(f"이진화 후 고유값: {np.unique(image_array)}")

      # 중앙 정렬 및 크롭핑
      coords = np.column_stack(np.where(image_array > 30))

      if len(coords) > 0:
          x0, y0 = coords.min(axis=0)
          x1, y1 = coords.max(axis=0)
          cropped = image_array[x0:x1+1, y0:y1+1]
          print(f"크롭된 크기: {cropped.shape}")

          # MNIST 스타일: 20x20 크기로 맞추고 4픽셀 패딩
          h, w = cropped.shape
          max_dim = max(h, w)

          # 20x20에 맞게 스케일링
          if max_dim > 20:
              scale = 20.0 / max_dim
              new_h, new_w = int(h * scale), int(w * scale)
              cropped_pil = Image.fromarray(cropped.astype(np.uint8))
              cropped_pil = cropped_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
              cropped = np.array(cropped_pil)
              h, w = cropped.shape
              print(f"스케일링 후 크기: {h}x{w}")

          # 28x28 캔버스 중앙에 배치
          padded = np.zeros((28, 28))
          start_x = (28 - h) // 2
          start_y = (28 - w) // 2
          padded[start_x:start_x+h, start_y:start_y+w] = cropped
          image_array = padded
      else:
          print("숫자를 찾을 수 없음!")
          image_array = np.zeros((28, 28))

      # 최종 이미지 저장
      img_to_save = image_array.astype(np.uint8)
      Image.fromarray(img_to_save).save("debug_final.png")

      # 정규화 및 텐서 변환
      image_array = image_array.astype(np.float32) / 255.0
      print(f"정규화 전 범위: {image_array.min()} ~ {image_array.max()}")

      image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
      image_tensor = (image_tensor - 0.1307) / 0.3081  # MNIST 정규화
      print(f"정규화 후 범위: {image_tensor.min():.4f} ~ {image_tensor.max():.4f}")
      print(f"텐서 평균: {image_tensor.mean():.4f}, 표준편차: {image_tensor.std():.4f}")

      return image_tensor
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSON 데이터 받기
        data = request.get_json()
        image_data = data['image']
        
        # 이미지 전처리
        processed_image = preprocess_image(image_data)
        
        # 예측
        with torch.no_grad():
            output = model(processed_image)

            # ★ 중요: raw logits 확인
            print("Raw logits:", output.numpy())
            print("Logits 분포:", f"min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

            # 예측값 디버깅 출력
            print("output:", output)
            print("probabilities:", probabilities)
            print("predicted_class:", predicted_class)
            print("confidence:", confidence)
        
        # 모든 클래스의 확률
        all_probabilities = probabilities[0].tolist()
        
        return jsonify({
            'prediction': int(predicted_class),
            'confidence': float(confidence),
            'probabilities': all_probabilities,
            'raw_logits': output[0].tolist()  # ★ 디버깅용 추가
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)