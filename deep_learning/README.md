# Deep Learning

[What is the class of this image ?](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)

## 심층 신경망

### 정확도 향상

- 앙상블 학습
- 학습률 감소
- 데이터 확장 data augmentation
  - 이미지 회전, 이동으로 데이터 수 증가

### 장점

1. 필요한 신경망의 매개변수 수가 적음
   - 적은 매개변수로 더 높은 표현력
     - 5x5 합성곱 연산 1회: 매개변수 25개 = 1회 * 5칸 * 5칸
     - 3x3 합성곱 연산 2회: 매개변수 18개 = 2회 * 3칸 * 3칸
   - 넓은 수용 영역 receptive field
   - 비선형 활성화 함수들이 겹쳐 복잡한 표현 가능
2. 분할 정복
   - 학습할 문제를 계층적으로 분해
   - 낮은 층은 더 단순한 문제 학습
   - 깊은 층은 낮은 층의 학습 정보를 활용
3. 학습 효율성 증가
   - 단순한 문제 = 필요한 데이터 적음
   - 학습 데이터 양 ↓
   - 학습 속도 ↑

## 역사

### 이미지 인식 대회

[Large Scale Visual Recognition Challenge](http://www.image-net.org/challenges/LSVRC/) (ILSVRC)

- [ImageNet](http://www.image-net.org/): 이미지 데이터베이스
- [Challenges](http://image-net.org/challenges/LSVRC/2016/index#comp)
  - Object localization
  - Object detection
  - Object detection from video
  - Scene classification
  - Scene parsing

### AlexNet

- [Wikipedia](https://en.wikipedia.org/wiki/AlexNet)

### VGG

Visual Geometry Group 16/19 Layers

모든 합성곱 레이어에서 3x3 필터 사용

- 논문: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [리뷰 1](https://medium.com/@msmapark2/vgg16-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-very-deep-convolutional-networks-for-large-scale-image-recognition-6f748235242a)

### GoogLeNet

인셉션 구조: 연산량을 늘리지 않고서 네트워크 크기를 증가

1x1 필터 목적:
1. 컴퓨터 병목현상을 제거하기 위한 차원축소 모듈
2. 네트워크 크기 제한

- 논문: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- [리뷰 1](https://leedakyeong.tistory.com/entry/%EB%85%BC%EB%AC%B8-GoogleNet-Inception-%EB%A6%AC%EB%B7%B0-Going-deeper-with-convolutions-1)
- [리뷰 2](https://ikkison.tistory.com/86)

### ResNet

모델은 너무 깊어지면 성능이 떨어진다.  
Degradation: 기울기가 너무 작아져 학습이 되지 않음

아무리 깊어지더라도 최소 기울기를 보장하여 소실 문제를 해결한다.

- 논문: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [리뷰 1](https://arclab.tistory.com/163)
- [리뷰 2](https://jxnjxn.tistory.com/22)
- [리뷰 3](https://ganghee-lee.tistory.com/41)
- [리뷰 4](https://leechamin.tistory.com/184)

### GPU

Graphics Processing Unit

#### 계산 능력

처리 시간 대부분이 합성곱 계층에서 소요된다.

- CUDA
- 분산 학습

### 병목

- 메모리 용량
- 버스 대역폭

- 32비트 단정밀도: single-precision
- 64비트 배정밀도: double-precision
- 16비트 반정밀도: half-precision

### 활용

- 사물 검출:
  - R-CNN Regions with CNN
    - 후보 영역 추출 + CNN 특징 계산
    - Selective Search 기법
  - Faster R-CNN: 후보 영역 추출까지 CNN으로 처리
- 분할 segmentation
  - FCN Fully Convolutional Network: 합성곱 계층만으로 구성된 네트워크
    - 한번의 순전파로 모든 픽셀의 클래스 분류
    - 공간 볼륨을 유지한 채 마지막 출력까지 처리
- 사진 캡션 생성
  - NIC Neural Image Caption
    - CNN: 이미지 특징 추출
    - 순환신경망 RNN Recurrent Neural Network
      - 텍스트 생성
  - 멀티모달 처리 multimodal processing: 여러 종류의 정보 조합 및 처리

### 미래

- 이미지 화풍 변환: [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- 이미지 생성: DCGAN Deep Convolutional Generative Adversarial Network
  - 논문: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- 자율 학습
  - Deep Belief Network
  - Deep Boltzmann Machine
- 자율 주행
  - 논문: [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561)
- 강화 학습 Reinforcement Learning
  - 논문: Deep Q-Network [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
