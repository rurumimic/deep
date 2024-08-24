import os, sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from collections import OrderedDict
# from common.layers import *
from common.gradient import numerical_gradient


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    # 순전파    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    # 역전파
    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x
        
        return dx, dy

        
class AddLayer:
    def __init__(self):
        pass
    
    # 순전파
    def forward(self, x, y):
        out = x + y
        return out
    
    # 역전파
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu:
    def __init__(self):
        self.mask = None
    
    # 순전파
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    # 역전파
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    # 순전파    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        return out
    
    # 역전파
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    # 순전파    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    # 역전파
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(one-hot)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx


class TwoLayerNet:
    '''2층 신경망 구현'''
    def __init__(self, input_size, 
                 hidden_size, output_size, weight_init_std=0.01):
        '''
        초기화 수행
        Params:
            - input_size: 입력층 뉴런 수
            - hidden_size: 은닉층 뉴런 수
            - output_size: 출력층 뉴런 수
            - weight_init_std: 가중치 초기화 시 정규분포의 스케일
        '''
        # 가중치 초기화
        self.params = {
            'W1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }
        
        # 계층 생성
        self.layers = OrderedDict({
            'Affine1': Affine(self.params['W1'], self.params['b1']),
            'Relu1': Relu(),
            'Affine2': Affine(self.params['W2'], self.params['b2'])
        })
        
        self.last_layer = SoftmaxWithLoss()
        
    
    def predict(self, x):
        '''예측(추론)
            Pararms:
                - x: 이미지 데이터'''
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
    
    def loss(self, x, t):
        '''
        손실함수의 값을 계산
        Params:
            - x: 이미지데이터, t: 정답 레이블
        '''
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t):
        '''
        정확도 계산
        Params:
            - x: 이미지 데이터
            - t: 정답 레이블
        '''
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy
    
    
    def numerical_gradient(self, x, t):
        '''
        미분을 통한 가중치 매개변수의 기울기 계산
        Params:
            - x: 이미지 데이터
            - t: 정답 레이블 
        '''
        loss_W = lambda W: self.loss(x, t)
        
        grads = {
            'W1': numerical_gradient(loss_W, self.params['W1']),
            'b1': numerical_gradient(loss_W, self.params['b1']),
            'W2': numerical_gradient(loss_W, self.params['W2']),
            'b2': numerical_gradient(loss_W, self.params['b2'])
        }
        return grads
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        # 결과 저장
        grads = {
            'W1': self.layers['Affine1'].dW, 'b1': self.layers['Affine1'].db,
            'W2': self.layers['Affine2'].dW, 'b2': self.layers['Affine2'].db
        }
        return grads