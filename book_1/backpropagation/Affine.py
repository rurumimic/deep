import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        # 4차원 텐서 데이터 고려
        self.x = None
        self.original_x_shape = None
        
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None
        
    # 4차원 텐서 데이터 고려
    def flat(self, x):
        self.original_x_shape = x.shape # 텐서 형태 기록: (a, b, c, d ...)
        return x.reshape(x.shape[0], -1) # x의 행 개수만큼 전체 원소 나누기
    
    def restore(self, dx):
        # 입력 데이터 모양 변경
        return dx.reshape(*self.original_x_shape)

    def forward(self, x):
        x = self.flat(x) # 데이터 차원 변경
        
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = self.restore(dx) # 데이터 차원 변경
        return dx
