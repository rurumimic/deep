# deep

밑바닥부터 시작하는 딥러닝 1, 2를 읽고 내용을 정리

## 내용

### Perceptron

- Content: [Perceptron](perceptron/perceptron.md)
- Code: [perceptron.py](perceptron/perceptron.py)
- Example: [Notebook](perceptron/example.ipynb)


### Neural Network

- Code: [activation.py](neural_network/activation.py)
- Example: [Notebook](neural_network/example.ipynb)

### Neural Network Learn

- Code: [loss.py](neural_network_learn/loss.py), [differentiation.py](neural_network_learn/differentiation.py)
- Example: [Notebook](neural_network_learn/example.ipynb)

### Backpropagation

- Code: [AddLayer.py](backpropagation/AddLayer.py), [MulLayer.py](backpropagation/MulLayer.py), [Relu.py](backpropagation/Relu.py), [Sigmoid.py](backpropagation/Sigmoid.py), [Affine.py](backpropagation/Affine.py), [SoftmaxWithLoss.py](backpropagation/SoftmaxWithLoss.py)
- Example: [Notebook](backpropagation/example.ipynb)


### Backpropagation

- Content: [Backpropagation](backpropagation/backpropagation.md)
- Code: [backpropagation.py](backpropagation/backpropagation.py)
- Example(Working in Progress): [Notebook](backpropagation/example.ipynb)

---

## 환경 설정

### Python 3.8

```bash
conda create -n deep python=3.8
conda activate deep
```

### Libraries

```bash
conda install -c anaconda numpy
```

### Jupyter Lab

```bash
conda install -c conda-forge jupyterlab
jupyter lab
```

---

## 참고

- Japan: [Deep Learning from Scratch](https://github.com/oreilly-japan/deep-learning-from-scratch)
- Korea: [밑바닥부터 시작하는 딥러닝](https://github.com/WegraLee/deep-learning-from-scratch)
- Blog: 
  - [밑바닥부터 시작하는 딥러닝 정리 - 1](https://velog.io/@dscwinterstudy/series/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D)
  - [밑바닥부터 시작하는 딥러닝 정리 - 2](https://excelsior-cjh.tistory.com/169)
  
### 강의

Stanford: [CS231N](http://cs231n.stanford.edu/)
