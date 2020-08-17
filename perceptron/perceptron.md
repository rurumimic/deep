# 퍼셉트론

신경망(딥러닝)의 기원이 되는 알고리즘   
퍼셉트론의 구조를 배우는 것은 신경망과 딥러닝으로 나아가는데 중요한 아이디어를 배우는 일   

## 퍼셉트론이란?

다수의 신호를 입력으로 받아 하나의 신호를 출력   
신호도 흐름을 만들고 정보를 앞으로 전달   
퍼셉트론의 결과 값 : `1(신호 흐름) or 0(신호 흐르지 않음)`   

![](./images/percepron_1.png)

**입력이 2개인 Perceptron**

X1, X2 : 입력신호   
Y : 출력신호   
W1, W2 :  가중치   
원 : 뉴런 or 노드   

뉴런에서 보내온 신호의 총합이 정해진 한계를 넘어설 때만 1을 출력 = `뉴런이 활성화 됨`   
퍼셉트론은 복수의 입력신호 각각에 고유한 `가중치`를 부여   
가중치는 각 신호가 결과에 주는 `영향력을 조절하는 요소`로 작용   
가중치가 `클수록` 해당 신호가 그만큼 `더 중요`   

---

## 단순 논리 회로 및 구현

퍼셉트론의 구조는 AND, NAND, OR 게이트 모두에서 같음   
다른 것은 `매개변수(가중치, 임계값)`의 값   
퍼셉트론의 매개변수의 값만 적절히 조정하여 `AND, NAND, OR`로 구현

### AND Gate

입력이 둘이고 출력은 하나   
두 입력이 모두 1일 때만 1을 출력, 나머지는 0 출력   

![](./images/percepron_2.png)

퍼셉트론(W1, W2, THETA) : (0.5, 0.5, 0.7), (0.5, 0.5, 0.8), (1.0, 1.0, 1.0)   

```python
def AND(x1, x2):
    x=np.array([x1, x2])
    w=np.array([0.5, 0.5])
    b=-0.7
    tmp=np.sum(w*x)+b
    if tmp<=0:
        return 0
    else:
        return 1
```

### NAND Gate

NAND : Not AND   
AND의 출력을 뒤집은 것   
모두 1일때만 0, 나머지는 0 출력   

![](./images/percepron_3.png)

퍼셉트론(W1, W2, THETA) : (-0.5, -0.5, -0.7)   

```python
def NAND(x1, x2):
    x=np.array([x1, x2])
    w=np.array([-0.5, -0.5])
    b=0.7
    tmp=np.sum(w*x)+b
    if tmp<=0:
        return 0
    else:
        return 1
```

### OR Gate

입력 신호 중 하나 이상이 1이면 출력이 1이 되는 논리 회로

![](./images/percepron_4.png)

퍼셉트론(W1, W2, THETA) : (0.5, 0.5, -0.2)   

```python
def OR(x1, x2):
    x=np.array([x1, x2])
    w=np.array([0.5, 0.5])
    b=-0.2
    tmp=np.sum(w*x)+b
    if tmp<=0:
        return 0
    else:
        return 1
```

---

## 퍼셉트론의 한계

### XOR Gate 구현 시도

XOR : 배타적 논리합   

![](./images/percepron_5.png)

퍼셉트론(W1, W2, THETA) : (-0.5, 1.0, 1.0)   

![](./images/percepron_6.png)   

시각화로 표현 시, `선형`으로는 두 영역으로 나눌 수 없음   

![](./images/percepron_7.png)   

이러한 `비선형 그래프`로 구분 가능

---

## 다층 퍼셉트론

`다층 퍼센트론(multi-layer perceptron)` 으로 XOR Gate를 구현가능   

### 기존 Gate 조합으로 NAND 만들기

![](./images/percepron_8.png)   

기호와 진리표로 표시하면 다음과 같다   

![](./images/percepron_9.png)   

```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```

참고 : NAND Gate 만으로 컴퓨터를 만들 수 있음

---

## 정리

- 퍼셉트론은 입출력을 갖춘 알고리즘이다. 입력을 주면 정해진 규칙에 따른 값을 출력한다
- 퍼셉트론에서는 ‘가중치’와 ‘편향’을 매개변수로 설정한다
- 퍼셉트론으로 AND, OR 게이트 등의 논리 회로를 표현할 수 있다
- XOR 게이트는 단층 퍼셉트론으로는 표현할 수 없다
- 2층 퍼셉트론을 이용하면 XOR 게이트를 표현할 수 있다
- 단층 퍼셉트론은 직선형 영역만 표현할 수 있고, 다층 퍼셉트론은 비선형 영역도 표현할 수 있다
- 다층 퍼셉트론은 (이론상) 컴퓨터를 표현할 수 있다