import sys, os
import numpy as np
import matplotlib.pylab as plt
from dataset import load_mnist
from common import sigmoid, step_function, relu, identity_function

def chap3_2_3():
    """3.2.3 계단 함수의 그래프
    계단 함수를 그래프로 그려봅니다.
    """
    x = np.arange(-5.0, 5.0, 0.1) # -5.0에서 5.0 전까지 0.1 간격의 numpy 배열 생성
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # y축의 범위 지정
    plt.show()

def chap3_2_4():
    """3.2.4 시그모이드 함수 구현하기
    시그모이드 함수를 그래프로 그려봅니다.
    """
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # y축의 범위 지정
    plt.show()

def chap3_2_7():
    """3.2.7 ReLU 함수
    ReLU 함수를 그래프로 그려봅니다.
    """
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-1, 6) # y축의 범위 지정
    plt.show()

def init_network():
    """가중치와 편향을 초기화하고 이들을 딕셔너리 변수인 network에 저장
    network에는 각 층에 필요한 매개변수(가중치와 편향)을 저장"""
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    """입력 신호를 출력으로 변환하는 처리과정을 구현
    함수 이름이 forward인 것은 신호가 순방향으로 전달됨을 알리기 위해"""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

def chap3_4():
    """3.4 3층 신경망 구현하기
    입력부터 출력까지의 처리(순방향 처리) 구현
    """
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)

#chap3_2_3()
#chap3_2_4()
#chap3_2_7()
#chap3_4()