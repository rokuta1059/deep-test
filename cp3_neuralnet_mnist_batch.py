import sys, os
import numpy as np
import pickle
from dataset import load_mnist
from common.functions import sigmoid, softmax

""" 
    배치: 하나로 묶은 데이터
    수치 계산 라이브러리 대부분은 큰 배열을 효율적으로 처리할 수 있도록 고도로 최적화 되어있음
    커다란 신경망은 데이터 전송이 병목이 되는 경우가 자주 있는데, 
    배치 처리를 통해 버스에 주는 부하를 줄임

    배치 처리를 위한 배열들의 형상 추이
    X           W1         W2         W3      -> Y
    100 X 784 > 784 x 50 > 50 X 100 > 100 X 10 > 100 X 10
"""

local = "D:\\Git_Workspace\\deep-test\\sample_weight.pkl"

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open(local, 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):  # 0에서부터 len(x)까지 batch_size 간격으로 증가하는 리스트
    x_batch = x[i:i+batch_size]         # 입력 데이터 묶기
    y_batch = predict(network, x_batch) # 배치 단위로 묶고 분류
    p = np.argmax(y_batch, axis=1)      # argmax()는 최대값의 인덱스를 가져옴
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
