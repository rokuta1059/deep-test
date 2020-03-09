# coding: utf-8
import sys, os
import numpy as np
import pickle
from dataset import load_mnist
from common import sigmoid, softmax

local = "D:\\Git_Workspace\\deep-test\\sample_weight.pkl"

"""
    3.6.2 신경망 추론 처리
    입력층 뉴런: 784개
    출력층 뉴런: 10개 - 문제가 0에서 9까지의 숫자를 구분하는 문제이므로

    은닉층: 총 2개 (뉴런 수 등은 임의)
    1번째 은닉층: 50개 뉴런
    2번쨰 은닉층: 100개 뉴런

    신경망 각 층의 배열 형상의 추이
    X     W1         W2         W3         Y
    784 > 784 x 50 > 50 X 100 > 100 X 10 > 10
"""

def get_data():
    #(훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_train, t_train, x_test, t_test


def init_network():
    """sample_weight.pkl에 저장된 '학습된 가중치 매개변수'를 읽는다
    """
    with open(local, 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x_train, t_train, x_test, t_test = get_data()
network = init_network()

# 훈련 이미지와 레이블에 대한 내용
accuracy_cnt = 0
for i in range(len(x_train)):           # 데이터를 한 장씩 읽는다
    y = predict(network, x_train[i])    # 한 장씩 분류
    p= np.argmax(y)                     # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t_train[i]:
        accuracy_cnt += 1

# 정확도 출력
print("Accuracy for train:\t" + str(float(accuracy_cnt) / len(x_train)))

# 시험 이미지와 레이블에 대한 내용
accuracy_cnt = 0
for i in range(len(x_test)):           # 데이터를 한 장씩 읽는다
    y = predict(network, x_test[i])    # 한 장씩 분류
    p= np.argmax(y)                     # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t_test[i]:
        accuracy_cnt += 1

# 정확도 출력
print("Accuracy for test:\t" + str(float(accuracy_cnt) / len(x_test)))