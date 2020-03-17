# 미니 배치
# 데이터의 일부를 추려 전체의 근사치로 이용한다.
# 신경망 학습에도 훈련 데이터로부터 일부만 골라 학습을 수행하며,
# 이 일부를 미니 배치라고 한다.
# 일부를 무작이로 뽑아 학습하는 방법을 미니 배치 학습이라고 한다.

import sys, os
import numpy as np
from dataset import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(t_train.shape)

# x_train의 차원 수를 확인한다. 
train_size = x_train.shape[0]
batch_size = 10
# train_size 미만의 수 중에서 batch_size 만큼 뽑아낸다
batch_mask = np.random.choice(train_size, batch_size)   
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]