#!/usr/bin/env python
# coding: utf-8

import sys
import os

clear = lambda: os.system('cls')

clear()  # 지우고 싶을 때 입력

### MNIST로 추론하는 신경망 구현하기
# 입력층 784개, 출력층 10개
# 은닉층 2개, 첫번째는 50개의 뉴런 이용, 두번째는 100개의 뉴런을 배치 (임의로 정한 값)


print(sys.path)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


# MNIST_SW data 파일 가져오기 22x22로 resizing 되어있으며, 0번째 열은 label로 저장됨.
# readlines() 메서드는 파일의 내용을 한줄씩 불러와서 문자열 리스트로 반환하는 함수
# 그림을 제대로 보기 위해서는 좌우반전 후 90회전을 해야 함.


import csv

# data_file = pd.read_csv('./data/train_resized14x14_v2.csv', delimiter=',')
data_file = pd.read_csv('data/train_and_test_data/1010_v2.csv', delimiter=',')
print(type(data_file))
print(data_file.shape)
data_file = np.array(data_file)
print(data_file.shape)

test_data_file = pd.read_csv('data/train_and_test_data/1010t_v2.csv', delimiter=',')
print(type(test_data_file))
print(test_data_file.shape)
test_data_file = np.array(test_data_file)
print(test_data_file.shape)

# t_train : train set의 라벨, x_train : train set 이미지
# t_test : test set의 라벨, x_test: test set 이미지

t_train, x_train = tf.one_hot(data_file[:, 0], depth=10), data_file[:, 1:] / 255
print(t_train.shape, x_train.shape)
t_train = t_train.numpy()

t_test, x_test = tf.one_hot(test_data_file[:, 0], depth=10), test_data_file[:, 1:] / 255
t_test = t_test.numpy()
print('t_test:', type(t_test))
print(t_test.shape, x_test.shape)

print(t_train)




# mnist_utils/layers.py




# two_layer_net.py

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

from collections import OrderedDict
from mnist_utils.layers import *
from mnist_utils.gradient import numerical_gradient


class LayerNet:
    '''2층 신경망 구현'''

    def __init__(self, input_size, output_size, weight_init_std=0.01):
        '''
        초기화 수행
        Params:
            - input_size: 입력층 뉴런 수

            - output_size: 출력층 뉴런 수
            - weight_init_std: 가중치 초기화 시 정규분포의 스케일
        '''
        # 가중치 초기화
        self.params = {
            'W1': weight_init_std * np.random.randn(input_size, output_size),
            'b1': np.zeros(output_size)

        }

        # 계층 생성
        self.layers = OrderedDict({
            'Affine1': Affine(self.params['W1'], self.params['b1']),
            'Relu1': Relu(),

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

        accuracy = np.sum(y == t) / float(x.shape[0])
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
            'W1': self.layers['Affine1'].dW, 'b1': self.layers['Affine1'].db

        }
        return grads


class SGD:
    def __init__(self, lr=0.6):
        self.lr = lr

    def update(self, params, grad):
        for key in params.keys():
            params[key] -= self.lr * grad[key]


class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.01, beta1=0.8, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


class Momentum:
    """모멘텀 SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# 네트워크 사이즈. 2개의 layer이용, 484*40, 40*10. 즉, 은닉층의 개수는 1개.

network = LayerNet(input_size=x_test.shape[1], output_size=10)
# optimizer = Momentum()
optimizer = Adam()
iters_num = 10000  # 반복횟수
train_size = x_train.shape[0]  # 60,000
batch_size = 500  # 미니배치 크기
# batch_size = 300  # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭 당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)  # 60000 / 60000 = 1
# epoch iters_num / iter_per_epoch

for i in range(iters_num):

    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 오차역전파법으로 기울기 구하기
    grad = network.gradient(x_batch, t_batch)

    params = network.params
    # 매개변수 갱신
    # for key in('W1','b1'):
    #    network.params[key] -=learning_rate*grad[key]

    optimizer.update(params, grad)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

print('done!')

epoch = round(iters_num / iter_per_epoch)
print(epoch)
epoch_ = np.arange(1, epoch)
plt.figure()
plt.plot(train_acc_list, label='train_acc')
plt.plot(test_acc_list, label='test_acc')
plt.legend()
plt.ylim(0, 1.0)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

W1 = network.params['W1']
W = []
W = [W1]

b1 = network.params['b1']
b = []
b = [b1]
print(train_acc_list)

m = np.shape(W1)[0]
n = np.shape(W1)[1]
'''
import csv
w = open(r'path/'+'10x10weight.csv','w')
for i in range(m):
    for j in range(n):
        weight = '%f'%(W1[i][j])
        w.write(weight + "  ")
    w.write('\n')
w.close()

import csv
b = open(r'path/'+'10x10bias.csv','w')
for j in range(n):
    bias = '%f'%(b1[j])
    b.write(bias + "  ")
b.write('\n')
b.close()
'''






