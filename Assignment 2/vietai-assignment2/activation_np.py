"""activation_np.py
This file provides activation functions for the NN 
Author: Phuong Hoang
"""

import numpy as np


def sigmoid(x):
    """sigmoid
    TODO: 
    Sigmoid function. Output = 1 / (1 + exp(-1)).
    :param x: input
    """
    #[TODO 1.1]
    return 1/(1+np.exp(-x))


def sigmoid_grad(a):
    """sigmoid_grad
    TODO:
    Compute gradient of sigmoid with respect to input. ouput = 1/ (1 + exp(-input))
    :param a: output of the sigmoid function
    """
    #[TODO 1.1]
    return a*(1-a)


def reLU(x):
    """reLU
    TODO:
    Rectified linear unit function. Output = max(0,x).
    :param x: input
    """
    #[TODO 1.1]
    return np.maximum(0,x)


def reLU_grad(a):
    """reLU_grad
    TODO:
    Compute gradient of ReLU with respect to input
    :param a: output of ReLU
    """
    #[TODO 1.1]
    grad=np.zeros_like(a)
    for i in range (a.shape[0]):
        for j in range (a.shape[1]):
            if a[i][j]>0:  grad[i][j]=1
            else: grad[i][j]=0
    return grad


def tanh(x):
    """tanh
    TODO:
    Tanh function.
    :param x: input
    """
    #[TODO 1.1]
    return np.tanh(x)


def tanh_grad(a):
    """tanh_grad
    TODO:
    Compute gradient for tanh w.r.t input
    :param a: output of tanh
    """
    #[TODO 1.1]
    return 1-(np.power(a,2))


def softmax(x):
    """softmax
    TODO:
    Softmax function.
    :param x: input
    """
    e=np.exp(-x)
    sum_e=np.sum(e,axis=0)
    return e/sum_e


def softmax_minus_max(x):
    """softmax_minus_max
    TODO:
    Stable softmax function.
    :param x: input
    """

    e=np.exp(-x-np.max(x,axis=0,keepdims=True))
    sum_e=np.sum(e,axis=0)
    return e/sum_e
