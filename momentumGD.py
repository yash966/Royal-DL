#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 19:36:40 2021

@author: yashshah
"""
import numpy as np


X = [0.5,2.5,3.4]
Y = [0.2,0.9,1.2]

def f(w,b,x):
    return 1.0/(1.0 + np.exp(-(w*x + b)))

def error (w,b):
    err = 0.0
    for x,y in zip(X,Y):
        fx = f(w,b,x)
        err += 0.5 * (fx-y) **2
    return err

def grad_b (w,b,x,y):
    fx = f(w,b,x)
    return (fx-y)*fx*(1-fx)

def grad_w (w,b,x,y):
    fx = f(w,b,x)
    return (fx-y)* fx *(1-fx) * x


def do_moumentum_grad():

    w,b,eta,max_epoch = 2,3,1.0,10
    gama = 0.1
    w_pre,b_pre = 0,0
    for i in range(max_epoch):
        dw,db = 0,0
        for (x,y) in zip(X,Y):
            dw += grad_w(w,b,x,y)
            db += grad_b(w,b,x,y)
        w = w - ((gama * w_pre) +(eta * dw))
        b = b - ((gama * b_pre) +(eta * db))
        w_pre = w
        b_pre = b
        print(w,b)
    print("Momentum Gradient Descent : ",error(w,b))
   
 
print("------------------------------------------------------")
do_moumentum_grad()
print("------------------------------------------------------")