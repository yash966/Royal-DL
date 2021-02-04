#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:27:24 2021

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

def do_nesterov_gradient():

    w,b,eta,max_epoch = 2,3,1,50
    gama = 0.1
    w_prev,b_prev = 0,0
    
    for i in range(max_epoch):
        dw,db=0,0
        v_w=gama*w_prev  
        v_b=gama*b_prev  
        for x,y in zip(X, Y):
            dw += grad_w(w-v_w,b-v_b,x, y)
            db += grad_b(w-v_w,b-v_b,x, y)
        
        v_w=gama*w_prev+eta*dw
        v_b=gama*b_prev+eta*db
        w=w-v_w
        b=b- v_b
        w_prev=v_w
        b_prev=v_b
        print(w,b)
        
    print("Nesterov Gradient Descent : ",error(w,b))


do_nesterov_gradient()