# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 03:18:22 2018


@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python. 
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper, 
   please contact the authors of related paper.
"""
import numpy as np
from sklearn import preprocessing
from numpy import random
import time
from sklearn import metrics
import math
from sklearn.metrics import mean_squared_error

def tansig(x):
    return (2/(1+np.exp(-2*x)))-1

def pinv(A,reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)

def relu(data):
    return np.maximum(data, 0)

def pidrelu(data):
    return 0.05 + 0.6 * np.maximum(data, 0) + 0.4 * 0.5 * np.maximum(data, 0) * np.maximum(data, 0)

def pid(data):
    return tansig(data) + np.log(np.exp(data) + 1) - data + 4 * np.exp(2*data) / (np.exp(2 * data) +1 ) ** 2

# def mape(y_true, y_pred):
#     return np.mean(np.abs((y_pred - y_true) / y_true+1)) * 100

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / (y_true+1))) * 100


'''
参数压缩
'''
def shrinkage(a,b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z
'''
参数稀疏化
'''
def sparse_bls(A,b):
    lam = 0.001
    itrs = 50
    AA = np.dot(A.T,A)   
    m = A.shape[1]
    n = b.shape[1]
    wk = np.zeros([m,n],dtype = 'double')
    ok = np.zeros([m,n],dtype = 'double')
    uk = np.zeros([m,n],dtype = 'double')
    L1 = np.mat(AA + np.eye(m)).I
    L2 = np.dot(np.dot(L1,A.T),b)
    for i in range(itrs):
        tempc = ok - uk
        ck =  L2 + np.dot(L1,tempc)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
        wk = ok
    return wk

def bls_regression(train_x,train_y,test_x,test_y,s,C,NumFea,NumWin,NumEnhan):
    NumFea = int(NumFea)
    NumWin = int(NumWin)
    NumEnhan = int(NumEnhan)
    u = 0
    WF = list()
    for i in range(NumWin):
        random.seed(i+u)
        WeightFea=2*random.randn(train_x.shape[1]+1,NumFea)-1;
        WF.append(WeightFea)
    random.seed(100)
    WeightEnhan=2*random.randn(NumWin*NumFea+1,NumEnhan)-1;
    time_start = time.time()
    H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0],1])]);
    y = np.zeros([train_x.shape[0],NumWin*NumFea])
    WFSparse = list()
    distOfMaxAndMin = np.zeros(NumWin)
    meanOfEachWindow = np.zeros(NumWin)
    for i in range(NumWin):
        WeightFea = WF[i]
        A1 = H1.dot(WeightFea)        
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
        A1 = scaler1.transform(A1)
        WeightFeaSparse  = sparse_bls(A1,H1).T
        WFSparse.append(WeightFeaSparse)
    
        T1 = H1.dot(WeightFeaSparse)
        meanOfEachWindow[i] = T1.mean()
        distOfMaxAndMin[i] = T1.max() - T1.min()
        T1 = (T1 - meanOfEachWindow[i])/distOfMaxAndMin[i] 
        y[:,NumFea*i:NumFea*(i+1)] = T1

    H2 = np.hstack([y,0.1 * np.ones([y.shape[0],1])])
    T2 = H2.dot(WeightEnhan)
    T2 = relu(T2);
    T3 = np.hstack([y,T2])
    WeightTop = pinv(T3,C).dot(train_y)

    Training_time = time.time()- time_start
    # print('Training has been finished!');
    # print('The Total Training Time is : ', round(Training_time,6), ' seconds' )
    NetoutTrain = T3.dot(WeightTop)

    RR2 = metrics.r2_score(train_y, NetoutTrain)
    # RMSE = np.mean(np.sqrt((NetoutTrain-train_y).T*(NetoutTrain-train_y)/train_y.shape[0]))
    RMSE = np.sqrt(mean_squared_error(NetoutTrain,train_y))
    # MAPE = sum(abs(NetoutTrain-train_y))/train_y.mean()/train_y.shape[0]
    MAPE = mape(train_y,NetoutTrain)
    train_ERR = RMSE
    train_MAPE = MAPE
    train_RR2 = RR2
    # print('Training RR2 is : ', train_RR2);
    # print('Training RMSE is : ',RMSE);
    # print('Training MAPE is : ', MAPE)
    time_start = time.time()
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0],1])])
    yy1=np.zeros([test_x.shape[0],NumWin*NumFea])
    for i in range(NumWin):
        WeightFeaSparse = WFSparse[i]
        TT1 = HH1.dot(WeightFeaSparse)
        TT1  = (TT1 - meanOfEachWindow[i])/distOfMaxAndMin[i]   
        yy1[:,NumFea*i:NumFea*(i+1)] = TT1

    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0],1])])
    TT2 = relu(HH2.dot( WeightEnhan));
    TT3 = np.hstack([yy1,TT2])
    NetoutTest = TT3.dot(WeightTop)
    RR2 = metrics.r2_score(NetoutTest, test_y)
    RMSE = np.mean(np.sqrt((NetoutTest-test_y).T*(NetoutTest-test_y)/test_y.shape[0]))
    # RMSE = np.sqrt(mean_squared_error(NetoutTest,test_y))
    # MAPE = sum(abs(NetoutTest-test_y))/test_y.mean()/test_y.shape[0]
    MAPE = mape(test_y,NetoutTest)
    test_ERR = RMSE
    test_MAPE = MAPE
    test_RR2 = RR2
#%% Calculate the testing accuracy
    Testing_time = time.time() - time_start
    # print('Testing has been finished!');
    # print('The Total Testing Time is : ', round(Testing_time,6), ' seconds' );
    # print('Testing RR2 is : ', test_RR2)
    # print('Testing RMSE is : ', RMSE)
    # print('Testing MAPE is : ', MAPE)
    return test_RR2,test_ERR,test_MAPE,train_RR2,train_ERR,train_MAPE

