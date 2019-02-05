#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 02:29:24 2019

@author: gauravmalik
"""

import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


dates= []
prices = []

def get_data(file):
    with open(file, 'r') as csvFile:
        csvFileReader = csv.reader(csvFile)
        next(csvFileReader)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('/')[0]))
            prices.append(float(row[3]))
    return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    
    svr_lin= SVR(kernel = 'linear', C=1e3)
    svr_poly = SVR(kernel = 'poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma=0.2)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    plt.scatter(dates, prices, color='black', label="Data")
    plt.plot(dates, svr_lin.predict(dates), color='blue', label='SVR Linear')
    plt.plot(dates, svr_poly.predict(dates), color='green', label='SVR Poly')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label="SVR Rbf")
    plt.xlabel('Dates')
    plt.ylabel('Prices')
    plt.title("SVR")
    plt.legend()
    plt.show()
    
    return svr_lin.predict(x)[0], svr_poly.predict(x)[0], svr_rbf.predict(x)[0]

get_data('HistoricalQuotes.csv')
predict_prices(dates, prices, 20)
print(predict_prices)