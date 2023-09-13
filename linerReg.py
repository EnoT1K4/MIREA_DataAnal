import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')


def linear_prediction(X, w):
    a = 0
    for x in X:
        for W in w:
            a += x * W           
    return a

def loss():
    pass



def gradient_descent(x,w,y,lr=0.001):
    num_features = x.shape[1]
    num_samples = x.shape[0]
    weights = np.random.rand(num_features)

    for epoch in range(w):
        # Step 3: Compute predicted values
        y_pred = np.dot(x, weights)

        # Step 4: Calculate error
        error = y_pred - y

        # Step 5: Compute gradients
        gradients = 2 * np.dot(x.T, error) / num_samples

        # Step 6: Update weights
        weights -= lr * gradients

    return weights