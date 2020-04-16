from variables import*
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_data():
    df = pd.read_csv(csv_path)
    closing_price = df['close'].values
    scalar, X, Y = create_data(closing_price)
    Ntrain  = len(closing_price) // 2
    Xtrain, Xtest = X[:Ntrain], X[Ntrain:]
    Ytrain, Ytest = Y[:Ntrain], Y[Ntrain:]
    return scalar, Xtrain, Ytrain, Xtest, Ytest

def create_data(closing_price):
    Ntrain  = len(closing_price) // 2
    scalar = StandardScaler()
    closing_price = closing_price.reshape(-1,1)
    scalar.fit(closing_price[:Ntrain])
    closing_price = scalar.transform(closing_price)

    N = len(closing_price)
    X = []
    Y = []
    for i in range(N-T):
        x = closing_price[i:T+i]
        y = closing_price[T+i]
        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    X = X.reshape(-1,T,1)
    print("Input  shape :",X.shape)
    print("Output shape :",Y.shape)

    return scalar, X, Y