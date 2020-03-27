import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from variables import csv_path

def plot_data(Xs, Ys, log_y=False, show_fig=True):
    plt.scatter(Xs, Ys, color='red')
    title_ = 'moore law' if not log_y else 'moore law logrithm'
    img = 'moore law.png' if not log_y else 'moore law logrithm.png'

    plt.title(title_)
    plt.ylabel('transistor count')
    plt.xlabel('year')
    plt.savefig(img)
    plt.show(img)

def get_regression_data():
    df = pd.read_csv(csv_path, header=None)
    Xs = df.iloc[:,0].to_numpy().reshape(-1,1)

    Ys = df.iloc[:,1].to_numpy()

    plot_data(Xs, Ys)
    Ys = np.log(Ys)
    plot_data(Xs, Ys,True)

    Xs = Xs - Xs.mean()
    return Xs,Ys