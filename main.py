import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import *


def read_data():

    return pd.read_csv('dataset.txt')


def plot():
    data = read_data()

    dates = np.array(data['Dates'])

    x_close = np.array(data['x_close'])
    y_close = np.array(data['y_close'])
    spread_close = x_close - y_close

    x_close_pct = np.array(data['x_close'].pct_change().fillna(0))
    y_close_pct = np.array(data['y_close'].pct_change())
    spread_pct = np.array((data['x_close'] - data['y_close']).pct_change().fillna(0))

    # plt.plot(x_close_pct, 'r')
    # plt.plot(y_close_pct, 'b')

    adf_result = adfuller(x_close_pct)

    # ADF Test analysis

    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print('\t%s: %.3f' % (key, value))

    plt.plot(spread_pct)
    plt.show()

if __name__ == '__main__':
    plot()