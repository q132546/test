import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import BayesianRidge


class Model():
    def __init__(self):
        self.data_pd = pd.read_csv('dataset.txt')

        self.dates = np.array(self.data_pd['Dates'])
        self.x_close = np.array(self.data_pd['x_close'])
        self.y_close = np.array(self.data_pd['y_close'])
        self.spread = self.x_close - self.y_close

    def plot(self):
        # check for extreme spread change

        last_spread = self.spread[0]

        for i in range(len(self.spread)):
            if abs(self.spread[i] - last_spread) > 1:
                print('last ', self.dates[i - 1], 'now ', self.dates[i])

            last_spread = self.spread[i]

        plt.subplot(2, 1, 1)
        plt.plot(self.x_close, 'r')
        plt.plot(self.y_close, 'b')

        plt.subplot(2, 1, 2)
        plt.plot(np.diff(self.spread))
        plt.show()

    def logistic_regression(self):
        # factors construction
        split_num = 3000

        training_x = self.x_close[:-split_num]
        training_y = self.y_close[:-split_num]
        training_spread = training_x - training_y

        testing_x = self.x_close[-split_num:]
        testing_y = self.y_close[-split_num:]
        testing_spread = testing_x - testing_y

        # training set

        x_price_diff = np.diff(training_x)
        x_price_diff = x_price_diff.reshape(len(x_price_diff), 1)

        spread_mean_3 = np.array(pd.DataFrame(training_spread).rolling(3).mean().dropna())
        spread_mean_10 = np.array(pd.DataFrame(training_spread).rolling(10).mean().dropna())

        min_len = min(len(x_price_diff), len(spread_mean_3), len(spread_mean_10))

        training_array = np.hstack((x_price_diff[-min_len:], spread_mean_3[-min_len:]))
        training_array = np.hstack((training_array, spread_mean_10[-min_len:]))

        training_array = training_array[:-1]
        target_array = training_spread[-min_len:][1:]
        target_array = target_array.reshape(len(target_array), 1)

        # testing set

        x_price_diff_test = np.diff(testing_x)
        x_price_diff_test = x_price_diff_test.reshape(len(x_price_diff_test), 1)

        spread_mean_3_test = np.array(pd.DataFrame(testing_spread).rolling(3).mean().dropna())
        spread_mean_10_test = np.array(pd.DataFrame(testing_spread).rolling(10).mean().dropna())

        min_len = min(len(x_price_diff_test), len(spread_mean_3_test), len(spread_mean_10_test))

        testing_array = np.hstack((x_price_diff_test[-min_len:], spread_mean_3_test[-min_len:]))
        testing_array = np.hstack((testing_array, spread_mean_10_test[-min_len:]))

        testing_array = testing_array[:-1]
        target_test_array = testing_spread[-min_len:][1:]
        target_test_array = target_test_array.reshape(len(target_test_array), 1)

        # model training

        clf = BayesianRidge().fit(training_array, target_array.ravel())

        for i in range(len(testing_array)):

            print(clf.predict(testing_array[i].reshape(1, -1)), target_test_array[i])


        # clf = LogisticRegression(random_state=0).fit(X, y)


if __name__ == '__main__':
    model = Model()
    model.logistic_regression()
