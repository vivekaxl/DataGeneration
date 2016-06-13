import pandas as pd
import numpy as np
from random import shuffle


def test_linear_regression(filename):
    coeffs = []
    rss = []
    from sklearn import datasets, linear_model
    df = pd.read_csv(filename)
    h_indep = [d for d in df.columns if "A" in d]
    h_dep = [d for d in df.columns if "A" not in d]
    for _ in xrange(10):
        msk = np.random.rand(len(df)) < 0.5
        train_data = df[msk]
        test_data = df[msk]


        train_indep = train_data[h_indep]
        train_dep = train_data[h_dep]

        test_indep = test_data[h_indep]
        test_dep = test_data[h_dep]

        regr = linear_model.LinearRegression()
        regr.fit(train_indep, train_dep)

        coeffs.append(regr.coef_)

        rss.append(np.mean((regr.predict(test_indep) - test_dep) ** 2))
        print np.mean((regr.predict(test_indep) - test_dep) ** 2)

    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    folder_name = "./RData/"
    from os import listdir
    files = sorted([folder_name + file for file in listdir(folder_name)])
    for file in files:
        print file
        file = "./RData/Regression_100_2_25_100.csv"
        test_linear_regression(file)
