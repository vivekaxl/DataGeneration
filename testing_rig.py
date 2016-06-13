import pandas as pd
import numpy as np
import sys, time


def test_linear_regression(filename):
    start_time = time.time()
    coeffs = []
    rss = []
    from sklearn import datasets, linear_model
    df = pd.read_csv(filename)
    h_indep = [d for d in df.columns if "A" in d]
    h_dep = [d for d in df.columns if "A" not in d]
    for _ in xrange(10):
        print "- ",
        sys.stdout.flush()
        msk = np.random.rand(len(df)) < 0.5
        train_data = df[msk]
        test_data = df[msk]

        assert(len(train_data) == len(test_data)), "Something is wrong"

        train_indep = train_data[h_indep]
        train_dep = train_data[h_dep]

        test_indep = test_data[h_indep]
        test_dep = test_data[h_dep]

        regr = linear_model.LinearRegression()
        regr.fit(train_indep, train_dep)

        coeffs.append(regr.coef_)

        rss.append(np.mean((regr.predict(test_indep) - test_dep) ** 2))

    extract_name = filename.split("/")[-1].split(".")[0] + ".p"
    import pickle
    pickle.dump(coeffs, open("./Results_Linear_Regression/coeffs_" + extract_name, "wb"))
    pickle.dump(rss, open("./Results_Linear_Regression/rss_" + extract_name, "wb"))
    print
    print "Total Time: ", time.time() - start_time


def _test_linear_regression():
    folder_name = "./RData/"
    from os import listdir
    files = sorted([folder_name + file for file in listdir(folder_name)])
    for file in files:
        print file
        test_linear_regression(file)


def test_logistic_regression(filename):
    start_time = time.time()
    coeffs = []
    acc = []
    from sklearn import datasets, linear_model
    df = pd.read_csv(filename)
    h_indep = [d for d in df.columns if "A" in d]
    h_dep = [d for d in df.columns if "A" not in d]
    for _ in xrange(10):
        print "- ",
        sys.stdout.flush()
        msk = np.random.rand(len(df)) < 0.5
        train_data = df[msk]
        test_data = df[msk]

        assert (len(train_data) == len(test_data)), "Something is wrong"

        train_indep = train_data[h_indep]
        train_dep = train_data[h_dep]

        test_indep = test_data[h_indep]
        test_dep = test_data[h_dep]
        logistic = linear_model.LogisticRegression()
        logistic.fit(train_indep, train_dep)

        import pdb
        pdb.set_trace()
        acc.append(logistic.score(X=test_indep, y=[i[-1] for i in test_dep.values.tolist()]))

    from numpy import mean
    print mean(acc)

if __name__ == "__main__":
    folder_name = "./CData/"
    from os import listdir
    files = sorted([folder_name + file for file in listdir(folder_name)])
    for file in files:
        print file
        file = "./CData/Classification_10000_32_2_50.0.csv"
        test_logistic_regression(file)