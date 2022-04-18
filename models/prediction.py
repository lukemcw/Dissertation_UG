import pandas as pd
import numpy as np
import math
import statsmodels as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.multivariate.pca import PCA

DATA_FILE = r"C:\Users\mcwat\PycharmProjects\Economics\Dissertation_UG\data\transformed_data.csv"


# INDIVIDUAL MODEL FORECAST FUNCTIONS
# AR(2)
def ar2_predict(train_data, hmax=4):
    """
    Function: uses statmodel to predict AR(2)
    Inputs: train_data
    Outputs: predictions (df)
    df = h      prediction
         1        y_t+1
         2        y_t+2
         3        y_t+3
         4        y_t+4
    """
    # Fit to test_data
    ar_model = AutoReg(train_data, lags=[1, 2]).fit()
    # predict
    pred = ar_model.predict(start=len(train_data), end=(len(train_data) + hmax - 1), dynamic=False)
    return pred


# K-NN
def knn(y_train):
    # average 1st 3
    av_3 = np.average([y_train.values[-i] for i in range(3)])
    # average first 5
    av_5 = np.average([y_train.values[-i] for i in range(7)])
    # average first 7
    av_7 = np.average([y_train.values[-i] for i in range(7)])
    # average of these averages = prediction for t+1
    t_plus_1 = np.average([av_3, av_5, av_7])
    return t_plus_1


def knn_predict(y_train):
    """
    Function: KNN prediction
    Inputs: train_data
    Outputs: predictions (df)
    df = h      prediction
         1        y_t+1
         2        y_t+2
         3        y_t+3
         4        y_t+4
    """
    # init list to store forecasts in
    preds = []
    for i in range(5):
        # save pred as pd.Series so it can be appended
        t_plus_1 = pd.Series(knn(y_train))
        # append to preds
        preds.append(t_plus_1.values[0])
        # append to training data
        y_train = y_train.append(t_plus_1)
        # re-index the Series to allow for next estimation
        y_train.reset_index(drop=True)
    return preds


# factor estimation (PCA)
def factor_em(X):
    pc = PCA(X[:], standardize=False, ncomp=8)  # See baing.py for check of ncomp=8
    return pc.factors


def fact_load():
    pc = PCA(X[:], standardize=False, ncomp=8)  # See baing.py for check of ncomp=8
    return pc.loadings


def top_10():
    facts = fact_load()
    top_10 = []
    for index, item in enumerate(fact_load()):
        series = facts.iloc[:, index].sort_values(ascending=False)[0:10]
        top_10.append(series)
    return top_10


def series_to_supervised(data, n_in=1, n_out=2, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    from pandas import DataFrame
    from pandas import concat
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('y%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('y%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('y%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def make_Z(data, factors):
    # drop first two rows from factors,
    factors = factors.drop(index=data.index[0:2])
    y = list(data['GDPC1'].values)
    y = series_to_supervised(y, 2)
    Z = pd.concat([factors, y])
    return Z


# Dynamic factor model
# in: factors, out: predction
def factor_prediction(factors, y, hmax=4):
    #set up Z matrix (y lags and factors)
    # fit with OLS, then predict using recent set of data:
    return pred


# random forest
# in: factors or data, out: prediction
def rf_predict(factors, y):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(Z, y)
    # make prediction
    pass


# main function to run recursive estimation routine for POOS experiment-
def predict_models(data, start=82):
    # initiate a model arrays and a true
    knn_forecasts = []
    ar2_forecasts = []
    factor_forecasts = []
    factor_rf_forecasts = []
    for i in data.index:  # think of i as prediction date
        if i >= start:  # once we hit the forecasting start point
            # select training data, and get a series of test data
            train_data = data[:i].copy(deep=True)
            y_train = train_data["'GDPC1'"]
            X_train = train_data.drop(['date', "'GDPC1'"], axis=1)
            # predict models
            ar2_pred = ar2_predict(y_train).tolist()  # 4 by 1 series/vector of prediction
            knn_pred = knn_predict(y_train)
            # factor_pred =
            # factor_rf_pred =
            # append to list
            ar2_forecasts.append(ar2_pred)
            knn_forecasts.append(knn_pred)
    predictions = {"AR": ar2_forecasts, "KNN": knn_forecasts}
    #
    factors_info = []
    return predictions


############################### evaluation:
# Extract indiviudual series for each model
def individual_series(forecasts):
    # loop through model's predictions and make a series for each forecasting herizon
    h1 = []
    h2 = []
    h3 = []
    h4 = []
    for forecast in forecasts:
        h1.append(forecast[0])
        h2.append(forecast[1])
        h3.append(forecast[3])
        h4.append(forecast[4])
    # find right dates to assign to each Series
    # h1

# Error metrics:


if __name__ == '__main__':
    data = pd.read_csv(DATA_FILE)
    gdp = data["'GDPC1'"]
    dates = data['date']
    X = data.drop(['date', "'GDPC1'"], axis=1)
    # lags = series_to_supervised(gdp.values.tolist())
    # print(lags)
    print(predict_models(data))