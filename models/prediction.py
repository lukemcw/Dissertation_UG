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

    # import packages for model
    from statsmodels.tsa.ar_model import AutoReg
    # fit to test_data
    ar_model = AutoReg(train_data, lags=2).fit()
    # predict
    pred = ar_model.predict(start=len(train_data), end=(len(train_data)+hmax), dynamic=False)
    #remove the first item in series - it includes the initial value (not a predicted value)
    return pred[1:]

# K-NN

# factor estimation (PCA)
# use baing function from FRED library
# use pre-made pca function as well to get factor estimates


# Dynamic factor model
# in: factors, out: predction


# random forest
# in: factors or data, out: prediction
def rf_forecast():
    pass

# COMBINE PREDICTIONS INTO SINGLE DF: