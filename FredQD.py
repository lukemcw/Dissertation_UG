# Built using Joe's FredMD implementation of the MATLAB code from FRED as a starting point.
# I have extended and changed to include models and recursive estimation.
# Original available on github:
# and can be downloaded via pip: $ pip install FredMD
import pandas as pd
import numpy as np
import sklearn.decomposition as skd
import sklearn.linear_model
import sklearn.preprocessing as skp
import sklearn.pipeline as skpipe
from sklearn.ensemble import RandomForestRegressor
import math
import statsmodels as sm
from statsmodels.tsa.ar_model import AutoReg


class FredQD:
    """
    FredQD object. Creates factors based off the FRED-QD dataset (https://research.stlouisfed.org/econ/mccracken/fred-databases/)
    Methods:
    1) FredQD(): initialize object with downloaded data
    2) estimate_recursive(): Runs full recursive estimation, producing forecasts for all specified models
    2) estimate_factors(): Runs the full FRED factor estimation for one period
    3) factors_em(): Estimates factors with the EM algorithm to handle missing observations
    4) baing(): Estimates the Bai-Ng factor selection alogrithm
    5) apply_transforms(): Apply the transform to each series
    6) remove_outliers(): Removes Outliers
    7) factor_standardizer_method(): Converts standard_method to appropiate sklearn.StandardScaler
    8) data_transforms(): Applies function to series to make data stationary as given by transform code
    9) download_data(): Download FRED-QD dataset
    10) V(): Explained variance function
    """

    def __init__(self, Nfactor=None, vintage=None, maxfactor=8, standard_method=2, ic_method=2,
                 start_date='1980-03-01') -> None:
        """
        Create fredmd object
        Arguments:
        1) Nfactor = None: Number of factors to estimate. If None then estimate number of true factors via information critea
        2) vintage = None: Vintage of data to use in "year-month" format (e.g. "2020-10"). If None use current vintage
        3) maxfactor = 8: Maximum number of factors to test against information criteria. If Nfactor is a number, then this is ignored
        4) standard_method = 2: method to standardize data before factors are estimate. 0 = Identity transform, 1 = Demean only, 2 = Demean and stardize to unit variance. Default = 2.
        5) ic_method = 2: information criteria penalty term. See http://www.columbia.edu/~sn2294/pub/ecta02.pdf page 201, equation 9 for options.
        """
        # Make sure arguments are valid
        if standard_method not in [0, 1, 2]:
            raise ValueError(
                f"standard_method must be in [0, 1, 2], got {standard_method}")
        if ic_method not in [1, 2, 3]:
            raise ValueError(
                f"ic_method must be in [1, 2, 3], got {ic_method}")

        # Download data
        self.rawseries, self.transforms = self.download_data(vintage)
        # Check maxfactor
        if maxfactor > self.rawseries.shape[1]:
            raise ValueError(
                f"maxfactor must be less then number of series. Maxfactor({maxfactor}) > N Series({self.rawseries.shape[1]})")

        self.standard_method = standard_method
        self.ic_method = ic_method
        self.maxfactor = maxfactor
        self.Nfactor = Nfactor
        self.start_date = pd.to_datetime(start_date)

    @staticmethod
    def download_data(vintage):
        if vintage is None:
            url = 'https://files.stlouisfed.org/files/htdocs/fred-md/quarterly/current.csv'
        else:
            url = f'https://files.stlouisfed.org/files/htdocs/fred-md/quarterly/{vintage}.csv'
        print(url)
        transforms = pd.read_csv(
            url, header=0, nrows=2, index_col=0).transpose().drop(labels="factors", axis=1)
        transforms.index.rename("series", inplace=True)
        transforms.columns = ['transform']
        # transforms.drop([transforms.index[178], transforms.index[180]], axis=0)
        transforms = transforms.to_dict()['transform']
        data = pd.read_csv(url, names=transforms.keys(), skiprows=3, index_col=0,
                           skipfooter=2, engine='python', parse_dates=True, infer_datetime_format=True)
        # data.drop([data.columns[178], data.columns[180]], axis=1)
        return data, transforms

    @staticmethod
    def factor_standardizer_method(code):
        """
        Outputs the sklearn standard scaler object with the desired features
        codes:
        0) Identity transform
        1) Demean only
        2) Demean and standardized
        """
        if code == 0:
            return skp.StandardScaler(with_mean=False, with_std=False)
        elif code == 1:
            return skp.StandardScaler(with_mean=True, with_std=False)
        elif code == 2:
            return skp.StandardScaler(with_mean=True, with_std=True)
        else:
            raise ValueError("standard_method must be in [0, 1, 2]")

    @staticmethod
    def data_transforms(series, transform):
        """
        Transforms a single series according to its transformation code
        Inputs:
        1) series: pandas series to be transformed
        2) transform: transform code for the series
        Returns:
        transformed series
        """
        if transform == 1:
            # level
            return series
        elif transform == 2:
            # 1st difference
            return series.diff()
        elif transform == 3:
            # second difference
            return series.diff().diff()
        elif transform == 4:
            # Natural log
            return np.log(series)
        elif transform == 5:
            # log 1st difference
            return np.log(series).diff()
        elif transform == 6:
            # log second difference
            return np.log(series).diff().diff()
        elif transform == 7:
            # First difference of percent change
            return series.pct_change().diff()
        else:
            raise ValueError("Transform must be in [1, 2, ..., 7]")

    def apply_transforms(self):
        """
        Apply the transformation to each series to make them stationary and drop the first 2 rows that are mostly NaNs
        Save results to self.series
        """
        self.series = pd.DataFrame({key: self.data_transforms(
            self.rawseries[key], value) for (key, value) in self.transforms.items()})
        self.series.drop(self.series.index[[0, 1]], inplace=True)

    def remove_outliers(self):
        """
        Removes outliers from each series in self.series
        Outlier definition: a data point x of a series X is considered an outlier if abs(x-median)>10*interquartile_range.
        """
        Z = abs((self.series - self.series.median()) /
                (self.series.quantile(0.75) - self.series.quantile(0.25))) > 10
        for col, _ in self.series.iteritems():
            self.series[col][Z[col]] = np.nan

    def factors_em(self, max_iter=50, tol=math.sqrt(0.000001)):
        """
        Estimates factors with EM algorithm to handle missing values
        Inputs:
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence between iterations of predicted series values
        Algorithm:
        1) initial_nas: Boolean mask of locations of NaNs
        2) working_data: Create Standardized data matrix with nan's replaced with means
        3) F: Preliminary factor estimates
        4) data_hat_last: Predicted standardized values of last SVD model. data_hat and data_hat_last will not exactly be mean 0 variance 1
        5) Iterate data_hat until convergence
        6) Fill in nans from original data
        Saves
        1) self.svdmodel: sklearn pipeline with standardization step and svd model
        2) self.series_filled: self.series with any NaNs filled in with predicted values from self.svdmodel
        """
        # Define our estimation pipelines
        pipe = skpipe.Pipeline([('Standardize', self.factor_standardizer_method(
            self.standard_method)), ('Factors', skd.TruncatedSVD(self.Nfactor, algorithm='arpack'))])
        inital_scalar = self.factor_standardizer_method(self.standard_method)

        # Make numpy arrays for calculations
        actual_data = self.series.to_numpy(copy=True)
        intial_nas = self.series.isna().to_numpy(copy=True)
        working_data = inital_scalar.fit_transform(self.series.fillna(
            value=self.series.mean(), axis='index').to_numpy(copy=True))

        # Estimate initial model
        if np.isnan(working_data).any() == True:

            # which columns?
            problem_columns = list(np.unique(np.argwhere(np.isnan(working_data))[:, 1]))

            # remove columns with nan:
            working_data = np.delete(working_data, problem_columns, 1)
            columns_to_remove = [self.rawseries.columns[problem_column] for problem_column in problem_columns]
            self.series = self.series.drop(columns_to_remove, axis=1)
            self.rawseries = self.rawseries.drop(columns_to_remove, axis=1)

            for column in columns_to_remove:
                self.transforms = self.transforms.pop(column, None)

        assert (np.isnan(working_data).any() == False)
        F = pipe.fit_transform(working_data)
        data_hat_last = pipe.inverse_transform(F)

        # Iterate until model convereges
        iter = 0
        distance = tol + 1
        while (iter < max_iter) and (distance > tol):
            F = pipe.fit_transform(working_data)
            data_hat = pipe.inverse_transform(F)
            distance = np.linalg.norm(
                data_hat - data_hat_last, 2) / np.linalg.norm(data_hat_last, 2)
            data_hat_last = data_hat.copy()
            working_data[intial_nas] = data_hat[intial_nas]
            iter += 1

        # Print results
        if iter == max_iter:
            print(
                f"EM alogrithm failed to converge afet Maximum iterations of {max_iter}. Distance = {distance}, tolerance was {tol}")
        else:
            print(f"EM algorithm converged after {iter} iterations")

        # Save Results
        actual_data[intial_nas] = inital_scalar.inverse_transform(working_data)[
            intial_nas]
        self.Lambda = None  # TODO: Manipulate pipe to give factor loadings
        self.svdmodel = pipe
        self.series_filled = pd.DataFrame(
            actual_data, index=self.series.index, columns=self.series.columns)
        self.factors = pd.DataFrame(F, index=self.series_filled.index, columns=[
            f"F{i}" for i in range(1, F.shape[1] + 1)])

    @staticmethod
    def V(X, F, Lambda):
        """
        Explained Variance of X by factors F with loadings Lambda
        """
        T, N = X.shape
        NT = N * T
        return np.linalg.norm(X - F @ Lambda, 2) / NT

    def baing(self):
        """
        Determine the number of factors to use using the Bai-Ng Information Critrion
        reference: http://www.columbia.edu/~sn2294/pub/ecta02.pdf
        """
        # Define our estimation pipelines
        pipe = skpipe.Pipeline([('Standardize', self.factor_standardizer_method(
            self.standard_method)), ('Factors', skd.TruncatedSVD(self.maxfactor, algorithm='arpack', random_state=42))])
        inital_scalar = self.factor_standardizer_method(self.standard_method)

        # Setup
        working_data = inital_scalar.fit_transform(self.series.fillna(
            value=self.series.mean(), axis='index').to_numpy(copy=True))
        T, N = working_data.shape
        NT = N * T
        NT1 = N + T
        # Make information critea penalties
        if self.ic_method == 1:
            CT = [i * math.log(NT / NT1) * NT1 / NT for i in range(self.maxfactor)]
        elif self.ic_method == 2:
            CT = [i * math.log(min(N, T)) * NT1 /
                  NT for i in range(self.maxfactor)]
        elif self.ic_method == 3:
            CT = [i * math.log(min(N, T)) / min(N, T)
                  for i in range(self.maxfactor)]
        else:
            raise ValueError("ic must be either 1, 2 or 3")

        # Fit model with max factors
        if np.isnan(working_data).any() == True:

            # which columns?
            problem_columns = list(np.unique(np.argwhere(np.isnan(working_data))[:, 1]))

            # remove columns with nan:
            working_data = np.delete(working_data, problem_columns, 1)
            columns_to_remove = [self.rawseries.columns[problem_column] for problem_column in problem_columns]
            self.series = self.series.drop(columns_to_remove, axis=1)
            self.rawseries = self.rawseries.drop(columns_to_remove, axis=1)

            for column in columns_to_remove:
                self.transforms.pop(column, None)
        assert (np.isnan(working_data).any() == False)
        F = pipe.fit_transform(working_data)
        Lambda = pipe['Factors'].components_
        Vhat = [self.V(working_data, F[:, 0:i], Lambda[0:i, :])
                for i in range(self.maxfactor)]
        IC = np.log(Vhat) + CT
        kstar = np.argmin(IC)
        self.Nfactor = kstar

    def estimate_factors(self):
        """
        Runs estimation routine.
        If number of factors is not specified then estimate the number to be used
        """
        self.apply_transforms()
        self.remove_outliers()
        self.baing()
        self.factors_em()

    def forecast_ar2(self):
        y_train = self.series_filled['GDPC1'].copy(deep=True)
        ar_model = AutoReg(y_train, lags=[1, 2]).fit()
        # predict
        pred = ar_model.predict(start=len(y_train), end=(len(y_train) + 8 - 1), dynamic=False).values
        self.ar2 = list(pred)

    @staticmethod
    def knn1(data):
        """
        Estimates KNN model for one step ahead.
        """
        # Average 3 nearest neighbours
        av_3 = np.average([data.values[-i] for i in range(3)])
        # Average 5 nearest neighbours
        av_5 = np.average([data.values[-i] for i in range(7)])
        # Average 7 nearest neighbours
        av_7 = np.average([data.values[-i] for i in range(7)])
        # average of these averages = prediction for t+1
        t_plus_1 = np.average([av_3, av_5, av_7])
        return t_plus_1

    def forecast_knn(self):
        """
        Recursively estimates t+1 ... t+4 with
        """
        y_train = self.series_filled['GDPC1'].copy(deep=True)
        preds = []
        for i in range(8):
            # save pred as pd.Series so it can be appended
            assert (type(y_train) is not list)
            pred = self.knn1(y_train).copy()
            t_plus_1 = pd.Series(pred)
            # append to preds
            preds.append(t_plus_1.values[0])
            # append to training data
            y_train = y_train.append(t_plus_1)
            # re-index the Series to allow for next estimation
            y_train.reset_index(drop=True)

            # t_plus_1 = pd.Series(self.knn(y_train))
            # pred = pred.append(t_plus_1, ignore_index=True)
            # y_train = y_train.append(t_plus_1, ignore_index=True)
        self.knn = preds

    def get_prediction_data(self, h):
        y = self.series_filled['GDPC1'].copy(deep=True)
        # Z = pd.DataFrame(dtype=object)
        f = self.factors.shift(h)
        Z = pd.concat([y, f], axis=1, join='inner')
        Z['y_lag1'] = Z['GDPC1'].shift(h)
        Z['y_lag2'] = Z['GDPC1'].shift(h + 1)
        dates_to_drop = [Z.index[i] for i in range(h + 1)]
        Z = Z.drop(dates_to_drop, axis=0)
        Z = Z.drop(['GDPC1'], axis=1)
        # make prediction vector x
        x = pd.concat([y, f], axis=1, join='inner')
        x['y_lag1'] = y
        x['y_lag2'] = y.shift(1)
        x = x.drop(['GDPC1'], axis=1)
        x = x[-2:-1]
        y_ = y[h+1:]
        return Z, y_, x

    @staticmethod
    def factor_model(Z, y, x):
        model = sklearn.linear_model.LinearRegression()
        model.fit(Z, y)
        pred = model.predict(x)
        return pred[0]

    def forecast_factor_model(self):
        forecasts = []
        for i in range(8):
            Z, y, x = self.get_prediction_data(i + 1)
            forecasts.append(self.factor_model(Z, y, x))
        self.factor_model_pred = forecasts

    @staticmethod
    def ridge_factor_model(Z, y, x):
        model = sklearn.linear_model.RidgeCV()
        model.fit(Z, y)
        pred = model.predict(x)
        return pred[0]

    def forecast_ridge_factor_model(self):
        forecasts = []
        for i in range(8):
            Z, y, x = self.get_prediction_data(i + 1)
            pred = self.ridge_factor_model(Z, y, x)
            forecasts.append(pred)
        self.ridge_factor = forecasts

    @staticmethod
    def lasso_factor_model(Z, y, x):
        model = sklearn.linear_model.LassoCV()
        model.fit(Z, y)
        pred = model.predict(x)
        return pred[0]

    def forecast_lasso_factor_model(self):
        forecasts = []
        for i in range(8):
            Z, y, x = self.get_prediction_data(i + 1)
            pred = self.ridge_factor_model(Z, y, x)
            forecasts.append(pred)
        self.lasso_factor = forecasts

    @staticmethod
    def en_factor_model(Z, y, x):
        model = sklearn.linear_model.ElasticNetCV()
        model.fit(Z, y)
        pred = model.predict(x)
        return pred[0]

    def forecast_en_factor_model(self):
        forecasts = []
        for i in range(8):
            Z, y, x = self.get_prediction_data(i + 1)
            pred = self.ridge_factor_model(Z, y, x)
            forecasts.append(pred)
        self.en_factor = forecasts

    @staticmethod
    def random_forest(Z, y, x):
        # define X with Z and factors
        model = RandomForestRegressor()
        model.fit(Z, y)
        return model.predict(x)[0]

    def forecast_rf(self):
        forecasts = []
        for i in range(8):
            Z, y, x = self.get_prediction_data(i + 1)
            pred = self.random_forest(Z, y, x)
            forecasts.append(pred)
        self.rf = forecasts

    # @staticmethod
    # def xgboosting():
    #     # define X with Z and factors
    #     model = ()
    #     model.fit(Z, y)
    #     return model.predict(x)
    #
    # def xgboost_forecast(self):
    #     forecasts = []
    #     for i in range(8):
    #         Z, y, x = self.get_prediction_data(i + 1)
    #         pred = self.xgboosting(Z, y, x)
    #         forecasts.append(pred)
    #     self.xgboost_forecast = forecasts

    def init_forecasts(self):
        # Initiate useful values:
        self.all_Nfac = []
        self.all_estimated_factors = []
        self.knn_forecasts = []
        self.ar2_forecasts = []
        self.factor_forecasts = []
        self.ridge_forecasts = []
        self.en_forecasts = []
        self.lasso_forecasts = []
        self.rf_forecasts = []
        self.data_copy = self.rawseries.copy(deep=True)

    def estimate_models(self, i):
        # set training data
        self.rawseries = self.data_copy[:i].copy(deep=True)
        # estimate factor for given i (date)
        self.estimate_factors()
        # estimate models
        self.forecast_ar2()
        self.forecast_knn()
        self.forecast_factor_model()
        self.forecast_en_factor_model()
        self.forecast_lasso_factor_model()
        self.forecast_ridge_factor_model()
        self.forecast_rf()
        # self.forecast_xgboost()
        # append results to list
        self.knn_forecasts.append(self.knn)
        self.ar2_forecasts.append(self.ar2)
        self.ridge_forecasts.append(self.factor_model_pred)
        self.en_forecasts.append(self.factor_model_pred)
        self.lasso_forecasts.append(self.factor_model_pred)
        self.factor_forecasts.append(self.factor_model_pred)
        self.rf_forecasts.append(self.rf)
        # self.xgboost_forecasts.append(self.xgboost_forecast)
        # additional info
        self.all_Nfac.append(self.Nfactor)
        self.all_estimated_factors.append(self.factors)

    def forecasts_to_dict(self):
        # self.forecasts = {"AR(2)": self.ar2_forecasts}
        self.forecasts = {"AR(2)": self.ar2_forecasts, "KNN": self.knn_forecasts, "DFM": self.factor_forecasts,
                          "Factor EN": self.en_forecasts, "Factor Ridge": self.ridge_forecasts, "Factor Lasso": self.lasso_forecasts,
                        "Factor Random Forests": self.rf_forecasts}
        # just use MATLAB script for loadings --> quicker than re-implementing that part

    # Extract indiviudual series for each model
    @staticmethod
    def split_by_h(model_forecasts):
        # loop through model's predictions and make a series for each forecasting horizon
        h1 = [i[0] for i in model_forecasts]
        h2 = [i[1] for i in model_forecasts]
        h3 = [i[2] for i in model_forecasts]
        h4 = [i[3] for i in model_forecasts]
        h5 = [i[4] for i in model_forecasts]
        h6 = [i[5] for i in model_forecasts]
        h7 = [i[6] for i in model_forecasts]
        h8 = [i[7] for i in model_forecasts]
        horizon_list = [h1, h2, h3, h4, h5, h6, h7, h8]
        return horizon_list

    def forecasts_to_df(self):
        # Split forecasts by horizon
        for model_name, model_forecasts in self.forecasts.items():
            self.forecasts[model_name] = self.split_by_h(model_forecasts)
        # dates for
        h1_dates = self.series_filled.index[self.series.index.to_list().index(self.start_date) + 1:]
        h2_dates = self.series_filled.index[self.series.index.to_list().index(self.start_date) + 2:]
        h3_dates = self.series_filled.index[self.series.index.to_list().index(self.start_date) + 3:]
        h4_dates = self.series_filled.index[self.series.index.to_list().index(self.start_date) + 4:]
        h5_dates = self.series_filled.index[self.series.index.to_list().index(self.start_date) + 5:]
        h6_dates = self.series_filled.index[self.series.index.to_list().index(self.start_date) + 6:]
        h7_dates = self.series_filled.index[self.series.index.to_list().index(self.start_date) + 7:]
        h8_dates = self.series_filled.index[self.series.index.to_list().index(self.start_date) + 8:]
        assert(len(h8_dates) < len(h1_dates))
        ################################################################################################################
        # Make a df of each horizon:
        # Within loop each model_forecast is a list of forecasts organised by horizon
        # e.g. model_forecast[0][:len(h1_dates)] = list of forecasts for h=1 with forecasts
        # up to the point we have data for
        ################################################################################################################
        model_names = self.forecasts.keys()
        h1 = pd.DataFrame([model_forecast[0][:len(h1_dates)] for model_name, model_forecast in self.forecasts.items()], index=model_names, columns=h1_dates)
        h2 = pd.DataFrame([model_forecast[1][:len(h2_dates)] for model_name, model_forecast in self.forecasts.items()], index=model_names, columns=h2_dates)
        h3 = pd.DataFrame([model_forecast[2][:len(h3_dates)] for model_name, model_forecast in self.forecasts.items()], index=model_names, columns=h3_dates)
        h4 = pd.DataFrame([model_forecast[3][:len(h4_dates)] for model_name, model_forecast in self.forecasts.items()], index=model_names, columns=h4_dates)
        h5 = pd.DataFrame([model_forecast[4][:len(h5_dates)] for model_name, model_forecast in self.forecasts.items()], index=model_names, columns=h5_dates)
        h6 = pd.DataFrame([model_forecast[5][:len(h6_dates)] for model_name, model_forecast in self.forecasts.items()], index=model_names, columns=h6_dates)
        h7 = pd.DataFrame([model_forecast[6][:len(h7_dates)] for model_name, model_forecast in self.forecasts.items()], index=model_names, columns=h7_dates)
        h8 = pd.DataFrame([model_forecast[7][:len(h8_dates)] for model_name, model_forecast in self.forecasts.items()], index=model_names, columns=h8_dates)
        self.herizon_forecasts = [h1, h2, h3, h4, h5, h6, h7, h8]

    def forecast_recursive(self):
        """
        Runs recursive estimation routine.
        """
        self.init_forecasts()
        # Loop through copied raw data and use this to set the length of the raw data
        # Model has a function which sets a self.model equal to its list/series of 4 predictions
        for i in self.data_copy.index.tolist():
            if i >= self.start_date:
                # estimates models and forecasts h=[1,...,8] with data up to date i
                self.estimate_models(i)
        # store in dicts
        self.forecasts_to_dict()
        self.forecasts_to_df()
        print(self.herizon_forecasts)


if __name__ == '__main__':
    # Initiate FredQD and estimate current factors
    fqd = FredQD(Nfactor=None, vintage=None, maxfactor=8, standard_method=2, ic_method=2, start_date="1999-01-01")
    # fqd.estimate_factors()
    # f = fqd.factors
    # print(f)
    # TODO: check recursive estimation of factors
    fqd.forecast_recursive()
    print(fqd.forecasts)
