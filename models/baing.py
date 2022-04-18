# This script runs the Bai&Ng criteria on the transformed data.
# It shows that nfac=8 for all periods considered.
# Functions used are from Python implementation of FRED MATLAB Code by Associate Professor George Milunovich.
# Repository available at https://github.com/geoluna/FactorModels
# Run the file to check factors on data
import numpy as np
import pandas as pd


def minindc(X):
    ''' =========================================================================
     takes np <-> returns np
     DESCRIPTION
     This function finds the index of the minimum value for each column of a
     given matrix. The function assumes that the minimum value of each column
     occurs only once within that column. The function returns an error if
     this is not the case.
     -------------------------------------------------------------------------
     INPUT
               x   = matrix
     OUTPUT
               pos = column vector with pos(i) containing the row number
                     corresponding to the minimum value of x(:,i)
     ========================================================================= '''

    mins = X.argmin(axis=0)
    assert sum(X == X[mins]) == 1, 'Minimum value occurs more than once.'
    return mins


def baing(X, kmax=100, jj=2):
    # take in and return numpy arrays
    ''' =========================================================================
    DESCRIPTION
    This function determines the number of factors to be selected for a given
    dataset using one of three information criteria specified by the user.
    The user also specifies the maximum number of factors to be selected.
    -------------------------------------------------------------------------
    INPUTS
               X       = dataset (one series per column)
               kmax    = an integer indicating the maximum number of factors
                         to be estimated
               jj      = an integer indicating the information criterion used
                         for selecting the number of factors; it can take on
                         the following values:
                               1 (information criterion PC_p1)
                               2 (information criterion PC_p2)
                               3 (information criterion PC_p3)
     OUTPUTS
               ic1     = number of factors selected
               chat    = values of X predicted by the factors
               Fhat    = factors
               eigval  = eivenvalues of X'*X (or X*X' if N>T)
     -------------------------------------------------------------------------
     SUBFUNCTIONS USED
     minindc() - finds the index of the minimum value for each column of a given matrix
     -------------------------------------------------------------------------
     BREAKDOWN OF THE FUNCTION
     Part 1: Setup.
     Part 2: Calculate the overfitting penalty for each possible number of
             factors to be selected (from 1 to kmax).
     Part 3: Select the number of factors that minimizes the specified
             information criterion by utilizing the overfitting penalties calculated in Part 2.
     Part 4: Save other output variables to be returned by the function (chat,
             Fhat, and eigval).
    ========================================================================= '''
    assert kmax <= X.shape[1] and kmax >= 1 and np.floor(kmax) == kmax or kmax == 99, 'kmax is specified incorrectly'
    assert jj in [1, 2, 3], 'jj is specified incorrectly'
    #  PART 1: SETUP
    T = X.shape[0]  # Number of observations per series (i.e. number of rows)
    N = X.shape[1]  # Number of series (i.e. number of columns)
    NT = N * T  # Total number of observations
    NT1 = N + T  # Number of rows + columns
    #  =========================================================================
    #  PART 2: OVERFITTING PENALTY
    #  Determine penalty for overfitting based on the selected information
    #  criterion.
    CT = np.zeros(kmax)  # overfitting penalty
    ii = np.arange(1, kmax + 1)  # Array containing possible number of factors that can be selected (1 to kmax)
    GCT = min(N, T)  # The smaller of N and T
    # Calculate penalty based on criterion determined by jj.
    if jj == 1:  # Criterion PC_p1
        CT[:] = np.log(NT / NT1) * ii * (NT1 / NT)
    elif jj == 2:  # Criterion PC_p2
        CT[:] = np.log(min(N, T)) * ii * (NT1 / NT)
    elif jj == 3:  # Criterion PC_p3
        CT[:] = np.log(GCT) / GCT * ii
    #  =========================================================================
    #  PART 3: SELECT NUMBER OF FACTORS
    #  Perform principal component analysis on the dataset and select the number
    #  of factors that minimizes the specified information criterion.
    #
    #  -------------------------------------------------------------------------
    #  RUN PRINCIPAL COMPONENT ANALYSIS
    #  Get components, loadings, and eigenvalues
    if T < N:
        ev, eigval, V = np.linalg.svd(np.dot(X, X.T))  # Singular value decomposition
        Fhat0 = ev * np.sqrt(T)  # Components
        Lambda0 = np.dot(X.T, Fhat0) / T  # Loadings
    else:
        ev, eigval, V = np.linalg.svd(np.dot(X.T, X))  # Singular value decomposition
        Lambda0 = ev * np.sqrt(N)  # Loadings
        Fhat0 = np.dot(X, Lambda0) / N  # Components
    #  -------------------------------------------------------------------------
    # SELECT NUMBER OF FACTORS
    # Preallocate memory
    Sigma = np.zeros(kmax + 1)  # sum of squared residuals divided by NT, kmax factors + no factor
    IC1 = np.zeros(kmax + 1)  # information criterion value, kmax factors + no factor
    for i in range(0, kmax):  # Loop through all possibilities for the number of factors
        Fhat = Fhat0[:, :i + 1]  # Identify factors as first i components
        lambda_ = Lambda0[:, :i + 1]  # % Identify factor loadings as first i loadings
        chat = np.dot(Fhat, lambda_.T)  # % Predict X using i factors
        ehat = X - chat  # Residuals from predicting X using the factors
        Sigma[i] = ((ehat * ehat / T).sum(axis=0)).mean()  # Sum of squared residuals divided by NT
        IC1[i] = np.log(Sigma[i]) + CT[i]  # Value of the information criterion when using i factors
    Sigma[kmax] = (X * X / T).sum(
        axis=0).mean()  # Sum of squared residuals when using no factors to predict X (i.e. fitted values are set to 0)
    IC1[kmax] = np.log(Sigma[kmax])  # % Value of the information criterion when using no factors
    ic1 = minindc(IC1)  # % Number of factors that minimizes the information criterion
    # Set ic1=0 if ic1>kmax (i.e. no factors are selected if the value of the
    # information criterion is minimized when no factors are used)
    ic1 = ic1 * (ic1 < kmax)  # if = kmax -> 0
    #  =========================================================================
    #  PART 4: SAVE OTHER OUTPUT
    #
    #  Factors and loadings when number of factors set to kmax
    Fhat = Fhat0[:, :kmax]  # factors
    Lambda = Lambda0[:, :kmax]  # factor loadings
    chat = np.dot(Fhat, Lambda.T)  # Predict X using kmax factors
    return ic1 + 1, chat, Fhat, eigval


if __name__ == '__main__':
    DATA_FILE = r"..\data\transformed_data.csv"
    data = pd.read_csv(DATA_FILE)
    dates = data['date']
    X = data.drop(['date', "'GDPC1'"], axis=1)
    nfac_series = pd.Series(index=dates, dtype=object)
    print("Commencing Bai and Ng Criteria check")
    for i in data.index:
        if i > 20:
            train_data = X[:i]
            nfac, chat, Fhat, eigval = baing(train_data)
            print(f"{nfac} latent factors in {dates[i]}")
            nfac_series[i] = nfac
    nfac_series = nfac_series.drop([dates[i] for i in range(21)])
    values = nfac_series.values.tolist()
    print("With kmax=8, the Bai and Ng information criteria PC_p2 gives k*=8.")
    print(nfac_series)
