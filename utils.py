import numpy as np
import pandas as pd

# Normalize
def mase_scaler(X, serie_season):
    MASEs = []
    for serie, season in zip(X, serie_season):
        MASEs.append( np.mean(abs(serie[season:] - serie[:-season])) )
    return MASEs

def std_scaler(X):
    params = []
    for serie in X:
        params.append((np.mean(serie), np.std(serie)))
    return params

#############################################################################################################

# Build a shifted dataset
def lag(X, lag, h, maxSamples=None, padding=False, filterZeros=False):
    Xlagged = []
    Y = []
    tsId = []
    for id_, serie in enumerate(X):
        i=0
        serie_samples_X = []
        serie_samples_Y = []
        serie_samples_id = []
        T = serie.shape[0]

        if (padding and T <= lag):
            padSize = lag - T + 1
            serie = np.pad(serie, (padSize,0), 'constant')
            T += padSize

        while True:
            serie_samples_X.append(serie[i:(i+lag)])
            serie_samples_Y.append(serie[(i+lag):(i+lag+h)])
            serie_samples_id.append(id_)
            i += 1
            if (i+lag+h > T): break

        if (not maxSamples or maxSamples > len(serie_samples_X)): # eventually, take maxSamples for each ts
            maxSamples = len(serie_samples_X)
        Xlagged += serie_samples_X[-maxSamples:]
        Y += serie_samples_Y[-maxSamples:]
        tsId += serie_samples_id[-maxSamples:]

    Xlagged = np.array(Xlagged)
    Y = np.array(Y)
    tsId = np.array(tsId)

    if filterZeros: # eventually, remove all zero samples
        mask = np.logical_not(np.sum( Xlagged, axis = 1) == 0)
        Xlagged = Xlagged[mask,:]
        Y = Y[mask,:]
    return Xlagged, Y, tsId

#############################################################################################################

# Performance Indicators

def MAE(y, yhat):
    return np.mean(np.abs(y - yhat), axis=0)

def RMSE(y, yhat):
    return np.sqrt(np.mean(np.square(y - yhat), axis=0))