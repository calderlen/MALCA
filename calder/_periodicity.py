import numpy as np
from astropy.timeseries import LombScargle

def lsp(df, n_terms):
    # df[i] = time, df[j] = flux
    #period, power = LombScargle(df["i"], df[j"]) obviously make this a vectorized operation; may need to increase number of Fourier terms to account for things that look like dippers
    # calculate the FAP (?)
    # return period_power
    pass

def lc_periodicity(path):
    # for a file path(s) of light curves, convert them to df one-by-one and run lsp (df) on them
    # maybe try analyzing the lc with several n_term LSPs
    pass