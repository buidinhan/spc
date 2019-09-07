import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def get_data(path):
    df = pd.read_csv(path, header=None)
    
    n_rows, n_cols = df.shape
    rows = ["Time {}".format(x+1) for x in range(n_rows)]
    cols = ["Measure {}".format(x+1) for x in range(n_cols)]
    df.columns = cols
    df.set_index(rows, inplace=True)

    return df


def get_mean(df):
    return df.mean(axis=1)


