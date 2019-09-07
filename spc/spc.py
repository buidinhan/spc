import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def generate_data():
    array = np.random.randint(100, 120, (5, 10))
    return pd.DataFrame(array)


def add_labels(df):
    n_rows, n_cols = df.shape
    rows = ["Time {}".format(x+1) for x in range(n_rows)]
    cols = ["Measure {}".format(x+1) for x in range(n_cols)]
    df.columns = cols
    df.set_index(rows, inplace=True)


def get_mean(df):
    return df.mean(axis=1)
