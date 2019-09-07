import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


CONSTANTS = pd.read_csv("cc_constants.csv", index_col="n")

def generate_data():
    array = np.random.randint(95, 105, (5, 10))
    return pd.DataFrame(array)


def add_labels(df):
    n_rows, n_cols = df.shape
    rows = ["Time {}".format(x+1) for x in range(n_rows)]
    cols = ["Measure {}".format(x+1) for x in range(n_cols)]
    
    df.columns = cols
    
    df["Group"] = rows
    df.set_index("Group", drop=True, inplace=True)


def get_mean(df):
    return df.mean(axis=1)


def run_chart(series, centerline=False,
              USL=None, LSL=None,
              x_label="no.", y_label="measure"):

    n_points = len(series)
    x = np.arange(n_points) + 1
    y = series
    
    plt.plot(x, y, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if centerline:
        center = np.mean(y)
        plt.plot(x, [center]*n_points, "k", label="mean")

    if USL is not None:
        plt.plot(x, [USL]*n_points, "r", label="USL")

    if LSL is not None:
        plt.plot(x, [LSL]*n_points, "r", label="LSL")
    
    plt.legend(loc="right", bbox_to_anchor=(1.25, 0.9))
    plt.show()


def histogram(series, bins=10, x_label="measure", y_label="frequency"):
    plt.hist(series, bins=bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
