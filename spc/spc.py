import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


##CONSTANTS = pd.read_csv("cc_constants.csv", index_col="n")


def generate_data():
    array = np.random.randint(95, 105, (10, 5))
    return pd.DataFrame(array)


def add_labels(df):
    n_rows, n_cols = df.shape
    rows = ["Group {}".format(x+1) for x in range(n_rows)]
    cols = ["Measure {}".format(x+1) for x in range(n_cols)]
    
    df.columns = cols
    
    df["Group"] = rows
    df.set_index("Group", drop=True, inplace=True)


def get_constant(n, c_name):
    return CONSTANTS.loc[n, c_name]


def get_means(df):
    return df.mean(axis=1)


def get_stds(df):
    return df.std(axis=1)


def get_ranges(df):
    return df.max(axis=1)-df.min(axis=1)


def run_chart(series, centerline=False,
              USL=None, LSL=None,
              x_label="No.", y_label="Measure"):

    n_points = len(series)
    x = np.arange(n_points) + 1
    y = series
    
    plt.plot(x, y, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if centerline:
        center = np.mean(y)
        plt.plot(x, [center]*n_points, "k", label="Mean")

    if USL is not None:
        plt.plot(x, [USL]*n_points, "r", label="USL")

    if LSL is not None:
        plt.plot(x, [LSL]*n_points, "r", label="LSL")
    
    plt.legend(loc="right", bbox_to_anchor=(1.25, 0.85))
    plt.show()


def histogram(series, bins=10, x_label="Measure", y_label="Frequency"):
    plt.hist(series, bins=bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def xbar_chart(df):
    means = get_means(df)
    stds = get_stds(df)    
    
    X_bar = means.mean()
    s_bar = stds.mean()

    n = df.shape[1]
    A1 = get_constant(n, "A1")
    UCL = X_bar + A1*s_bar
    LCL = X_bar - A1*s_bar

    groups = means.index

    plt.plot(groups, means, marker="o")
    plt.plot(groups, [X_bar]*len(groups), "k", label="Mean")
    plt.plot(groups, [UCL]*len(groups), "r", label="UCL")
    plt.plot(groups, [LCL]*len(groups), "r", label="LCL")
    plt.xticks(rotation=90)
    plt.ylabel("X-bar")
    plt.legend(loc="right", bbox_to_anchor=(1.25, 0.85))
    
    plt.show()
