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
    
    df["Group"] = rows
    df.set_index("Group", drop=True, inplace=True)


def get_mean(df):
    return df.mean(axis=1)


def run_chart(series, x_label="no.", y_label="measure"):
    n_points = len(series)
    x = np.arange(n_points) + 1
    y = series
    
    plt.plot(x, y, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    plt.savefig("run_chart.png")
    plt.show()


def histogram(series, bins=10, x_label="measure", y_label="frequency"):
    plt.hist(series, bins=bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def test_run_chart():
    np.random.seed(0)
    df = generate_data()
    add_labels(df)
    run_chart(df.iloc[0])


if __name__ == "__main__":
    test_run_chart()
