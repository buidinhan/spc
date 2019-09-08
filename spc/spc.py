import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import anderson


CONSTANTS = pd.read_csv("constants.csv", index_col="n")


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
    return df.max(axis=1) - df.min(axis=1)


def run_chart(series, centerline=False,
              LSL=None, USL=None,
              x_label="No.", y_label="Measure",
              save=True, show=False):

    n_points = len(series)
    x = np.arange(n_points) + 1
    y = series

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, marker="o")
    plt.title("Run Chart")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if USL is not None:
        plt.plot(x, [USL]*n_points, "r", label="USL")

    if centerline:
        center = np.mean(y)
        plt.plot(x, [center]*n_points, "k", label="Mean")

    if LSL is not None:
        plt.plot(x, [LSL]*n_points, "r", label="LSL")
    
    plt.legend(loc="best", fancybox=True, framealpha=0.5)

    if save:
        plt.savefig("output__run_chart.png")

    if show:
        plt.show()


def histogram(series, bins=10, x_label="Measure", y_label="Frequency"):
    plt.figure(figsize=(9, 6))
    plt.hist(series, bins=bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def xbar_s_chart(df, save=True, show=False):
    means = get_means(df)
    stds = get_stds(df)    
    
    X_bar = means.mean()
    s_bar = stds.mean()

    n = df.shape[1]
    A3 = get_constant(n, "A3")
    UCL = X_bar + A3*s_bar
    LCL = X_bar - A3*s_bar

    groups = means.index

    plt.figure(figsize=(9, 6))
    plt.plot(groups, means, marker="o")
    plt.plot(groups, [UCL]*len(groups), "r",
             label="UCL={:.2f}".format(UCL))
    plt.plot(groups, [X_bar]*len(groups), "k",
             label="Mean={:.2f}".format(X_bar))
    plt.plot(groups, [LCL]*len(groups), "r",
             label="LCL={:.2f}".format(LCL))
    plt.xticks(rotation=45)
    plt.ylabel("X-bar")
    plt.legend(loc="best", fancybox=True, framealpha=0.5)
    plt.title("X-bar (S) Chart")

    if save:
        plt.savefig("output__xbar_s_chart.png")

    if show:
        plt.show()


def s_chart(df, save=True, show=False):
    stds = get_stds(df)
    s_bar = stds.mean()
    
    n = df.shape[1]
    B4 = get_constant(n, "B4")
    B3 = get_constant(n, "B3")
    UCL = B4 * s_bar
    LCL = B3 * s_bar

    groups = stds.index

    plt.figure(figsize=(9, 6))
    plt.plot(groups, stds, marker="o")
    plt.plot(groups, [UCL]*len(groups), "r",
             label="UCL={:.2f}".format(UCL))
    plt.plot(groups, [s_bar]*len(groups), "k",
             label="s-bar={:.2f}".format(s_bar))
    plt.plot(groups, [LCL]*len(groups), "r",
             label="LCL={:.2f}".format(LCL))
    plt.xticks(rotation=45)
    plt.ylabel("s")
    plt.legend(loc="best", fancybox=True, framealpha=0.5)
    plt.title("S Chart")

    if save:
        plt.savefig("output__s_chart.png")

    if show:
        plt.show()


def xbar_r_chart(df, save=True, show=False):
    means = get_means(df)
    ranges = get_ranges(df)    
    
    X_bar = means.mean()
    r_bar = ranges.mean()

    n = df.shape[1]
    A2 = get_constant(n, "A2")
    UCL = X_bar + A2*r_bar
    LCL = X_bar - A2*r_bar

    groups = means.index

    plt.figure(figsize=(9, 6))
    plt.plot(groups, means, marker="o")
    plt.plot(groups, [UCL]*len(groups), "r",
             label="UCL={:.2f}".format(UCL))
    plt.plot(groups, [X_bar]*len(groups), "k",
             label="Mean={:.2f}".format(X_bar))
    plt.plot(groups, [LCL]*len(groups), "r",
             label="LCL={:.2f}".format(LCL))
    plt.xticks(rotation=45)
    plt.ylabel("X-bar")
    plt.legend(loc="best", fancybox=True, framealpha=0.5)
    plt.title("X-bar (R) Chart")

    if save:
        plt.savefig("output__xbar_r_chart.png")

    if show:
        plt.show()


def r_chart(df, save=True, show=False):
    ranges = get_ranges(df)
    r_bar = ranges.mean()
    
    n = df.shape[1]
    D4 = get_constant(n, "D4")
    D3 = get_constant(n, "D3")
    UCL = D4 * r_bar
    LCL = D3 * r_bar

    groups = ranges.index

    plt.figure(figsize=(9, 6))
    plt.plot(groups, ranges, marker="o")
    plt.plot(groups, [UCL]*len(groups), "r",
             label="UCL={:.2f}".format(UCL))
    plt.plot(groups, [r_bar]*len(groups), "k",
             label="R-bar={:.2f}".format(r_bar))
    plt.plot(groups, [LCL]*len(groups), "r",
             label="LCL={:.2f}".format(LCL))
    plt.xticks(rotation=45)
    plt.ylabel("Range")
    plt.legend(loc="best", fancybox=True, framealpha=0.5)
    plt.title("R Chart")

    if save:
        plt.savefig("output__r_chart.png")

    if show:
        plt.show()


def group_scattering(df, y_label="Measure", save=True, show=False):
    groups = df.index
    n = df.shape[1]

    plt.figure(figsize=(9, 6))
    
    for group in groups:
        plt.scatter([group]*n, df.loc[group, :], c="blue")

    plt.xticks(rotation=45)
    plt.ylabel(y_label)

    if save:
        plt.savefig("output__scatter.png")

    if show:
        plt.show()


def moving_range_chart(df, save=True, show=False):
    means = get_means(df)
    MRs = np.abs(means[1:].values-means[:-1].values)
    indices = [str(x+2) for x in range(len(MRs))]
    
    MR_bar = np.mean(MRs)
    
    D4 = get_constant(2, "D4")
    UCL = D4 * MR_bar
    LCL = 0

    plt.figure(figsize=(9, 6))
    plt.plot(indices, MRs, marker="o")
    plt.plot(indices, [UCL]*len(MRs), "r",
             label="UCL={:.2f}".format(UCL))
    plt.plot(indices, [MR_bar]*len(MRs), "k",
             label="MR-bar={:.2f}".format(MR_bar))
    plt.plot(indices, [LCL]*len(MRs), "r",
             label="LCL={:.2f}".format(LCL))
    
    plt.ylabel("Moving Range")
    plt.legend(loc="best", fancybox=True, framealpha=0.5)
    plt.title("Subgroup MR Chart")

    if save:
        plt.savefig("output__mr_chart.png")

    if show:
        plt.show()


def capability_histogram(df, x_label="Measure", bins=10,
                         LSL=None, USL=None,
                         save=True, show=False):
    values = df.values.ravel()
    mean = np.mean(values)
    std = np.std(values)
    
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.axes()
    
    ax1.hist(values, bins=bins)
    
    # Drawing vertical lines
    if USL is not None:
        plt.axvline(x=USL, c="red", label="USL={:.2f}".format(USL))
    
    ax1.axvline(x=mean+3*std, c="black",
                label="mean+3s={:.2f}".format(mean+3*std))
    ax1.axvline(x=mean-3*std, c="black",
                label="mean-3s={:.2f}".format(mean-3*std))
    
    if LSL is not None:
        ax1.axvline(x=LSL, c="red", label="LSL={:.2f}".format(LSL))
        
    
    # Probability density function
    def f(x):
        coef = 1/np.sqrt(2*np.pi*(std**2))
        power = -(x-mean)**2 / (2*(std**2))
        return coef * np.exp(power)
    
    Xs = np.arange(mean-4*std, mean+4*std, 0.01)
    Ys = f(Xs)
    
    ax2 = ax1.twinx()
    ax2.plot(Xs, Ys, c="green")
    
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Count")
    ax1.legend(loc="upper right", fancybox=True, framealpha=0.5)
    ax1.set_title("Capability Histogram")

    if save:
        plt.savefig("output__histogram.png")

    if show:
        plt.show()


def normality_test(df):
    values = df.values.ravel()
    
    test_results = anderson(values, dist="norm")
    AD = test_results[0]
    p_value = test_results[1][2]
    
    with open("output__normality_test.txt", "w") as f:
        f.write("Anderson-Darling Test for Normality\n")
        f.write("   Anderson-Darling test static = {:.4f}\n".format(AD))
        f.write("   p-value = {:.4f}".format(p_value))
    

def performance_indices(df, LSL, USL):
    values = df.values.ravel()
    mean = np.mean(values)
    std = np.std(values)
    
    Pp = abs(USL-LSL) / (6*std)
    Ppk = min(abs(LSL-mean), abs(USL-mean)) / (3*std)
    
    return Pp, Ppk


def capability_indices(df, LSL, USL,
                       estimation_method="std"):
    means = get_means(df)
    X_bar = np.mean(means)
    n = df.shape[1]
    
    if estimation_method == "range":
        ranges = get_ranges(df)
        r_bar = np.mean(ranges)
        d2 = get_constant(n, "d2")
        estimated_s = r_bar / d2
        
    elif estimation_method == "std":
        stds = get_stds(df)
        s_bar = np.mean(stds)
        c4 = get_constant(n, "c4")
        estimated_s = s_bar / c4
    
    else:
        raise ValueError('estimation_method should be "std" or "range"')
        
    Cp = abs(USL-LSL) / (6*estimated_s)
    Cpk = min(abs(USL-X_bar), abs(LSL-X_bar)) / (3*estimated_s)
    
    return Cp, Cpk


def output_indices(df, LSL, USL, estimation_method="std"):
    Cp, Cpk = capability_indices(df, LSL, USL,
                  estimation_method=estimation_method)
    Pp, Ppk = performance_indices(df, LSL, USL)

    with open("output__indices.txt", "w") as f:
        f.write("Capability Indices:\n")
        f.write("    Cp  = {:.2f}\n".format(Cp))
        f.write("    Cpk = {:.2f}\n\n".format(Cpk))
        f.write("Performance Indices:\n")
        f.write("    Pp  = {:.2f}\n".format(Pp))
        f.write("    Ppk = {:.2f}".format(Ppk))


def test():
    path = "fridge.csv"
    df = pd.read_csv(path)
    df.drop("shift", axis=1, inplace=True)
    add_labels(df)

    run_chart(df.values.ravel(), centerline=True,
              LSL=1, USL=4, save=False, show=True)
    group_scattering(df, save=False, show=True)

    xbar_s_chart(df, save=False, show=True)
    s_chart(df, save=False, show=True)
    xbar_r_chart(df, save=False, show=True)
    r_chart(df, save=False, show=True)
    moving_range_chart(df, save=False, show=True)
    capability_histogram(df, LSL=1, USL=4, save=False, show=True)
    
    normality_test(df)
    output_indices(df, 1, 4)


if __name__ == "__main__":
    test()
