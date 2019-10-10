import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from utils.control_charts import *
from utils.capability_indices import *


# UTILITY FUNCTIONS
def show_and_save_plot(show=True, save=False, filename="plot.png"):
    plt.gcf()

    if save:
        plt.savefig(filename)

    if show:
        plt.show()


# MAIN PLOTTING FUNCTIONS
def plot_run_chart(series, LSL, USL, show_mean=True, x_label=None,
                   y_label=None, title=None, ax=None, show=True,
                   save=False, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6),
                               gridspec_kw={"left": 0.07,
                                            "right": 0.89,
                                            "bottom": 0.1,
                                            "top": 0.94})

    ax.axhline(y=LSL)
    ax.annotate("LSL={:.3f}".format(LSL), xy=(1.01, LSL),
                xycoords=("axes fraction", "data"))
    
    ax.axhline(y=USL)
    ax.annotate("USL={:.3f}".format(USL), xy=(1.01, USL),
                xycoords=("axes fraction", "data"))

    if show_mean:
        X_bar = np.mean(series)
        ax.axhline(y=X_bar)
        ax.annotate("Mean={:.3f}".format(X_bar), xy=(1.01, X_bar),
                    xycoords=("axes fraction", "data"))

    indices = [x+1 for x in range(len(series))]
    ax.plot(indices, series, **kwargs)
    ax.set_title(title)

    if x_label is None:
        x_label = "No."

    if y_label is None:
        y_label = "Measure"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    show_and_save_plot(show=show, save=save, filename="run_chart.png")


def plot_histogram():
    # TO BE IMPLEMENTED
    return


def plot_xbar_s_chart():
    # TO BE IMPLEMENTED
    return


def plot_xbar_r_chart():
    # TO BE IMPLEMENTED
    return


def plot_r_chart():
    # TO BE IMPLEMENTED
    return


def plot_s_chart():
    # TO BE IMPLEMENTED
    return


def plot_moving_range_chart():
    # TO BE IMPLEMENTED
    return


def plot_p_chart():
    # TO BE IMPLEMENTED
    return


def plot_single_measure_control_chart():
    # TO BE IMPLEMENTED
    return


# TESTING
def test_plot_run_chart():
    df, target, LSL, USL = load_input("../testing/fridge.dat")
    all_values = df.to_numpy().ravel()
    plot_run_chart(all_values, LSL, USL, y_label="Thickness",
                   title="Run Chart", ax=None, marker="o")


if __name__ == "__main__":
    test_plot_run_chart()


##def histogram(series, bins=10, x_label="Measure", y_label="Frequency"):
##    plt.figure(figsize=(WIDTH, HEIGHT))
##    plt.hist(series, bins=bins, edgecolor="k")
##    plt.xlabel(x_label)
##    plt.ylabel(y_label)
##    plt.show()
##
##
##def xbar_s_chart(df, save=True, show=False):
##    means = get_means(df)
##    stds = get_stds(df)    
##    
##    X_bar = means.mean()
##    s_bar = stds.mean()
##
##    n = df.shape[1]
##    A3 = get_constant(n, "A3")
##    UCL = X_bar + A3*s_bar
##    LCL = X_bar - A3*s_bar
##
##    groups = means.index.values
##   
##    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
##    ax.plot(groups, means, marker="o")
##    
##    plt.plot(groups, [UCL]*len(groups), "r")
##    plt.annotate(s="UCL={:.2f}".format(UCL), xy=(len(groups)-1+0.1, UCL))
##    
##    plt.plot(groups, [X_bar]*len(groups), "k")
##    plt.annotate(s="Mean={:.2f}".format(X_bar), xy=(len(groups)-1+0.1, X_bar))
##    
##    plt.plot(groups, [LCL]*len(groups), "r")
##    plt.annotate(s="LCL={:.2f}".format(LCL), xy=(len(groups)-1+0.1, LCL))
##    
##    plt.xticks(rotation=60)
##    plt.ylabel("X-bar")
##    ax.spines['top'].set_visible(False)
##    ax.spines['right'].set_visible(False)
##    plt.title("X-bar (S) Chart\n")
##
##    if save:
##        plt.savefig("output__xbar_s_chart.png")
##
##    if show:
##        plt.show()
##
##
##def s_chart(df, save=True, show=False):
##    stds = get_stds(df)
##    s_bar = stds.mean()
##    
##    n = df.shape[1]
##    B4 = get_constant(n, "B4")
##    B3 = get_constant(n, "B3")
##    UCL = B4 * s_bar
##    LCL = B3 * s_bar
##
##    groups = stds.index.values
##
##    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
##    ax.plot(groups, stds, marker="o")
##    
##    plt.plot(groups, [UCL]*len(groups), "r")
##    plt.annotate(s="UCL={:.2f}".format(UCL), xy=(len(groups)-1+0.1, UCL))
##    
##    plt.plot(groups, [s_bar]*len(groups), "k")
##    plt.annotate(s="s-bar={:.2f}".format(s_bar), xy=(len(groups)-1+0.1, s_bar))
##    
##    plt.plot(groups, [LCL]*len(groups), "r")
##    plt.annotate(s="LCL={:.2f}".format(LCL), xy=(len(groups)-1+0.1, LCL))
##    
##    ax.spines['top'].set_visible(False)
##    ax.spines['right'].set_visible(False)
##    plt.xticks(rotation=60)
##    plt.ylabel("S     ", rotation=0)
##    plt.title("S Chart\n")
##
##    if save:
##        plt.savefig("output__s_chart.png")
##
##    if show:
##        plt.show()
##
##
##def xbar_r_chart(df, save=True, show=False):
##    means = get_means(df)
##    ranges = get_ranges(df)    
##    
##    X_bar = means.mean()
##    r_bar = ranges.mean()
##
##    n = df.shape[1]
##    A2 = get_constant(n, "A2")
##    UCL = X_bar + A2*r_bar
##    LCL = X_bar - A2*r_bar
##
##    groups = means.index.values
##
##    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
##    plt.plot(groups, means, marker="o")
##    
##    plt.plot(groups, [UCL]*len(groups), "r")
##    plt.annotate(s="UCL={:.2f}".format(UCL), xy=(len(groups)-1+0.1, UCL))
##    
##    plt.plot(groups, [X_bar]*len(groups), "k")
##    plt.annotate(s="Mean={:.2f}".format(X_bar), xy=(len(groups)-1+0.1, X_bar))
##    
##    plt.plot(groups, [LCL]*len(groups), "r")
##    plt.annotate(s="LCL={:.2f}".format(LCL), xy=(len(groups)-1+0.1, LCL))
##    
##    ax.spines['top'].set_visible(False)
##    ax.spines['right'].set_visible(False)
##    plt.xticks(rotation=60)
##    plt.ylabel("X-bar")
##    plt.title("X-bar (R) Chart\n")
##
##    if save:
##        plt.savefig("output__xbar_r_chart.png")
##
##    if show:
##        plt.show()
##
##
##def r_chart(df, save=True, show=False):
##    ranges = get_ranges(df)
##    r_bar = ranges.mean()
##    
##    n = df.shape[1]
##    D4 = get_constant(n, "D4")
##    D3 = get_constant(n, "D3")
##    UCL = D4 * r_bar
##    LCL = D3 * r_bar
##
##    groups = ranges.index.values
##
##    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
##    ax.plot(groups, ranges, marker="o")
##    
##    plt.plot(groups, [UCL]*len(groups), "r")
##    plt.annotate(s="UCL={:.2f}".format(UCL), xy=(len(groups)-1+0.1, UCL))
##    
##    plt.plot(groups, [r_bar]*len(groups), "k")
##    plt.annotate(s="R-bar={:.2f}".format(r_bar), xy=(len(groups)-1+0.1, r_bar))
##             
##    plt.plot(groups, [LCL]*len(groups), "r")
##    plt.annotate(s="LCL={:.2f}".format(LCL), xy=(len(groups)-1+0.1, LCL))
##    
##    ax.spines['top'].set_visible(False)
##    ax.spines['right'].set_visible(False)
##    plt.xticks(rotation=60)
##    plt.ylabel("R     ", rotation=0)
##    plt.title("R Chart\n")
##
##    if save:
##        plt.savefig("output__r_chart.png")
##
##    if show:
##        plt.show()
##
##
##def group_scattering(df, y_label="Measure", save=True, show=False):
##    groups = df.index
##    n = df.shape[1]
##
##    plt.figure(figsize=(WIDTH, HEIGHT))
##    
##    for group in groups:
##        plt.scatter([group]*n, df.loc[group, :], c="blue")
##
##    plt.xticks(rotation=60)
##    plt.ylabel(y_label)
##
##    if save:
##        plt.savefig("output__scatter.png")
##
##    if show:
##        plt.show()
##
##
##def moving_range_chart(df, save=True, show=False):
##    means = get_means(df)
##    MRs = np.abs(means[1:].values-means[:-1].values)
##    indices = [str(x+2) for x in range(len(MRs))]
##    
##    MR_bar = np.mean(MRs)
##    
##    D4 = get_constant(2, "D4")
##    UCL = D4 * MR_bar
##    LCL = 0
##
##    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
##    ax.plot(indices, MRs, marker="o")
##    
##    plt.plot(indices, [UCL]*len(MRs), "r")
##    plt.annotate(s="UCL={:.2f}".format(UCL), xy=(len(MRs)-1+0.1, UCL))
##    
##    plt.plot(indices, [MR_bar]*len(MRs), "k")
##    plt.annotate(s="MR-bar={:.2f}".format(MR_bar), xy=(len(MRs)-1+0.1, MR_bar))
##    
##    plt.plot(indices, [LCL]*len(MRs), "r")
##    plt.annotate(s="LCL={:.2f}".format(LCL), xy=(len(MRs)-1+0.1, LCL))
##    
##    ax.spines['top'].set_visible(False)
##    ax.spines['right'].set_visible(False)
##    plt.ylabel("Moving Range")
##    plt.title("Subgroup MR Chart\n")
##
##    if save:
##        plt.savefig("output__mr_chart.png")
##
##    if show:
##        plt.show()
##
##
##def capability_histogram(df, x_label="Measure", bins=10, LSL=None, USL=None,
##                         save=True, show=False):
##    values = df.values.ravel()
##    mean = np.mean(values)
##    std = np.std(values, ddof=1)
##    
##    fig = plt.figure(figsize=(WIDTH, HEIGHT))
##    ax1 = plt.axes()
##    
##    ax1.hist(values, bins=bins, edgecolor="k")
##    
##    # Drawing vertical lines
##    if USL is not None:
##        plt.axvline(x=USL, c="red", label="USL={:.2f}".format(USL))
####        plt.annotate(s="USL={:.2f}".format(USL), xy=(USL+0.01, 5), rotation=90)
##    
##    ax1.axvline(x=mean+3*std, c="black",
##                label="mean+3s={:.2f}".format(mean+3*std))
##    
##    ax1.axvline(x=mean-3*std, c="black",
##                label="mean-3s={:.2f}".format(mean-3*std))
##    
##    if LSL is not None:
##        ax1.axvline(x=LSL, c="red", label="LSL={:.2f}".format(LSL))
##        
##    # Probability density function
##    def f(x):
##        coef = 1/np.sqrt(2*np.pi*(std**2))
##        power = -(x-mean)**2 / (2*(std**2))
##        return coef * np.exp(power)
##    
##    Xs = np.arange(mean-4*std, mean+4*std, 0.01)
##    Ys = f(Xs)
##    
##    ax2 = ax1.twinx()
##    ax2.plot(Xs, Ys, c="green")
##    
##    ax1.set_xlabel(x_label)
##    ax1.set_ylabel("Count")
##    ax1.legend(loc="upper right", fancybox=True, framealpha=1)
##    ax1.set_title("Capability Histogram")
##    ax2.set_ylabel("Density Function")
##
##    if save:
##        plt.savefig("output__histogram.png")
##
##    if show:
##        plt.show()
##
##
##def probability_plot(df, distribution="norm", save=True, show=False):
##    values = df.values.ravel()
##    
##    fig = plt.figure(figsize=(WIDTH, HEIGHT))
##    ax = plt.axes()
##    stats.probplot(values, dist=distribution, plot=ax)
##    
##    if save:
##        plt.savefig("output__prob_plot.png")
##        
##    if show:
##        plt.show()
##    
##
##def normality_test(df):
##    values = df.values.ravel()
##    
##    test_results = stats.anderson(values, dist="norm")
##    AD = test_results[0]
##    crit_val = test_results[1][2]
##    
##    with open("output__normality_test.txt", "w") as f:
##        f.write("Anderson-Darling (A-D) Test for Normality\n")
##        f.write("   H0: The data are normally distributed.\n")
##        f.write("   HA: The data are not normally distributed.\n")
##        f.write("   H0 is rejected if A-D test static > A-D critical value.\n\n")
##        f.write("Results\n")
##        f.write("   Significance level: alpha = 0.05\n")
##        f.write("   A-D test static = {:.4f}\n".format(AD))
##        f.write("   A-D critical value = {:.4f}".format(crit_val))
##    
##
##def performance_indices(df, LSL, USL):
##    values = df.values.ravel()
##    mean = np.mean(values)
##    std = np.std(values, ddof=1)
##    
##    Pp = abs(USL-LSL) / (6*std)
##    Ppk = min(abs(LSL-mean), abs(USL-mean)) / (3*std)
##    
##    return Pp, Ppk
##
##
##def capability_indices(df, LSL, USL,
##                       estimation_method="std"):
##    means = get_means(df)
##    X_bar = np.mean(means)
##    n = df.shape[1]
##    
##    if estimation_method == "range":
##        ranges = get_ranges(df)
##        r_bar = np.mean(ranges)
##        d2 = get_constant(n, "d2")
##        estimated_s = r_bar / d2
##        
##    elif estimation_method == "std":
##        stds = get_stds(df)
##        s_bar = np.mean(stds)
##        c4 = get_constant(n, "c4")
##        estimated_s = s_bar / c4
##    
##    else:
##        raise ValueError('estimation_method should be "std" or "range"')
##        
##    Cp = abs(USL-LSL) / (6*estimated_s)
##    Cpk = min(abs(USL-X_bar), abs(LSL-X_bar)) / (3*estimated_s)
##    
##    return Cp, Cpk
##
##
##def output_indices(df, LSL, USL, estimation_method="std"):
##    Cp, Cpk = capability_indices(df, LSL, USL,
##                  estimation_method=estimation_method)
##    Pp, Ppk = performance_indices(df, LSL, USL)
##
##    with open("output__indices.txt", "w") as f:
##        f.write("Capability Indices:\n")
##        f.write("    Cp  = {:.2f}\n".format(Cp))
##        f.write("    Cpk = {:.2f}\n\n".format(Cpk))
##        f.write("Performance Indices:\n")
##        f.write("    Pp  = {:.2f}\n".format(Pp))
##        f.write("    Ppk = {:.2f}".format(Ppk))