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
                                            "right": 0.87,
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

    show_and_save_plot(show=show, save=save,
                       filename="run_chart.png")


def plot_xbar_s_chart(df, group_axis="row", x_label=None, y_label=None,
                      title=None, ax=None, show=True, save=False,
                      **kwargs):
    
    means, X_bar, _, LCL, UCL = xbar_s_chart(df, group_axis=group_axis)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6),
                               gridspec_kw={"left": 0.07,
                                            "right": 0.87,
                                            "bottom": 0.1,
                                            "top": 0.94})

    group_indices = [x+1 for x in range(len(means))]
    group_labels = [str(i) for i in group_indices]
    ax.plot(group_indices, means, **kwargs)
    ax.set_xticks(group_indices)
    ax.set_xticklabels(group_labels)

    ax.axhline(y=X_bar)
    ax.annotate("Mean={:.3f}".format(X_bar), xy=(1.01, X_bar),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=LCL)
    ax.annotate("LCL={:.3f}".format(LCL), xy=(1.01, LCL),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=UCL)
    ax.annotate("UCL={:.3f}".format(UCL), xy=(1.01, UCL),
                xycoords=("axes fraction", "data"))

    ax.set_title(title)

    if x_label is None:
        x_label = "Group No."

    ax.set_xlabel(x_label)

    if y_label is None:
        y_label = "Group Mean"

    ax.set_ylabel(y_label)

    show_and_save_plot(show=show, save=save,
                       filename="xbar_s_chart.png")


def plot_xbar_r_chart(df, group_axis="row", x_label=None, y_label=None,
                      title=None, ax=None, show=True, save=False,
                      **kwargs):
    
    means, X_bar, _, LCL, UCL = xbar_r_chart(df, group_axis=group_axis)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6),
                               gridspec_kw={"left": 0.07,
                                            "right": 0.87,
                                            "bottom": 0.1,
                                            "top": 0.94})

    group_indices = [x+1 for x in range(len(means))]
    group_labels = [str(i) for i in group_indices]
    ax.plot(group_indices, means, **kwargs)
    ax.set_xticks(group_indices)
    ax.set_xticklabels(group_labels)

    ax.axhline(y=X_bar)
    ax.annotate("Mean={:.3f}".format(X_bar), xy=(1.01, X_bar),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=LCL)
    ax.annotate("LCL={:.3f}".format(LCL), xy=(1.01, LCL),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=UCL)
    ax.annotate("UCL={:.3f}".format(UCL), xy=(1.01, UCL),
                xycoords=("axes fraction", "data"))

    ax.set_title(title)

    if x_label is None:
        x_label = "Group No."

    ax.set_xlabel(x_label)

    if y_label is None:
        y_label = "Group Mean"

    ax.set_ylabel(y_label)

    show_and_save_plot(show=show, save=save,
                       filename="xbar_r_chart.png")


def plot_r_chart(df, group_axis="row", x_label=None, y_label=None,
                 title=None, ax=None, show=True, save=False, **kwargs):
    
    ranges, r_bar, LCL, UCL = r_chart(df, group_axis=group_axis)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6),
                               gridspec_kw={"left": 0.07,
                                            "right": 0.87,
                                            "bottom": 0.1,
                                            "top": 0.94})

    group_indices = [x+1 for x in range(len(ranges))]
    group_labels = [str(i) for i in group_indices]
    ax.plot(group_indices, ranges, **kwargs)
    ax.set_xticks(group_indices)
    ax.set_xticklabels(group_labels)

    ax.axhline(y=r_bar)
    ax.annotate("Mean={:.3f}".format(r_bar), xy=(1.01, r_bar),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=LCL)
    ax.annotate("LCL={:.3f}".format(LCL), xy=(1.01, LCL),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=UCL)
    ax.annotate("UCL={:.3f}".format(UCL), xy=(1.01, UCL),
                xycoords=("axes fraction", "data"))

    ax.set_title(title)

    if x_label is None:
        x_label = "Group No."

    ax.set_xlabel(x_label)

    if y_label is None:
        y_label = "Group Range"

    ax.set_ylabel(y_label)

    show_and_save_plot(show=show, save=save, filename="r_chart.png")


def plot_s_chart(df, group_axis="row", x_label=None, y_label=None,
                 title=None, ax=None, show=True, save=False, **kwargs):
    
    stds, s_bar, LCL, UCL = s_chart(df, group_axis=group_axis)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6),
                               gridspec_kw={"left": 0.07,
                                            "right": 0.87,
                                            "bottom": 0.1,
                                            "top": 0.94})

    group_indices = [x+1 for x in range(len(stds))]
    group_labels = [str(i) for i in group_indices]
    ax.plot(group_indices, stds, **kwargs)
    ax.set_xticks(group_indices)
    ax.set_xticklabels(group_labels)

    ax.axhline(y=s_bar)
    ax.annotate("Mean={:.3f}".format(s_bar), xy=(1.01, s_bar),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=LCL)
    ax.annotate("LCL={:.3f}".format(LCL), xy=(1.01, LCL),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=UCL)
    ax.annotate("UCL={:.3f}".format(UCL), xy=(1.01, UCL),
                xycoords=("axes fraction", "data"))

    ax.set_title(title)

    if x_label is None:
        x_label = "Group No."

    ax.set_xlabel(x_label)

    if y_label is None:
        y_label = "Group Std."

    ax.set_ylabel(y_label)

    show_and_save_plot(show=show, save=save, filename="s_chart.png")


def plot_moving_range_chart(df, group_axis="row", x_label=None,
                            y_label=None, title=None, ax=None,
                            show=True, save=False, **kwargs):
    
    MRs, MR_bar, LCL, UCL = moving_range_chart(df, group_axis=group_axis)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6),
                               gridspec_kw={"left": 0.07,
                                            "right": 0.87,
                                            "bottom": 0.1,
                                            "top": 0.94})

    group_indices = [x+2 for x in range(len(MRs))]
    group_labels = [str(i) for i in group_indices]
    ax.plot(group_indices, MRs, **kwargs)
    ax.set_xticks(group_indices)
    ax.set_xticklabels(group_labels)

    ax.axhline(y=MR_bar)
    ax.annotate("Mean={:.3f}".format(MR_bar), xy=(1.01, MR_bar),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=LCL)
    ax.annotate("LCL={:.3f}".format(LCL), xy=(1.01, LCL),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=UCL)
    ax.annotate("UCL={:.3f}".format(UCL), xy=(1.01, UCL),
                xycoords=("axes fraction", "data"))

    ax.set_title(title)

    if x_label is None:
        x_label = "Group No."

    ax.set_xlabel(x_label)

    if y_label is None:
        y_label = "Moving Range"

    ax.set_ylabel(y_label)

    show_and_save_plot(show=show, save=save,
                       filename="moving_range_chart.png")


def plot_p_chart(defect_col, total_col, df, x_label=None, y_label=None,
                 title=None, ax=None, show=True, save=False, **kwargs):

    proportions, p_bar, LCL, UCL = p_chart(defect_col, total_col, df)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6),
                               gridspec_kw={"left": 0.07,
                                            "right": 0.87,
                                            "bottom": 0.1,
                                            "top": 0.94})

    group_indices = [x+1 for x in range(len(proportions))]
    group_labels = [str(i) for i in group_indices]
    ax.plot(group_indices, proportions, **kwargs)
    ax.set_xticks(group_indices)
    ax.set_xticklabels(group_labels)

    ax.axhline(y=p_bar)
    ax.annotate("p-bar={:.3f}".format(p_bar), xy=(1.01, p_bar),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=LCL)
    ax.annotate("LCL={:.3f}".format(LCL), xy=(1.01, LCL),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=UCL)
    ax.annotate("UCL={:.3f}".format(UCL), xy=(1.01, UCL),
                xycoords=("axes fraction", "data"))

    ax.set_title(title)

    if x_label is None:
        x_label = "Group No."

    ax.set_xlabel(x_label)

    if y_label is None:
        y_label = "Proportion"

    ax.set_ylabel(y_label)

    show_and_save_plot(show=show, save=save, filename="p_chart.png")


def plot_single_measure_control_chart(series, x_label=None,
                                      y_label=None, title=None,
                                      ax=None, show=True, save=False,
                                      **kwargs):
    
    _, x_bar, LCL, UCL = single_measure_control_chart(series)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6),
                               gridspec_kw={"left": 0.07,
                                            "right": 0.87,
                                            "bottom": 0.1,
                                            "top": 0.94})

    indices = [x+1 for x in range(len(series))]
    labels = [str(i) for i in indices]
    ax.plot(indices, series, **kwargs)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels)

    ax.axhline(y=x_bar)
    ax.annotate("Mean={:.3f}".format(x_bar), xy=(1.01, x_bar),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=LCL)
    ax.annotate("LCL={:.3f}".format(LCL), xy=(1.01, LCL),
                xycoords=("axes fraction", "data"))

    ax.axhline(y=UCL)
    ax.annotate("UCL={:.3f}".format(UCL), xy=(1.01, UCL),
                xycoords=("axes fraction", "data"))

    ax.set_title(title)

    if x_label is None:
        x_label = "No."

    ax.set_xlabel(x_label)

    if y_label is None:
        y_label = "Measure"

    ax.set_ylabel(y_label)

    show_and_save_plot(show=show, save=save,
                       filename="single_measure_control_chart.png")


# TESTING
def test_plot_run_chart():
    df, target, LSL, USL = load_input("../testing/fridge.dat")
    all_values = df.to_numpy().ravel()
    plot_run_chart(all_values, LSL, USL, y_label="Thickness",
                   title="Run Chart", marker="o")


def test_plot_xbar_s_chart():
    df, _, _, _ = load_input("../testing/fridge.dat")
    plot_xbar_s_chart(df, group_axis="row", x_label=None, y_label=None,
                      title="X-bar (S) Chart")


def test_plot_xbar_r_chart():
    df, _, _, _ = load_input("../testing/fridge.dat")
    plot_xbar_r_chart(df, group_axis="row", x_label=None, y_label=None,
                      title="X-bar (R) Chart", marker="o", c="g")


def test_plot_r_chart():
    df, _, _, _ = load_input("../testing/fridge.dat")
    plot_r_chart(df, group_axis="row", x_label=None, y_label=None,
                 title="R Chart")


def test_plot_s_chart():
    df, _, _, _ = load_input("../testing/fridge.dat")
    plot_s_chart(df, group_axis="row", x_label=None, y_label=None,
                 title="S Chart")


def test_plot_moving_range_chart():
    df, _, _, _ = load_input("../testing/fridge.dat")
    plot_moving_range_chart(df, group_axis="row", x_label=None,
                            y_label=None, title="MR Chart", ax=None)


def test_plot_p_chart():
    df = pd.read_csv("../testing/defects.csv")
    plot_p_chart("num_of_reworks", "total_production", df,
                 title="p Chart")


def test_plot_single_measure_control_chart():
    df = pd.read_csv("../testing/complaints.csv")
    plot_single_measure_control_chart(df["number_of_complaints"],
                                      title="Control Chart",
                                      marker=".")
                                      

if __name__ == "__main__":
    test_plot_xbar_s_chart()


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
