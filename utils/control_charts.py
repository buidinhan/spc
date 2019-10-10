import pandas as pd
import numpy as np


# CALCULATIONS FOR DRAWING OF CONTROL CHARTS
CONSTANTS = pd.read_csv("constants.csv", index_col="n")


def load_input(path):
    with open(path, "r") as f:
            target = float(f.readline().split("=")[-1])
            LSL = float(f.readline().split("=")[-1])
            USL = float(f.readline().split("=")[-1])

    df = pd.read_csv(path, skiprows=5, header=None,
                     delim_whitespace=True)

    return df, target, LSL, USL
                     

def get_constant(n, c_name):
    return CONSTANTS.loc[n, c_name]


def xbar_s_chart(df, group_axis="row"):
    if group_axis == "row":
        n_groups = df.shape[0]
        n_repeats = df.shape[1]
        means_by_group = df.mean(axis=1)
        stds_by_group = df.std(axis=1, ddof=1)
    elif group_axis == "column":
        n_groups = df.shape[1]
        n_repeats = df.shape[0]
        means_by_group = df.mean(axis=0)
        stds_by_group = df.std(axis=0, ddof=1)
    else:
        raise ValueError("group_axis should be 'row' or 'column'.")

    X_bar = np.mean(means_by_group)
    s_bar = np.mean(stds_by_group)

    A3 = get_constant(n_repeats, "A3")
    UCL = X_bar + A3*s_bar
    LCL = X_bar - A3*s_bar
    
    return means_by_group, X_bar, LCL, UCL


def xbar_r_chart(df, group_axis="row"):
    if group_axis == "row":
        n_groups = df.shape[0]
        n_repeats = df.shape[1]
        means_by_group = df.mean(axis=1)
        ranges_by_group = df.max(axis=1) - df.min(axis=1)
    elif group_axis == "column":
        n_groups = df.shape[1]
        n_repeats = df.shape[0]
        means_by_group = df.mean(axis=0)
        ranges_by_group = df.max(axis=0) - df.min(axis=0)
    else:
        raise ValueError("group_axis should be 'row' or 'column'.")

    X_bar = np.mean(means_by_group)
    r_bar = np.mean(ranges_by_group)

    A2 = get_constant(n_repeats, "A2")
    UCL = X_bar + A2*r_bar
    LCL = X_bar - A2*r_bar
    
    return means_by_group, X_bar, LCL, UCL


def r_chart(df, group_axis="row"):
    if group_axis == "row":
        n_repeats = df.shape[1]
        ranges_by_group = df.max(axis=1) - df.min(axis=1)
    elif group_axis == "column":
        n_repeats = df.shape[0]
        ranges_by_group = df.max(axis=0) - df.min(axis=0)
    else:
        raise ValueError("group_axis should be 'row' or 'column'.")

    r_bar = np.mean(ranges_by_group)
    
    D4 = get_constant(n_repeats, "D4")
    D3 = get_constant(n_repeats, "D3")
    UCL = D4 * r_bar
    LCL = D3 * r_bar
    
    return ranges_by_group, r_bar, LCL, UCL


def s_chart(df, group_axis="row"):
    if group_axis == "row":
        n_repeats = df.shape[1]
        stds_by_group = df.std(axis=1, ddof=1)
    elif group_axis == "column":
        n_repeats = df.shape[0]
        stds_by_group = df.std(axis=0, ddof=1)
    else:
        raise ValueError("group_axis should be 'row' or 'column'.")

    s_bar = np.mean(stds_by_group)
    B4 = get_constant(n_repeats, "B4")
    B3 = get_constant(n_repeats, "B3")
    UCL = B4 * s_bar
    LCL = B3 * s_bar
    
    return stds_by_group, s_bar, LCL, UCL


def moving_range_chart(df, group_axis="row"):
    if group_axis == "row":
        n_groups = df.shape[0]
        n_repeats = df.shape[1]
        means_by_group = df.mean(axis=1)
    elif group_axis == "column":
        n_groups = df.shape[1]
        n_repeats = df.shape[0]
        means_by_group = df.mean(axis=0)
    else:
        raise ValueError("group_axis should be 'row' or 'column'.")

    MRs_by_group = np.abs(means_by_group[1:].values - \
                          means_by_group[:-1].values)
    MR_bar = np.mean(MRs_by_group)

    D4 = get_constant(2, "D4")
    UCL = D4 * MR_bar
    LCL = 0    
    
    return MRs_by_group, MR_bar, LCL, UCL


def p_chart(series):
    # TO BE IMPLEMENTED


def single_measure_control_chart(series):
    # TO BE IMPLEMENTED


# TESTING
def test_load_input():
    df, target, LSL, USL = load_input("../testing/fridge.dat")
    print(df, end="\n")
    print(target, LSL, USL)


def test_xbar_s_chart():
    df, _, _, _ = load_input("../testing/fridge.dat")
    means, X_bar, LCL, UCL = xbar_s_chart(df, group_axis="row")
    print(means)
    print(X_bar, LCL, UCL)


def test_xbar_r_chart():
    df, _, _, _ = load_input("../testing/fridge.dat")
    means, X_bar, LCL, UCL = xbar_r_chart(df, group_axis="row")
    print(means)
    print(X_bar, LCL, UCL)


def test_r_chart():
    df, _, _, _ = load_input("../testing/fridge.dat")
    ranges, r_bar, LCL, UCL = r_chart(df, group_axis="row")
    print(r_bar, LCL, UCL)


def test_s_chart():
    df, _, _, _ = load_input("../testing/fridge.dat")
    stds, s_bar, LCL, UCL = s_chart(df, group_axis="row")
    print(s_bar, LCL, UCL)


def test_moving_range_chart():
    df, _, _, _ = load_input("../testing/fridge.dat")
    MRs, MR_bar, LCL, UCL = moving_range_chart(df, group_axis="column")
    print(MR_bar, LCL, UCL)
    

def test_p_chart(series):
    # TO BE IMPLEMENTED


def test_single_measure_control_chart(series):
    # TO BE IMPLEMENTED

    
if __name__ == "__main__":
    test_moving_range_chart()
