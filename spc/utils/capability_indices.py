import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


# CAPABILITY INDICES
def Cp(series, LSL, USL):
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    
    return np.abs(USL-LSL) / (6*std)


def Cpk(series, LSL, USL):
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    
    return min(np.abs(USL-mean), np.abs(mean-LSL)) / (3*std)


def Cpm(series, LSL, USL, target):
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    
    numerator = np.abs(USL-LSL)
    denominator = 6 * np.sqrt(std**2 + (mean-target)**2)
    
    return numerator / denominator


def Cpl(series, LSL):
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    
    return np.abs(mean-LSL) / (3*std)


def Cpu(series, USL):
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    
    return np.abs(USL-mean) / (3*std)


## CONFIDENCE INTERVAL OF CAPABILITY INDICES
def Cp_CI(alpha, series, LSL, USL):
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    dof = len(series) - 1
    
    Cp = np.abs(USL-LSL) / (6*std)
    
    Cp_lower = Cp * np.sqrt(stats.chi2.ppf(alpha/2, dof)/dof)
    Cp_upper = Cp * np.sqrt(stats.chi2.ppf(1-alpha/2, dof)/dof)
    
    return Cp_lower, Cp_upper


def Cpu_CI(alpha, series, USL):
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    n = len(series)
    
    Cpu = np.abs(USL-mean) / (3*std)
    
    Cpu_lower = Cpu - stats.norm.ppf(1-alpha) * np.sqrt(1/(9*n) + Cpu**2/(2*(n-1)))
    Cpu_upper = Cpu + stats.norm.ppf(1-alpha) * np.sqrt(1/(9*n) + Cpu**2/(2*(n-1)))
    
    return Cpu_lower, Cpu_upper


def Cpl_CI(alpha, series, LSL):
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    n = len(series)
    
    Cpl = np.abs(mean-LSL) / (3*std)
    
    Cpl_lower = Cpl - stats.norm.ppf(1-alpha) * np.sqrt(1/(9*n) + Cpl**2/(2*(n-1)))
    Cpl_upper = Cpl + stats.norm.ppf(1-alpha) * np.sqrt(1/(9*n) + Cpl**2/(2*(n-1)))
    
    return Cpl_lower, Cpl_upper


def Cpk_CI(alpha, series, LSL, USL):
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    n = len(series)
    
    Cpk = min(np.abs(mean-LSL), np.abs(USL-mean)) / (3*std)
    
    Cpk_lower = Cpk - stats.norm.ppf(1-alpha/2) * np.sqrt(1/(9*n) + Cpk**2/(2*(n-1)))
    Cpk_upper = Cpk + stats.norm.ppf(1-alpha/2) * np.sqrt(1/(9*n) + Cpk**2/(2*(n-1)))
    
    return Cpk_lower, Cpk_upper


def Cpm_CI(alpha, series, USL, LSL, target):
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    n = len(series)
    
    Cpm = np.abs(USL-LSL) / (6 * np.sqrt(std**2 + (mean-target)**2))
    
    a = (mean-target) / std
    dof = n * (1+a**2)**2 / (1+2*a**2)
    
    Cpm_lower = Cpm * np.sqrt(stats.chi2.ppf(alpha/2, dof)/dof)
    Cpm_upper = Cpm * np.sqrt(stats.chi2.ppf(1-alpha/2, dof)/dof)
    
    return Cpm_lower, Cpm_upper


# TESTING
def test():
    data_file = "../testing/fridge.csv"
    df = pd.read_csv(data_file)
    series = df[["thick1",
                 "thick2",
                 "thick3",
                 "thick4",
                 "thick5"]].to_numpy().ravel()
    LSL, USL = 1.0, 4.0
    print("Cp={}, CI={}".format(
                Cp(series, LSL, USL),
                Cp_CI(0.05, series, LSL, USL)))
    print("Cpk={}, CI={}".format(
                Cpk(series, LSL, USL),
                Cpk_CI(0.05, series, LSL, USL)))


if __name__ == "__main__":
    test()
   
