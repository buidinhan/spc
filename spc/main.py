import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def get_data(path):
    df = pd.read_csv(path, header=None)
    n_row, n_col = df.shape
    rows, cols = [], []
    for i in range(n_col):
