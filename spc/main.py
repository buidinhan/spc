import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from spc import (add_labels, run_chart, xbar_s_chart,
                 s_chart, xbar_r_chart, r_chart,
                 group_scattering, moving_range_chart,
                 capability_histogram, normality_test,
                 output_indices)

# Preparation
# Observations
data_path = "input.csv"
df = pd.read_csv(path, header=None)
add_labels(df)
all_values = df.values.ravel()

# Specification limits
spec_path ="spec.csv"
spec_df = pd.read(spec_path)
LSL = spec_df.loc[0, "LSL"]
USL = spec_df.loc[0, "USL"]

# Plotting
run_chart(all_values, centerline=True,
          LSL=LSL, USL=USL)
xbar_s_chart(df)
xbar_r_chart(df)
s_chart(df)
r_chart(df)
moving_range_chart(df)
capability_histogram(df, LSL=LSL, USL=USL)

# Calculation
normality_test(df)
output_indices(df)
