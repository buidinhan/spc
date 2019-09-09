import pandas as pd
import matplotlib.pyplot as plt

from spc import *


# Preparation
# Observations
data_path = "input.csv"
df = pd.read_csv(data_path, header=None)
add_labels(df)
all_values = df.values.ravel()

# Specification limits
spec_path = "spec.csv"
spec_df = pd.read_csv(spec_path)
LSL = spec_df.loc[0, "LSL"]
USL = spec_df.loc[0, "USL"]


# Plotting
plt.style.use("seaborn-white")
run_chart(all_values, centerline=True, LSL=LSL, USL=USL)
xbar_s_chart(df)
s_chart(df)
xbar_r_chart(df)
r_chart(df)
moving_range_chart(df)
capability_histogram(df, LSL=LSL, USL=USL)
probability_plot(df)


# Calculation
normality_test(df)
output_indices(df, LSL, USL)
