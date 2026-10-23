import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils import *

ts_constituents = np.load("ts_constituents.npy")
ts_index = np.load("ts_index.npy")

rebalancing_indices = np.array([40, 150, 200])
wo_stars = solve_w_rebalance(ts_constituents, ts_index, rebalancing_indices)

# -------- Test & plot --------
sim_rets = sim_rets(ts_constituents, wo_stars, rebalancing_indices)
index_rets = ts_index[1:] / ts_index[:-1] - 1
error = sim_rets - index_rets
rel_error = error / index_rets
corr = np.corrcoef(sim_rets, index_rets)[0, 1]
mse = np.mean(error**2)

df = pd.DataFrame({"Simulated returns": sim_rets, "Index returns": index_rets, "Delta": error, "Relative error": rel_error})
print(f"Correlation: {corr}\tMSE: {mse}")
print(df)

plot(sim_rets, index_rets, rebalancing_indices)
