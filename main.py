import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ts_constituents = np.load("ts_constituents.npy")
ts_index = np.load("ts_index.npy")
n_time_poimts = ts_constituents.shape[0]
n_underlyings = ts_constituents.shape[1]

ts_gross_rets_consts = ts_constituents[1:] / ts_constituents[:-1]
ts_gross_rets_index = ts_index[1:] / ts_index[:-1]

A = np.divide(ts_constituents, ts_constituents[0])[1:]
b = (ts_index / ts_index[0])[1:]
w0_start = np.ones(shape=(n_underlyings,)) / n_underlyings

V = np.ones(shape=(n_underlyings, n_underlyings - 1)) / n_underlyings # (v_1, ..., v_n-1)
for i in range(n_underlyings - 1):
    V[i, i] = -(n_underlyings - 1) / n_underlyings
AV = np.matmul(A, V)

# Results
z = np.linalg.pinv(AV) @ (b - np.matmul(A, w0_start))
w0_star = w0_start + V @ z
print(w0_star)

# Checks
if not np.isclose(np.sum(w0_star), 1., atol=1e-2):
    raise Exception(f"Weighs sum up to {np.sum(w0_star)}!")

# -------- Test --------
proj_weights = [w0_star]
for i in range(n_time_poimts - 2):
    w_grown = proj_weights[-1] * ts_gross_rets_consts[i]
    proj_weights.append(w_grown / w_grown.sum())
proj_weights = np.array(proj_weights)
sim_rets = np.sum(np.multiply(proj_weights, ts_gross_rets_consts - 1), axis=-1)
index_rets = ts_gross_rets_index - 1
error = sim_rets - index_rets

df = pd.DataFrame({"Simulated returns": sim_rets, "Index returns": index_rets, "Delta": error})

print(np.corrcoef(sim_rets, index_rets)[0, 1])

print(df)
df.plot()
plt.show()