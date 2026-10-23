import numpy as np
import matplotlib.pyplot as plt

# ----- Optimization problem -----
def solve_w_rebalance(ts_constituents, ts_index, rebalancing_indices):
    sub_ts_consts = np.split(ts_constituents, rebalancing_indices)
    sub_ts_index = np.split(ts_index, rebalancing_indices)

    wo_stars = []
    for i in range(len(sub_ts_consts)):
        if len(sub_ts_index[i]) == 0:
            continue
        w0_star = solve_wo_rebalance(sub_ts_consts[i], sub_ts_index[i])
        wo_stars.append(w0_star)

    return wo_stars


def solve_wo_rebalance(ts_constituents, ts_index):
    n_underlyings = ts_constituents.shape[1]

    # Set up problem
    A = np.divide(ts_constituents, ts_constituents[0])[1:]
    b = (ts_index / ts_index[0])[1:]
    w0_start = np.ones(shape=(n_underlyings,)) / n_underlyings
    V = np.ones(shape=(n_underlyings, n_underlyings - 1)) / n_underlyings  # (v_1, ..., v_n-1)
    for i in range(n_underlyings - 1):
        V[i, i] = -(n_underlyings - 1) / n_underlyings
    AV = np.matmul(A, V)

    # Results
    z = np.linalg.pinv(AV) @ (b - np.matmul(A, w0_start))
    w0_star = w0_start + V @ z
    #print(w0_star)

    # Checks
    if not np.isclose(np.sum(w0_star), 1., atol=1e-2):
        raise Exception(f"Weighs sum up to {np.sum(w0_star)}!")

    return w0_star

# ----- Simulation -----
def sim_rets(ts_constituents, w0_stars, rebalancing_indices):
    n_time_pts = ts_constituents.shape[0]
    ts_gross_rets_consts = ts_constituents[1:] / ts_constituents[:-1]
    it = iter(w0_stars)
    proj_weights = [next(it)]
    for i in range(n_time_pts - 2):
        w_grown = proj_weights[-1] * ts_gross_rets_consts[i] if not np.any(rebalancing_indices == i) else next(it)
        proj_weights.append(w_grown / w_grown.sum())
    proj_weights = np.array(proj_weights)
    sim_rets = np.sum(np.multiply(proj_weights, ts_gross_rets_consts - 1), axis=-1)
    return sim_rets

# ----- Plotting -----
def plot(sim_rets, index_rets, rebalancing_indices):
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    fig.suptitle('Index Replication')
    abs_error = sim_rets - index_rets
    rel_error = abs_error / index_rets
    mse_cum = np.cumulative_sum(abs_error ** 2)

    axs[0].set_title('Simulated vs. Index Returns')
    axs[0].vlines(rebalancing_indices, np.min(sim_rets), np.max(sim_rets), color='r', linestyle='--')
    axs[0].plot(sim_rets)
    axs[0].plot(index_rets)

    axs[1].set_title('Absolute Error')
    axs[1].vlines(rebalancing_indices, np.min(abs_error), np.max(abs_error), color='r', linestyle='--')
    axs[1].plot(abs_error)

    axs[2].set_title('Relative Error')
    axs[2].vlines(rebalancing_indices, np.min(rel_error), np.max(rel_error), color='r', linestyle='--')
    axs[2].plot(rel_error)

    plt.show()