# Index performance reverse engineering

Regression of a time series on others. 

## Overview

- **Goal:** Replicate an index/asset performance with other as index/assets
- **Method:** Least-squares weight fitting with periodic rebalancing
- **Output:** Simulated returns

## Usage

```python
ts_constituents = np.load("ts_constituents.npy")
ts_index = np.load("ts_index.npy")

rebalancing_indices = np.array([40, 150, 200])
wo_stars = solve_w_rebalance(ts_constituents, ts_index, rebalancing_indices)

sim_rets = sim_rets(ts_constituents, wo_stars, rebalancing_indices)
```

## Methodology

See [documentation](Documentation.pdf).