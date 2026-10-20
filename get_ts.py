import yfinance as yf
import numpy as np

tickers_constituents = ["SAP.DE", "SIE.DE", "DTE.DE", "ALV.DE", "ENR.DE",
           "RHM.DE", "MUV2.DE", "EOAN.DE", "DHL.DE", "IFX.DE"]
ticker_index = ["^GDAXI"]

data = yf.download(tickers_constituents + ticker_index, start="2024-01-01", end="2024-12-31", auto_adjust=True)["Close"]

ts_constituents = data[tickers_constituents].to_numpy()
ts_index = data[ticker_index].to_numpy().squeeze()

np.save("ts_constituents.npy", ts_constituents)
np.save("ts_index.npy", ts_index)