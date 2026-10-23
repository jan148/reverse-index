import yfinance as yf
import numpy as np

tickers_all_dax = [
    "ADS.DE",   # Adidas
    "AIR.DE",   # Airbus
    "ALV.DE",   # Allianz
    "BAS.DE",   # BASF
    "BAYN.DE",  # Bayer
    "BEI.DE",   # Beiersdorf
    "BMW.DE",   # BMW
    "BNR.DE",   # Brenntag
    "CBK.DE",   # Commerzbank
    "CON.DE",   # Continental
    "DTG.DE",   # Daimler Truck
    "DBK.DE",   # Deutsche Bank
    "DB1.DE",   # Deutsche Börse
    "DHL.DE",   # DHL Group
    "DTE.DE",   # Deutsche Telekom
    "EOAN.DE",  # E.ON
    "FME.DE",   # Fresenius Medical Care
    "FRE.DE",   # Fresenius
    "G1A.DE",   # GEA Group
    "HNR1.DE",  # Hannover Rück
    "HEI.DE",   # Heidelberg Materials
    "HEN3.DE",  # Henkel (pref.)
    "HOT.DE",   # Hochtief
    "IFX.DE",   # Infineon
    "MBG.DE",   # Mercedes-Benz Group
    "MRK.DE",   # Merck KGaA
    "MTX.DE",   # MTU Aero Engines
    "MUV2.DE",  # Munich Re
    "QIA.DE",   # Qiagen
    "RHM.DE",   # Rheinmetall
    "RWE.DE",   # RWE
    "SAP.DE",   # SAP
    "G24.DE",   # Scout24
    "SIE.DE",   # Siemens
    "ENR.DE",   # Siemens Energy
    "SHL.DE",   # Siemens Healthineers
    "SY1.DE",   # Symrise
    "VOW3.DE",  # Volkswagen (pref.)
    "VNA.DE",   # Vonovia
    "ZAL.DE",   # Zalando
]
tickers_top5 = [
    "SAP.DE",   # SAP
    "SIE.DE",   # Siemens
    "DTE.DE",   # Deutsche Telekom
    "AIR.DE",   # Airbus
    "ALV.DE",   # Allianz
]
tickers_top10 = [
    "SAP.DE",   # SAP              ~€192B
    "SIE.DE",   # Siemens          ~€166B
    "DTE.DE",   # Deutsche Telekom ~€159B
    "AIR.DE",   # Airbus           ~€152B
    "ALV.DE",   # Allianz          ~€139B
    "ENR.DE",   # Siemens Energy   ~€138B
    "VOW3.DE",  # Volkswagen       ~€74B
    "RHM.DE",   # Rheinmetall      ~€74B
    "MUV2.DE",  # Munich Re        ~€71B
    "DBK.DE",   # Deutsche Bank    ~€56B
]

ticker_index = ["^GDAXI"]
tickers_constituents = tickers_top5

data = yf.download(tickers_constituents + ticker_index, start="2024-01-01", end="2024-12-31", auto_adjust=True)["Close"]
data = data.dropna()

ts_constituents = data[tickers_constituents].to_numpy()
ts_index = data[ticker_index].to_numpy().squeeze()

np.save("ts_constituents.npy", ts_constituents)
np.save("ts_index.npy", ts_index)