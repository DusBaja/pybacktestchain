import yfinance as yf
import pandas as pd 
from sec_cik_mapper import StockMapper
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging 
from scipy.optimize import minimize
import numpy as np
import os ##added
from glob import glob##added
from scipy.interpolate import interp1d ##added
import requests
import subprocess
import time
import json
import pandas as pd
from data_module import start_ngrok,get_data_api,get_volatility_from_api,get_stock_data,get_stocks_data,get_volatility,get_index_data_vol
ngrok_url = start_ngrok()
if ngrok_url:
    print(f"ngrok is running at: {ngrok_url}")
    
    # Define ticker, date range, and base URL
    selected_ticker = "^GSPC"  # Choose ticker "^STOXX50E" for Euro Stoxx 50
    selected_start_date = "2024-10-01"
    selected_end_date = "2024-10-20"
    
    # Call the function with the API URL and data range
    result_df = get_index_data_vol(selected_ticker, selected_start_date, selected_end_date, 1, ngrok_url)
    
    # Display the result
    print(result_df)
else:
    print("Could not start ngrok or fetch the URL.")