#%%
import yfinance as yf
import pandas as pd 
from sec_cik_mapper import StockMapper
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging 
from scipy.optimize import minimize
import numpy as np
import os
from glob import glob

# Setup logging
logging.basicConfig(level=logging.INFO)

#---------------------------------------------------------
# Constants
#---------------------------------------------------------

UNIVERSE_SEC = list(StockMapper().ticker_to_cik.keys())
UNIVERSE_SEC.extend(["^GSPC", "^STOXX50E"])

#---------------------------------------------------------
# Functions
#---------------------------------------------------------

# function that retrieves historical data on prices for a given stock
def get_stock_data(ticker, start_date, end_date):
    """get_stock_data retrieves historical data on prices for a given stock

    Args:
        ticker (str): The stock ticker
        start_date (str): Start date in the format 'YYYY-MM-DD'
        end_date (str): End date in the format 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: A pandas dataframe with the historical data

    Example:
        df = get_stock_data('AAPL', '2000-01-01', '2020-12-31')
    """
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
    # as dataframe 
    df = pd.DataFrame(data)
    df['ticker'] = ticker
    df.reset_index(inplace=True)
    return df

def get_stocks_data(tickers, start_date, end_date):
    """get_stocks_data retrieves historical data on prices for a list of stocks

    Args:
        tickers (list): List of stock tickers
        start_date (str): Start date in the format 'YYYY-MM-DD'
        end_date (str): End date in the format 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: A pandas dataframe with the historical data

    Example:
        df = get_stocks_data(['AAPL', 'MSFT'], '2000-01-01', '2020-12-31')
    """
    # get the data for each stock
    # try/except to avoid errors when a stock is not found
    dfs = []
    for ticker in tickers:
        try:
            df = get_stock_data(ticker, start_date, end_date)
            # append if not empty
            if not df.empty:
                dfs.append(df)
        except:
            logging.warning(f"Stock {ticker} not found")
    # concatenate all dataframes
    data = pd.concat(dfs)
    return data

# function that retrieves historical data on prices and ATM implied for a given index
def get_atm_volatility(vol_surface_df, index_price):
    """
    Get the ATM volatility for a given index price by interpolating between the closest strikes.

    Parameters:
        vol_surface_df (pd.DataFrame): The volatility surface dataframe with strikes as columns.
        index_price (float): The current index price to approximate the ATM volatility.

    Returns:
        atm_volatility (float): Interpolated linearly ATM volatility .
    """
    vol_surface_df.index = pd.to_numeric(vol_surface_df.index, errors='coerce')
    days_to_expiry = pd.Series(vol_surface_df.index, index=vol_surface_df.index).astype(float)
    
    # Find the expiry row closest to the 1M expiry (20 days)
    closest_expiry_idx = (days_to_expiry - 21).abs().idxmin()
    closest_expiry_row = vol_surface_df.loc[closest_expiry_idx]

    strikes = vol_surface_df.columns[1:].astype(float)  # The first column is for the expiries
    lower_strike = strikes[strikes <= index_price].max()
    upper_strike = strikes[strikes >= index_price].min()
    
    if pd.isna(lower_strike) or pd.isna(upper_strike):
        raise ValueError("Index price is outside the range of available strikes.")
    
    vol_lower = closest_expiry_row[lower_strike]
    vol_upper = closest_expiry_row[upper_strike]
    
    if lower_strike == upper_strike:
        atm_volatility = vol_lower
    else:
        atm_volatility = vol_lower + (index_price - lower_strike) / (upper_strike - lower_strike) * (vol_upper - vol_lower)
    
    return atm_volatility


def get_index_data_atm(ticker, start_date, end_date):
    """
    Retrieves historical index data and appends ATM volatility data.

    Parameters:
        ticker (str): The ticker symbol for the index (e.g., '^GSPC' or '^STOXX50E').
        start_date (str): Start date for the historical data.
        end_date (str): End date for the historical data.

    Returns:
        pd.DataFrame: DataFrame with historical data and ATM volatility for each date.
    """
    if ticker == "^GSPC" or ticker == "^STOXX50E":
        index = yf.Ticker(ticker)
        data = index.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
        df = pd.DataFrame(data)
        df['ticker'] = ticker
        df.reset_index(inplace=True)
        
        folder_path = os.path.join(os.getcwd(), "Data_treated")
        
        for date in df['Date']:
            date_str = date.strftime('%Y-%m-%d')
            
            file_pattern = os.path.join(folder_path, f"vol_surface_{'S&P 500' if ticker == '^GSPC' else 'Euro Stoxx 50'}_{date_str}_*.xlsx")
            files = glob(file_pattern)
            
            if files:
                
                file_path = files[0]
                vol_surface_df = pd.read_excel(file_path)
                index_price = df.loc[df['Date'] == date, 'Close'].values[0]
                atm_volatility = get_atm_volatility(vol_surface_df, index_price)
                df.loc[df['Date'] == date, 'ATM vol for the close'] = atm_volatility

            else:
                
                print(f"No volatility surface file found for date: {date_str}")
                df.loc[df['Date'] == date, 'ATM vol for the close'] = None

    else: 
        raise ValueError("The index ticker you provided is not available in our database. Please choose '^GSPC' or '^STOXX50E'")

    return df


# I need to get the data constrain for this function !!!!!!!!!
ticker = '^GSPC'  # Example ticker symbol
start_date = '2024-10-01'
end_date = '2024-10-20'
result_df = get_index_data_atm(ticker, start_date, end_date)
print(result_df)


#---------------------------------------------------------
# Classes 
#---------------------------------------------------------

# Class that represents the data used in the backtest. 
@dataclass
class DataModule:
    data: pd.DataFrame

# Interface for the information set 
@dataclass
class Information:
    s: timedelta = timedelta(days=360) # Time step (rolling window)
    data_module: DataModule = None # Data module
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column: str = 'Close'

    def slice_data(self, t : datetime):
        # Get the data module 
        data = self.data_module.data
        # Get the time step 
        s = self.s

        # Convert both `t` and the data column to timezone-aware, if needed
        if t.tzinfo is not None:
            # If `t` is timezone-aware, make sure data is also timezone-aware
            data[self.time_column] = pd.to_datetime(data[self.time_column]).dt.tz_localize(t.tzinfo.zone, ambiguous='NaT', nonexistent='NaT')
        else:
            # If `t` is timezone-naive, ensure the data is timezone-naive as well
            data[self.time_column] = pd.to_datetime(data[self.time_column]).dt.tz_localize(None)
        
        # Get the data only between t-s and t
        data = data[(data[self.time_column] >= t - s) & (data[self.time_column] < t)]
        return data

    def get_prices(self, t : datetime):
        # gets the prices at which the portfolio will be rebalanced at time t 
        data = self.slice_data(t)
        
        # get the last price for each company
        prices = data.groupby(self.company_column)[self.adj_close_column].last()
        # to dict, ticker as key price as value 
        prices = prices.to_dict()
        return prices

    def compute_information(self, t : datetime):  
        pass

    def compute_portfolio(self, t : datetime,  information_set : dict):
        pass

       
        
@dataclass
class FirstTwoMoments(Information):
    def compute_portfolio(self, t:datetime, information_set):
        try:
            mu = information_set['expected_return']
            Sigma = information_set['covariance_matrix']
            gamma = 1 # risk aversion parameter
            n = len(mu)
            # objective function
            obj = lambda x: -x.dot(mu) + gamma/2 * x.dot(Sigma).dot(x)
            # constraints
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            # bounds, allow short selling, +- inf 
            bounds = [(0.0, 1.0)] * n
            # initial guess, equal weights
            x0 = np.ones(n) / n
            # minimize
            res = minimize(obj, x0, constraints=cons, bounds=bounds)

            # prepare dictionary 
            portfolio = {k: None for k in information_set['companies']}

            # if converged update
            if res.success:
                for i, company in enumerate(information_set['companies']):
                    portfolio[company] = res.x[i]
            else:
                raise Exception("Optimization did not converge")

            return portfolio
        except Exception as e:
            # if something goes wrong return an equal weight portfolio but let the user know 
            logging.warning("Error computing portfolio, returning equal weight portfolio")
            logging.warning(e)
            return {k: 1/len(information_set['companies']) for k in information_set['companies']}

    def compute_information(self, t : datetime):
        # Get the data module 
        data = self.slice_data(t)
        # the information set will be a dictionary with the data
        information_set = {}

        # sort data by ticker and date
        data = data.sort_values(by=[self.company_column, self.time_column])

        # expected return per company
        data['return'] =  data.groupby(self.company_column)[self.adj_close_column].pct_change() #.mean()
        
        # expected return by company 
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # covariance matrix

        # 1. pivot the data
        data = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column)
        # drop missing values
        data = data.dropna(axis=0)
        # 2. compute the covariance matrix
        covariance_matrix = data.cov()
        # convert to numpy matrix 
        covariance_matrix = covariance_matrix.to_numpy()
        # add to the information set
        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data.columns.to_numpy()
        return information_set


        







# %%
