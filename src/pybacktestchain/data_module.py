#%%
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
from scipy.stats import norm##added
import requests
import subprocess
import time
import json


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

def get_data_api(date, name, base_url):
    """
    Fetch data from the Flask API for a specific date and index name, returning it as a DataFrame.

    This function handles mapping specific index names, makes multiple attempts to fetch the data
    from the API in case of failures, and logs the process for troubleshooting.

    Args:
        date (str): The date for which to fetch data, in the format "YYYY-MM-DD".
        name (str): The name of the index (e.g., "^GSPC" for S&P 500 or "^STOXX50E" for Euro Stoxx 50).
        base_url (str): The base URL of the Flask API.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the fetched data. If the request fails or
                      the response is empty, an empty DataFrame is returned.

    Notes:
        - The function maps "^GSPC" to "S&P 500" and "^STOXX50E" to "Euro Stoxx 50" before making the API request.
        - It attempts to fetch data up to 3 times with exponential backoff in case of errors.
        - The function logs success, warning, and error messages for better traceability.
    """
    # Map specific index names
    if name == "^GSPC":
        name = "S&P 500"
    if name == "^STOXX50E":
        name = "Euro Stoxx 50"
    
   
    # Make the API request
    #response = requests.get(f"{base_url}/api/data", params={"date": date, "index": name})
    #data = response.json()
    #try: 
    #    df = pd.DataFrame(data)
    #    return df
    #except ValueError as e:
    #    print(f"Raw API response for {name} on {date}: {response.text}")
    #    logging.error(f"Error creating DataFrame for {name} on {date}: {e}. Data: {data}")
    #    return pd.DataFrame()
    for attempt in range(3):
        try:
            logging.info(f"Attempting to fetch data for {name} on {date} (Attempt {attempt + 1}/{3})")
            # Make the API request
            response = requests.get(f"{base_url}/api/data", params={"date": date, "index": name}, timeout=10)
            # Raise an exception for HTTP errors
            response.raise_for_status()
            # Parse the response as JSON
            data = response.json()
            df = pd.DataFrame(data)
            if not df.empty:
                logging.info(f"Successfully fetched data for {name} on {date}")
                return df
            else:
                logging.warning(f"No data returned for {name} on {date}. Returning empty DataFrame.")
                return pd.DataFrame()
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {name} on {date}: {e}")
            if attempt < 3 - 1:  # Retry only if attempts remain
                wait_time = 2 ** attempt
                logging.info(f"Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"All retry attempts failed for {name} on {date}. Returning empty DataFrame.")
                return pd.DataFrame()

        except ValueError as e:
            logging.error(f"Error processing data for {name} on {date}: {e}")
            return pd.DataFrame()

    # Fallback if all retries fail
    logging.error(f"Failed to fetch data for {name} on {date} after {3} retries.")
    return pd.DataFrame()
    
def get_volatility_from_api(date, index_name, base_url):
    """ Fetch volatility surface data from the Flask API and return it as a DataFrame """
    return get_data_api(date, index_name, base_url)

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

# function that retrieves historical data on prices and implied for a given index
def get_volatility(vol_surface_df, index_price, percentage_spot):
    """
    Get the volatility for a given index price percentage by interpolating between the closest strikes.

    Args:
        vol_surface_df (pd.DataFrame): The volatility surface dataframe with strikes as columns.
        index_price (float): The current index price to approximate the volatility.
        percentage_spot (float): Percentage of the index price to target for strike selection.

    Returns:
        float: Interpolated or extrapolated volatility. Returns NaN for invalid or non-positive values.
    """
    if vol_surface_df.empty:
        logging.warning("Volatility surface DataFrame is empty. Returning NaN.")
        return np.nan

    try:
        # Ensure index is numeric and handle invalid entries
        vol_surface_df.index = pd.to_numeric(vol_surface_df.index, errors='coerce')
        days_to_expiry = pd.Series(vol_surface_df.index).astype(float)

        # Find the closest expiry to 1M (20 days)
        closest_expiry_idx = (days_to_expiry - 21).abs().idxmin()
        closest_expiry_row = vol_surface_df.loc[closest_expiry_idx]

        # Ensure strikes are numeric
        strikes = vol_surface_df.columns[:-1].astype(float)
        target_strike = index_price * percentage_spot

        # Interpolate volatility using cubic interpolation
        cubic_interpolator = interp1d(
            strikes,
            closest_expiry_row.values[:-1],
            kind='cubic',
            fill_value="extrapolate"
        )
        volatility = cubic_interpolator(target_strike)

        # Set to NaN if the interpolated volatility is non-positive
        if volatility <= 0:
            logging.warning("Interpolated volatility is non-positive. Returning NaN.")
            return np.nan

    except KeyError as e:
        logging.error(f"KeyError during volatility calculation: {e}")
        return np.nan
    except ValueError as e:
        logging.error(f"Interpolation failed: {e}")
        return np.nan
    except Exception as e:
        logging.error(f"Unexpected error in get_volatility: {e}")
        return np.nan

    return volatility

def get_index_data_vol(ticker,start_date,end_date, percentage_spot=1, base_url=None):
    """
    Retrieves historical index data and appends ATM volatility data from an API.

    Args:
        ticker (str): The ticker symbol for the index (e.g., '^GSPC' or '^STOXX50E').
        start_date: start date for the historical data, after 2024-09-30.
        end_date: end date for the historical data 
        base_url (str): The base URL of the Flask API.

    Returns:
        pd.DataFrame: DataFrame with historical data and ATM volatility for each date.
    """
    if isinstance(start_date, str):
        start_date_ = datetime.strptime(start_date.split(" ")[0], '%Y-%m-%d').date()  # Remove time if present
    elif isinstance(start_date, datetime):
        start_date_ = start_date.date()  # Get just the date part
    if start_date_ < datetime.strptime('2024-09-30', '%Y-%m-%d').date():
        raise ValueError("Start date must be after 2024-09-30 due to data availability.")

    if ticker not in ["^GSPC", "^STOXX50E"]:
        raise ValueError("The index ticker must be either '^GSPC' or '^STOXX50E'.")

    index = yf.Ticker(ticker)
    data = index.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
    
    df = pd.DataFrame(data)
    df['ticker'] = ticker
    df.reset_index(inplace=True)
    for date in df['Date']:
        date_str = date.strftime('%Y-%m-%d')

        # Fetch volatility surface data from the API
        vol_surface_df = get_volatility_from_api(date_str, "S&P 500" if ticker == "^GSPC" else "Euro Stoxx 50", base_url)
        
        if vol_surface_df is not None and not vol_surface_df.empty:
            index_price = df.loc[df['Date'] == date, 'Close'].values[0]
            volatility = get_volatility(vol_surface_df, index_price, percentage_spot)
            
            df.loc[df['Date'] == date, 'Percentage Spot selected vol for the close'] = volatility
            
        else:
            logging.warning(f"No volatility surface data found for date: {date_str}")
            df.loc[df['Date'] == date, 'Percentage Spot selected vol for the close'] = np.nan

        
        
    return df

def get_index_data_vols(tickers,start_date,end_date, percentage_spot=1, base_url=None):
    """
    Retrieves historical index data and appends ATM volatility data from an API.

    Args:
        tickers (str): tickers symbol for the index (e.g., '^GSPC' and '^STOXX50E').
        start_date: start date for the historical data, after 2024-09-30.
        end_date: end date for the historical data 
        base_url (str): The base URL of the Flask API.

    Returns:
        pd.DataFrame: DataFrame with historical data and ATM volatility for each date.
    """
    if isinstance(start_date, str):
        start_date_ = datetime.strptime(start_date.split(" ")[0], '%Y-%m-%d').date()  # Remove time if present
    elif isinstance(start_date, datetime):
        start_date_ = start_date.date()  # Get just the date part
    if start_date_ < datetime.strptime('2024-09-30', '%Y-%m-%d').date():
        raise ValueError("Start date must be after 2024-09-30 due to data availability.")

    for ticker in tickers:
        if ticker not in ["^GSPC", "^STOXX50E"]:
            raise ValueError("The index ticker must be either '^GSPC' or '^STOXX50E'.")
    dfs = []
    for ticker in tickers: 
        try:
            index = yf.Ticker(ticker)
            data = index.history(start=start_date, end=end_date  , auto_adjust=False, actions=False)
            
            df = pd.DataFrame(data)
            df['ticker'] = ticker
            df.reset_index(inplace=True)
            for date in df['Date']:
                date_str = date.strftime('%Y-%m-%d')

                # Fetch volatility surface data from the API
                vol_surface_df = get_volatility_from_api(date_str, "S&P 500" if ticker == "^GSPC" else "Euro Stoxx 50", base_url)
                
                if vol_surface_df is not None and not vol_surface_df.empty:
                    index_price = df.loc[df['Date'] == date, 'Close'].values[0]
                    volatility = get_volatility(vol_surface_df, index_price, percentage_spot)
                    
                    df.loc[df['Date'] == date, 'Percentage Spot selected vol for the close'] = volatility
                    
                else:
                    logging.warning(f"No volatility surface data found for date: {date_str}")
                    df.loc[df['Date'] == date, 'Percentage Spot selected vol for the close'] = np.nan

                
            #print("Our data",df)
            if not df.empty:
                dfs.append(df)
        except:
            logging.warning(f"Index {ticker} not found")    
    data = pd.concat(dfs,ignore_index=True)
    
    return data

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
    vol_column: str = 'Percentage Spot selected vol for the close'# It's the implied vol'ImpliedVol'
    indices: list = None
    option_type: str = 'call'
    percentage_spot: float = 1.0
    strategy_type: str = 'cash'
    
    ####modified: 
    def slice_data(self, t: datetime):
        """
        Filters data to include only rows within the time window [t - s, t).
        Ensures consistency between tz-aware and tz-naive datetime formats.
        """
        
        data = self.data_module.data
        s = self.s

        # Convert `self.time_column` to naive datetime for uniformity and for the vol strategy to work
        data[self.time_column] = pd.to_datetime(data[self.time_column], utc=True).dt.tz_localize(None)

        # Ensure `t` is also naive
        if t.tzinfo is not None:
            t = t.replace(tzinfo=None)

        # Filter data to [t - s, t) range
        data = data[(data[self.time_column] >= (t - s)) & (data[self.time_column] < t)]
        return data

   
    @staticmethod
    def black_scholes(spot_price, strike_price, T, r, sigma, option_type='call'):
        """
        Function to compute the Black-Scholes option price.

        Args:
            spot_price (float): The current spot price of the underlying asset
            strike_price (float): The strike price of the option
            T (float): Time to expiration in years
            r (float): Risk-free interest rate (annualized)
            sigma (float): Volatility of the underlying asset (annualized)
            option_type (str): The type of option ('call' or 'put')

        Returns:
            float: The price of the option.
        """
        d1 = (np.log(spot_price / strike_price) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type.lower() == 'call':
            option_price = spot_price * norm.cdf(d1) - strike_price * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            option_price = strike_price * np.exp(-r * T) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        return option_price
    @staticmethod
    def compute_delta(spot_price, strike_price, T, r, sigma, option_type='call'):
        """
        Compute the delta of an option using the Black-Scholes model.

        Args:
            spot_price (float): The current spot price of the underlying asset.
            strike_price (float): The strike price of the option.
            T (float): Time to expiration in years.
            r (float): Risk-free interest rate (annualized).
            sigma (float): Volatility of the underlying asset (annualized).
            option_type (str): The type of option ('call' or 'put').

        Returns:
            float: Delta of the option.
        """
        d1 = (np.log(spot_price / strike_price) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type.lower() == 'call':
            return norm.cdf(d1)
        elif option_type.lower() == 'put':
            return norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    def get_prices(self, t : datetime,strategy_type: str,class_name:str):
        """
        Retrieve the prices for portfolio rebalancing at a specific time `t`.

        This method retrieves either the adjusted closing prices (for cash strategies) or option prices 
        and associated delta hedging costs (for volatility strategies). Prices are calculated differently 
        based on the selected strategy type and the class implementing the strategy.

        Args:
            t (datetime): The specific time for which prices are retrieved.
            strategy_type (str): The trading strategy, either "cash" or "vol".
            class_name (str): The name of the strategy class, either "Momentum" or "ShortSkew", used 
                            to determine the method for option price and delta hedging cost calculation.

        Returns:
            dict: 
                - For "cash" strategy: A dictionary mapping company tickers to their adjusted closing prices.
                - For "vol" strategy: A dictionary containing:
                    - Adjusted close prices (`Adj Close`).
                    - Implied volatilities (`vol`).
                    - Option prices (`Price Option`) calculated using the Black-Scholes model.
                    - Delta hedging costs (`Cost Hedging`).

        Notes:
            - For "cash" strategies, the last available adjusted closing price for each company is used.
            - For "Momentum" volatility strategies:
                - A one-month (21/365) expiration and a 105% strike (call option) are assumed.
                - Option prices and delta hedging costs are calculated using the Black-Scholes model.
            - For "ShortSkew" volatility strategies:
                - A one-month (21/365) expiration and a 95% strike (put option) are assumed.
                - Option prices and delta hedging costs are calculated using the black_scholes function.
            - Delta hedging costs are negative for the long call options (we need to short shares, so the cost are negative: it's a gain) 
                    and negative for short put options (we need to short shares, so the cost are negative: it's a gain).
            
        """
        # gets the prices at which the portfolio will be rebalanced at time t 
        data = self.slice_data(t)
        
        if strategy_type == "cash":
        # get the last price for each company
            prices = data.groupby(self.company_column)[self.adj_close_column].last()
        elif strategy_type =="vol":   
            if class_name == "Momentum":     
                data = data.groupby(self.company_column).agg({self.adj_close_column: 'last',self.vol_column:'last'})   
                #spot_price = data.groupby(self.company_column)[self.adj_close_column].last()
                #implied_vol = data.groupby(self.company_column)[self.vol_column].last() if self.vol_column in data.columns.to_list() else None
                Price_option =[]
                Cost_heging =[]
                
                for idx in range(len(data)):
                    T = 21/365  # We assume 1 month exp
                    r = 0.0315
                    self.percentage_spot=1.05
                    self.option_type="Call"  
                    K = data[self.adj_close_column].iloc[idx] * self.percentage_spot  
                    option_price = self.black_scholes(data[self.adj_close_column].iloc[idx], K, T, r, data[self.vol_column].iloc[idx], option_type=self.option_type)
                    delta = self.compute_delta(data[self.adj_close_column].iloc[idx], K, T, r, data[self.vol_column].iloc[idx], self.option_type)
                    cost_delta_hedging = -delta*data[self.adj_close_column].iloc[idx]
                    Price_option.append(option_price)
                    Cost_heging.append(cost_delta_hedging)
                prices= data.groupby(self.company_column).agg({self.adj_close_column: 'last',self.vol_column:'last'})                   
                prices["Price Option"]=Price_option
                prices["Cost Hedging"]=Cost_heging
                prices.reset_index(inplace=True)#to get the index information !!   
            elif class_name == "ShortSkew":
                data = data.groupby(self.company_column).agg({self.adj_close_column: 'last',self.vol_column:'last'})   
                Price_option =[]
                Cost_heging =[]
                
                for idx in range(len(data)):
                    T = 21/365  # We assume 1 month exp
                    r = 0.0315
                    
                    self.percentage_spot=0.95
                    self.option_type="Put"  
                    K = data[self.adj_close_column].iloc[idx] * self.percentage_spot  
                    option_price = self.black_scholes(data[self.adj_close_column].iloc[idx], K, T, r, data[self.vol_column].iloc[idx], option_type=self.option_type)
                    delta = self.compute_delta(data[self.adj_close_column].iloc[idx], K, T, r, data[self.vol_column].iloc[idx], self.option_type)
                    cost_delta_hedging = delta*data[self.adj_close_column].iloc[idx]
                    Price_option.append(option_price)
                    Cost_heging.append(cost_delta_hedging)
                prices= data.groupby(self.company_column).agg({self.adj_close_column: 'last',self.vol_column:'last'})                   
                prices["Price Option"]=Price_option
                prices["Cost Hedging"]=Cost_heging
                prices.reset_index(inplace=True)#to


            
            prices = prices.to_dict()
            return prices

    def compute_information(self, t : datetime):  
        pass

    def compute_portfolio(self, t : datetime,  information_set : dict):
        pass

             
@dataclass
class FirstTwoMoments(Information):
    """
    A class for implementing portfolio optimization using the first two moments (mean and covariance).
    This class computes the portfolio weights based on expected returns and covariance of assets (historical data).
    
    Attributes:
        strategy_type (str): The type of strategy only supportes "cash".
    """
    def compute_portfolio(self, t:datetime, information_set):
        """
        Compute portfolio weights based on expected returns and covariance matrix.

        This method performs quadratic optimization to find the portfolio weights that maximize 
        the risk-adjusted return. The optimization considers the expected returns, covariance matrix, 
        and risk aversion parameter.

        Args:
            t (datetime): The current date for which the portfolio is being computed.
            information_set (dict): A dictionary containing:
                - 'expected_return': Array of expected returns for each asset.
                - 'covariance_matrix': Covariance matrix of asset returns.
                - 'companies': List of company tickers.

        Returns:
            dict: A dictionary mapping company tickers to portfolio weights. If the optimization fails, 
                  an equal-weight portfolio is returned.

        Raises:
            Exception: If the optimization does not converge or if an error occurs during computation.

        Notes:
            - Short selling is not allowed; portfolio weights are constrained to [0.0, 1.0].
            - If the optimization fails, logs a warning and defaults to an equal-weight portfolio.
            - Risk aversion parameter (`gamma`) is set to 1 by default.
        """
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

    def compute_information(self, t : datetime,base_url=None): # I added the base_url part even if not used as it would block the code otherwise
        """
        Compute the information set required for portfolio optimization.

        This method calculates the expected returns and covariance matrix for the assets in the portfolio 
        based on historical data. The information set is stored in a dictionary and includes:
            - Expected returns per asset.
            - Covariance matrix of asset returns.
            - List of asset tickers.

        Args:
            t (datetime): The current date for which the information is being computed.
            base_url (str, optional): Base URL for additional data fetching (not used here but included for compatibility).

        Returns:
            dict: A dictionary containing:
                - 'expected_return': Array of expected returns for each asset.
                - 'covariance_matrix': Covariance matrix of asset returns.
                - 'companies': List of company tickers.

        """
        try:
            # Get the data module 
            data = self.slice_data(t)
            # the information set will be a dictionary with the data
            information_set = {}

            # sort data by ticker and date
            data = data.sort_values(by=[self.company_column, self.time_column])
            ###################
            ## modified/added: 
            if self.strategy_type == 'cash':
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
            elif self.strategy_type == 'vol':
                logging.error(f"There is no vol strategy with the FirstTwoMoments")
            #### end of the modifications 
            ############################
            return information_set
        except Exception as e:
            logging.error(f"Error computing information: {e}")
            return {}

class Momentum(Information):
    """
    Implements the Momentum strategy for portfolio construction.

    The Momentum strategy selects assets based on their past performance, focusing on those expected to 
    perform best in the future. This class supports both "cash" and "vol" strategy types. For the vol strategy
    it's implemented by going full long a call 105% strike on the best expected performer. We compute the 
    expected return directly on the implied vol: which has increased the most is selected therefore.


    Attributes:
        previous_best_performer (str): Tracks the last best-performing asset in the "vol" strategy.
        previous_position (dict): Tracks the last portfolio allocation for the "vol" strategy.
    """
    previous_best_performer: str = None  # Tracks the last best performer
    previous_position: dict = None  # Tracks the last position (index and option characteristics)

    def compute_portfolio(self, t: datetime, information_set):
        """
        Constructs a portfolio based on the selected strategy type.

        For the "cash" strategy:
        - Allocates weights proportional to the expected returns of assets.
        - If the total expected return is non-positive, it assigns equal weights to all assets.

        For the "vol" strategy:
        - Allocates 100% of the portfolio to the asset (index) with the highest implied volatility.
        - Maintains the same position if the best-performing asset does not change.

        Args:
            t (datetime): The current date for which the portfolio is being constructed.
            information_set (dict): The data required for portfolio construction:

        Returns:
            dict: A dictionary mapping asset names to portfolio weights. If information is invalid or missing, 
                  returns an empty portfolio or retains the previous position.

        """
        if self.strategy_type == 'cash':
            expected_return = information_set.get('expected_return', np.array([]))
            companies = information_set.get('companies', np.array([]))

            # Ensure valid information set
            if expected_return.size == 0 or companies.size == 0:
                logging.error("Information set is empty or incomplete. Cannot construct portfolio.")
                return {} 
         
            # Cash strategy logic
            mu = expected_return
            n = len(mu)
            # Prepare dictionary
            portfolio = {company: 0 for company in companies}

            # Assign weights proportional to expected returns
            total_mu = sum(mu)
            if total_mu > 0:
                for i, company in enumerate(companies):
                    portfolio[company] = mu[i] / total_mu  # Normalized weights
            else:
                logging.warning("Total expected return is non-positive. Returning equal-weight portfolio.")
                portfolio = {company: 1 / n for company in companies}

            return portfolio

        elif self.strategy_type == 'vol':
            
            implied = information_set.get("expected_return_implied_vol", [])
            companies = information_set.get("companies", [])
            
            if  len(companies) == 0 or len(implied) == 0:
                logging.warning("No valid indices found. Returning empty portfolio.")
                return {}  # Empty portfolio fallback for vol strategy

            # Map highest implied to indices
            
            highest_implied_vol_index = np.argmax(implied)
            index_with_highest_vol = companies[highest_implied_vol_index]
                
            best_performer = index_with_highest_vol
            
            # Check if we need to switch positions
            if best_performer == self.previous_best_performer:
                return self.previous_position

            # Update the position
            self.previous_best_performer = best_performer

            # Allocate 100% to the best performer
            portfolio = {company: 0 for company in companies}
            portfolio[best_performer] = 1
            
            # Store the current position
            self.previous_position = portfolio

            return portfolio

        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")
        
    def compute_information(self, t: datetime, base_url=None):
        """
        Prepares the information set required for portfolio construction based on strategy type.

        For the "cash" strategy:
        - Calculates the expected return as the mean of percentage price changes for each asset.
        - Includes the list of asset tickers in the information set.

        For the "vol" strategy:
        - Calculates the expected return as the mean of percentage changes in implied volatilities.
        - Includes the list of indices in the information set.

        Args:
            t (datetime): The current date for which the information is being computed.
            base_url (str, optional): Base URL for additional data fetching (not used here but included for compatibility).

        Returns:
            dict: A dictionary containing:
                - For "cash" strategy:
                    - 'expected_return': Array of expected returns for each asset.
                    - 'companies': List of asset tickers.
                - For "vol" strategy:
                    - 'expected_return_implied_vol': Array of expected returns of implied volatilities.
                    - 'companies': List of indices.
        """
        data =self.slice_data(t)
        information_set = {}
        if self.strategy_type == 'cash':           
            # sort data by ticker and date
            data = data.sort_values(by=[self.company_column, self.time_column])
            
            data['return'] =  data.groupby(self.company_column)[self.adj_close_column].pct_change()
            
            # expected return by company 
            expected_return = data.groupby(self.company_column)['return'].mean().to_numpy()            
            companies = data[self.company_column].unique()
            information_set = {
                    "expected_return": expected_return,
                    "companies": companies,
                }
            
            return information_set
        elif self.strategy_type == 'vol':
            '''For the momentum strategy applied to vol, we will take 
            as position the underlying whose implied has the highest expect return (increase)'''
            information_set = {
                #"implied_vols": {},
                #"spot_prices": {},
                #"companies": indices,
            }
            
            data = data.sort_values(by=[self.company_column, self.time_column])
            data['return'] =  data.groupby(self.company_column)[self.vol_column].pct_change()
            # expected return by company 
            expected_return = data.groupby(self.company_column)['return'].mean().to_numpy()            
            companies = data[self.company_column].unique()
            information_set = {
                    "expected_return_implied_vol": expected_return,
                    "companies": companies,
                }

            logging.info("Information set in Momentum's compute information for vol stratrategy is: ",information_set)
            return information_set

class ShortSkew(Information):
    """
    Implements the ShortSkew strategy for portfolio construction.

    The ShortSkew strategy involves shorting a 1-month 95% put option on the index 
    with the smallest realized volatility over the past 10 days. This strategy is 
    only applicable to the "vol" strategy type.

    Attributes:
        previous_best_performer (str): Tracks the currently shorted index which is actually the one with the smallest realized (less risky).
        previous_position (dict): Tracks the previous portfolio allocation.
    """
    previous_best_performer: str = None  # Tracks the currently shorted index
    previous_position: dict = None  # Tracks the previous portfolio allocation

    def compute_portfolio(self, t: datetime, information_set):
        """
        Constructs a portfolio by shorting the index with the smallest realized volatility.

        The portfolio is adjusted to reflect a short position (-1 weight) on the 
        index identified as the best performer (smallest realized volatility). If 
        the best performer remains unchanged, the previous portfolio allocation 
        is retained.

        Args:
            t (datetime): The current date for which the portfolio is being constructed.
            information_set (dict): Data required for portfolio construction:
                - 'realized_vols': Array of realized volatilities for indices.
                - 'companies': List of index tickers.

        Returns:
            dict: A dictionary mapping index names to portfolio weights. If information 
                  is invalid or incomplete, an empty portfolio or the previous position is returned.

        """
    
        if self.strategy_type != 'vol':
            raise ValueError("ShortSkew strategy is only valid for 'vol' strategy type.")
        
        # Identify the index with the smallest 20-day realized volatility
        realized_vols = information_set.get("realized_vols",[])
        companies = information_set.get("companies",[])
        
        if  len(companies) == 0 or len(realized_vols) == 0:
                logging.warning("No valid indices found. Returning empty portfolio.")
                return {}  # Empty portfolio fallback for vol strategy
        # If both realized volatilities are 0, select the first one
        if np.all(realized_vols == 0):  # Check if all values are 0
            logging.warning("Both realized volatilities are 0. Defaulting to the first index.")
            smallest_realized_vol = 0  # Default to the first index
        else:
            smallest_realized_vol = np.argmin(realized_vols)
        
        index_with_the_smallest_realized = companies[smallest_realized_vol]
        best_performer = index_with_the_smallest_realized

        if best_performer == self.previous_best_performer:
                # If the index hasn't changed, retain the previous position
                return self.previous_position
        #update the position
        self.previous_best_performer = best_performer

        # Short 100% on the best performer 
        portfolio = {company: 0 for company in companies}
        portfolio[best_performer] = -1  # Short position
        print("Our portfolio :",portfolio)

        # Store the current position
        self.previous_position = portfolio

        return portfolio
        
    def compute_information(self, t: datetime, base_url=None):
        """
        Prepares the information set required for portfolio construction.

        The information set includes realized volatilities, implied volatilities, and 
        the list of indices. Realized volatilities are computed over the past 10 days 
        and annualized.

        Args:
            t (datetime): The current date for which the information is being computed.
            base_url (str, optional): Base URL for additional data fetching.

        Returns:
            dict: A dictionary containing:
                - 'realized_vols': Array of 10-day annualized realized volatilities for indices.
                - 'implied_vols': Array of implied volatilities for indices.
                - 'companies': List of index tickers.

        """
        data =self.slice_data(t)
        information_set = {}
        
        if self.strategy_type != 'vol':
            raise ValueError("ShortSkew strategy is only valid for 'vol' strategy type.")
        
        data = data.sort_values(by=[self.company_column, self.time_column])
        data['log_return'] =  np.log(data[self.adj_close_column] / data[self.adj_close_column].shift(1))
        data['realized_vol_10d'] = (data.groupby(self.company_column)['log_return'].rolling(window=10)).std().reset_index(level=0, drop=True)
        # Handle NaN values by replacing them with 0 (e.g., for the first few rows where there are insufficient data points)
        data['realized_vol_10d'] = data['realized_vol_10d'].fillna(0)
        data['realized_vol_10d'] = data['realized_vol_10d']*16 #the squared part of 252 to annulize our 10 Rvol 
        data.drop(columns=['log_return'], inplace=True)
        realized_vol = data.groupby(self.company_column)['realized_vol_10d'].last().to_numpy()
        companies = data[self.company_column].unique()
        implied_vol=data.groupby(self.company_column)[self.vol_column].last()

        
        information_set = {
            'realized_vols': realized_vol,  # To store realized volatilities
            'implied_vols': implied_vol,
           
            'companies': companies,
        }
        print("information set in compute_information for vol strat in ShortSkew: ",information_set)
        return information_set
        

