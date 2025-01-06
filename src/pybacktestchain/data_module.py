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
import pandas as pd


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
    """Fetch data from the Flask API based on date and index name and return as a DataFrame."""
    if name == "^GSPC":
        name = "S&P 500"
    if name =="^STOXX50E":
        name = "Euro Stoxx 50"
    try:
        response = requests.get(f"{base_url}/api/data", params={"date": date, "index": name})
        
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                return df
            else:
                print(f"No data available for {name} on {date}.")
                return None
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return None

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

# function that retrieves historical data on prices and ATM implied for a given index
def get_volatility(vol_surface_df, index_price, percentage_spot):
    """
    Get the volatility for a given index price percentage by interpolating between the closest strikes.

    Parameters:
        vol_surface_df (pd.DataFrame): The volatility surface dataframe with strikes as columns.
        index_price (float): The current index price to approximate the volatility.
        percentage_spot (float): Percentage of the index price to target for strike selection.

    Returns:
        volatility (float): Interpolated or extrapolated volatility at the specified strike using cubic interpolation.
                            Returns NaN if the interpolated volatility is non-positive.
    """
    vol_surface_df.index = pd.to_numeric(vol_surface_df.index, errors='coerce')
    days_to_expiry = pd.Series(vol_surface_df.index, index=vol_surface_df.index).astype(float)
    
    # The row closest to the 1M expiry (20 days)
    closest_expiry_idx = (days_to_expiry - 21).abs().idxmin()
    closest_expiry_row = vol_surface_df.loc[closest_expiry_idx]
    strikes = vol_surface_df.columns[:-1].astype(float)  # The last column is expiry data
    target_strike = index_price * percentage_spot

    # Cubic interpolation with extrapolation allowed
    try:
        cubic_interpolator = interp1d(
            strikes, 
            closest_expiry_row.values[:-1],  # Exclude the expiry column 
            kind='cubic', 
            fill_value="extrapolate"
        )
        volatility = cubic_interpolator(target_strike)
        
        # Set to NaN if the volatility is non-positive 
        if volatility <= 0:
            volatility = np.nan

    except Exception as e:
        raise ValueError(f"Interpolation or extrapolation failed: {e}")
    
    return volatility


def get_index_data_vol(ticker, start_date, end_date, percentage_spot=1, base_url=None):
    """
    Retrieves historical index data and appends ATM volatility data from an API.

    Parameters:
        ticker (str): The ticker symbol for the index (e.g., '^GSPC' or '^STOXX50E').
        start_date (str): Start date for the historical data, after 2024-09-30
        end_date (str): End date for the historical data, up to today
        base_url (str): The base URL of the Flask API (e.g., 'http://localhost:5000')

    Returns:
        pd.DataFrame: DataFrame with historical data and ATM volatility for each date.
    """
    if datetime.strptime(start_date, '%Y-%m-%d') < datetime.strptime('2024-09-30', '%Y-%m-%d'):
        raise ValueError("Our database for the implied volatility starts from 2024-09-30, your start date is before: we do not have the data")

    if ticker == "^GSPC" or ticker == "^STOXX50E":
        index = yf.Ticker(ticker)
        data = index.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
        df = pd.DataFrame(data)
        df['ticker'] = ticker
        df.reset_index(inplace=True)
        
        for date in df['Date']:
            date_str = date.strftime('%Y-%m-%d')
            
            # Fetch volatility surface data from the API
            vol_surface_df = get_volatility_from_api(date_str, "S&P 500" if ticker == "^GSPC" else "Euro Stoxx 50", base_url)
            
            if vol_surface_df is not None:
                index_price = df.loc[df['Date'] == date, 'Close'].values[0]
                volatility = get_volatility(vol_surface_df, index_price * percentage_spot, percentage_spot)
                df.loc[df['Date'] == date, 'Percentage Spot selected vol for the close'] = volatility
            else:
                print(f"No volatility surface data found for date: {date_str}")
                df.loc[df['Date'] == date, 'Percentage Spot selected vol for the close'] = None
            print("Columns from the vol surface :", vol_surface_df)
    else: 
        raise ValueError("The index ticker you provided is not available in our database. Please choose '^GSPC' or '^STOXX50E'")

    return df


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
    vol_column: str = 'ImpliedVol'
    indices: list = None
    option_type: str = 'call'
    percentage_spot: float = 1.0
    strategy_type: str = 'cash'
    
    #def slice_data(self, t : datetime):
    ##    # Get the data module 
    #    data = self.data_module.data
    #    # Get the time step 
    #    s = self.s

        # Convert both `t` and the data column to timezone-aware, if needed
    #    if t.tzinfo is not None:
            # If `t` is timezone-aware, make sure data is also timezone-aware
    #        data[self.time_column] = pd.to_datetime(data[self.time_column]).dt.tz_localize(t.tzinfo.zone, ambiguous='NaT', nonexistent='NaT')
    #    else:
    ##        # If `t` is timezone-naive, ensure the data is timezone-naive as well
    #        data[self.time_column] = pd.to_datetime(data[self.time_column]).dt.tz_localize(None)
        
        # Get the data only between t-s and t
    #    data = data[(data[self.time_column] >= t - s) & (data[self.time_column] < t)]
    #    return data
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


    ######end of the modification
    @staticmethod
    def black_scholes(spot_price, strike_price, T, r, sigma, option_type='call'):
        """
        Function to compute the Black-Scholes option price.

        Parameters:
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

        Parameters:
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

    def get_prices(self, t : datetime,strategy_type: str):
        # gets the prices at which the portfolio will be rebalanced at time t 
        data = self.slice_data(t)
        
        # get the last price for each company
        prices = data.groupby(self.company_column)[self.adj_close_column].last()
        if strategy_type =="vol":
            for index in ["^GSPC", "^STOXX50E"]:
                if index in prices:
                    index_data = data[data[self.company_column] == index]
                    spot_price = index_data[self.adj_close_column].iloc[-1]
                    implied_vol = index_data[self.vol_column].iloc[-1] if self.vol_column in index_data else None
                    if implied_vol is not None:
                        T = 21/365  # We assume 1 month exp
                        r = 0.0315  
                        sigma = implied_vol
                        K = spot_price * self.percentage_spot  
                        option_price = self.black_scholes(spot_price, K, T, r, sigma, option_type=self.option_type)
                        prices[index] = option_price     
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
            ########@ to be redefined 
            information_set['expected_return'] = np.zeros(len(data[self.company_column].unique()))  # Placeholder
            information_set['covariance_matrix'] = np.zeros((len(data[self.company_column].unique()), len(data[self.company_column].unique())))  # Placeholder
            information_set['companies'] = data[self.company_column].unique()
        #### end of the modifications 
        ############################
        return information_set
class Momentum(Information):
    previous_best_performer: str = None  # Tracks the last best performer
    previous_position: dict = None  # Tracks the last position (index and option characteristics)

    def compute_portfolio(self, t: datetime, information_set):
        """
        Constructs the portfolio based on the selected strategy type (cash or vol).
        """
        if self.strategy_type == 'cash':
            # Cash strategy logic
            mu = information_set['expected_return']
            n = len(mu) if len(mu) % 2 == 0 else len(mu) - 1
            companies = information_set['companies']

            # Prepare dictionary
            portfolio = {company: 0 for company in companies}  # Default weight is 0
            returns_dict = {company: mu[i] for i, company in enumerate(companies)}
            sorted_returns = sorted(returns_dict.items(), key=lambda item: item[1], reverse=True)
            top = sorted_returns[:n // 2]
            bottom = sorted_returns[n // 2:]

            # Assign weights
            for company, _ in top:
                portfolio[company] = 1 / n
            for company, _ in bottom:
                portfolio[company] = -1 / n

            return portfolio

        elif self.strategy_type == 'vol':
            
            indices = ["^GSPC", "^STOXX50E"]
            expected_returns = {idx: information_set['expected_return'][i]
                                for i, idx in enumerate(information_set['companies']) if idx in indices}
            if not expected_returns:
                # Default to cash strategy if no valid indices are found
                return self.compute_portfolio(t, {'expected_return': [], 'companies': []})

            
            best_performer = max(expected_returns, key=expected_returns.get)

            # Check if we need to switch positions
            if best_performer == self.previous_best_performer:
                # Keep the previous position
                return self.previous_position

            # Update the position
            self.previous_best_performer = best_performer

            # Get the spot price and implied volatility
            spot_price = information_set['spot_prices'][best_performer]
            implied_vol = information_set['implied_vols'][best_performer]

            strike_price = spot_price * 1.05 # going long 105% call option for the best performer
            time_to_expiry = 21 / 365  # 1 month to expiry
            risk_free_rate = 0.0315  # Assuming 3.15% annualized rate

            if implied_vol is None or spot_price <= 0:
                raise ValueError("Invalid spot price or implied volatility for the best performer index.")

            call_option_price = Information.black_scholes(
            spot_price, strike_price, time_to_expiry, risk_free_rate, implied_vol, option_type='call')


            # Allocate 100% to the best performer 
            portfolio = {company: 0 for company in information_set['companies']}
            portfolio[best_performer] = call_option_price

            # Store the current position
            self.previous_position = portfolio

            return portfolio

        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")

    def compute_information(self, t: datetime, base_url=None):
        """
        Prepares the information set required for portfolio construction based on strategy type.
        """
        if self.strategy_type == 'vol':
            indices = ["^GSPC", "^STOXX50E"]
            start_date = (t - self.s).strftime('%Y-%m-%d')
            end_date = t.strftime('%Y-%m-%d')
            percentage_spot = self.percentage_spot

            information_set = {
                'expected_return': [],
                'implied_vols': {},
                'spot_prices': {},
                'companies': indices,
            }

            for index in indices:
                try:
                    index_data = get_index_data_vol(index, start_date, end_date, percentage_spot, base_url)
                    if index_data is not None and not index_data.empty:
                        latest_data = index_data.iloc[-1]
                        spot_price = latest_data['Close']
                        implied_vol = latest_data['Percentage Spot selected vol for the close']

                        information_set['expected_return'].append(spot_price)  # Placeholder for expected return
                        information_set['implied_vols'][index] = implied_vol
                        information_set['spot_prices'][index] = spot_price
                    else:
                        logging.warning(f"No data returned for index {index}.")
                except Exception as e:
                    logging.warning(f"Error fetching data for index {index}: {e}")

            return information_set

        elif self.strategy_type == 'cash':
            # Default cash strategy logic
            return super().compute_information(t)

        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")

@dataclass
class ShortSkew(Information):
    previous_short_index: str = None  # Tracks the currently shorted index
    previous_position: dict = None  # Tracks the previous portfolio allocation

    def compute_portfolio(self, t: datetime, information_set):
        """
        Constructs the portfolio by shorting a 1-month 90% put option 
        on the index with the smallest realized volatility over the past 20 days.
        """
        if self.strategy_type != 'vol':
            raise ValueError("ShortSkew strategy is only valid for 'vol' strategy type.")
        
        # Identify the index with the smallest 20-day realized volatility
        realized_vols = information_set['realized_vols']
        if not realized_vols:
            raise ValueError("Realized volatility data is missing from the information set.")
        
        best_index = min(realized_vols, key=realized_vols.get)

        # Check if the best index has changed
        if best_index == self.previous_short_index:
            # If the index hasn't changed, retain the previous position
            return self.previous_position

        # Update the shorted index
        self.previous_short_index = best_index

        # Get spot price and implied volatility for the best index
        spot_price = information_set['spot_prices'][best_index]
        implied_vol = information_set['implied_vols'][best_index]

        # Define option parameters
        strike_price = spot_price * 0.9  # 90% put option
        time_to_expiry = 21 / 365  # 1 month to expiry
        risk_free_rate = 0.0315  # Assuming 3.15% annualized rate

        if implied_vol is None or spot_price <= 0:
            raise ValueError("Invalid spot price or implied volatility for the selected index.")

        # Compute the put option price using Black-Scholes
        put_option_price = Information.black_scholes(
            spot_price, strike_price, time_to_expiry, risk_free_rate, implied_vol, option_type='put'
        )

        # Close the previous short position and reallocate
        portfolio = {company: 0 for company in information_set['companies']}
        portfolio[best_index] = -put_option_price  # Short position

        # Store the current position
        self.previous_position = portfolio

        return portfolio

    def compute_information(self, t: datetime, base_url=None):
        """
        Prepares the information set required for portfolio construction.
        """
        if self.strategy_type != 'vol':
            raise ValueError("ShortSkew strategy is only valid for 'vol' strategy type.")
        
        indices = ["^GSPC", "^STOXX50E"]
        start_date = (t - timedelta(days=20)).strftime('%Y-%m-%d')  # Look back 20 days
        end_date = t.strftime('%Y-%m-%d')
        percentage_spot = self.percentage_spot

        information_set = {
            'realized_vols': {},  # To store realized volatilities
            'implied_vols': {},
            'spot_prices': {},
            'companies': indices,
        }

        for index in indices:
            try:
                # Fetch historical data
                index_data = get_index_data_vol(index, start_date, end_date, percentage_spot, base_url)
                if index_data is not None and not index_data.empty:
                    # Compute realized volatility over the past 20 days
                    log_returns = np.log(index_data['Close']).diff().dropna()
                    realized_vol = log_returns.std() * np.sqrt(252)  # Annualize the volatility
                    information_set['realized_vols'][index] = realized_vol

                    # Extract the most recent implied volatility and spot price
                    latest_data = index_data.iloc[-1]
                    spot_price = latest_data['Close']
                    implied_vol = latest_data['Percentage Spot selected vol for the close']

                    information_set['implied_vols'][index] = implied_vol
                    information_set['spot_prices'][index] = spot_price
                else:
                    logging.warning(f"No data returned for index {index}.")
            except Exception as e:
                logging.warning(f"Error fetching data for index {index}: {e}")

        return information_set

