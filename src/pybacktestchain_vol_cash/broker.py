import pandas as pd
import logging
from dataclasses import dataclass,field
from datetime import datetime
import os 
import pickle
from pybacktestchain_vol_cash.data_module import UNIVERSE_SEC, FirstTwoMoments, get_stocks_data, DataModule, Information,Momentum,ShortSkew, get_index_data_vol,get_index_data_vols
from pybacktestchain_vol_cash.utils import generate_random_name
from pybacktestchain_vol_cash.blockchain import Block, Blockchain
from flask_app.utils import start_flask_app, start_ngrok

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datetime import timedelta, datetime

#---------------------------------------------------------
# Classes
#---------------------------------------------------------

@dataclass
class Position:
    ticker: str
    quantity: int
    entry_price: float
    position_type: str


@dataclass
class Broker:
    cash: float 
    positions: dict = None
    transaction_log: pd.DataFrame = None
    entry_prices: dict = None
    verbose: bool = True

    def initialize_blockchain(self, name: str):
        '''Check if the blockchain is already initialized and stored in the blockchain folder'''
        chains = os.listdir('blockchain')
        ending = f'{name}.pkl'
        if ending in chains:
            if self.verbose:
                logging.warning(f"Blockchain with name {name} already exists. Please use a different name.")
            with open(f'blockchain/{name}.pkl', 'rb') as f:
                self.blockchain = pickle.load(f)
            return

        self.blockchain = Blockchain(name)
        # Store the blockchain
        self.blockchain.store()

        if self.verbose:
            logging.info(f"Blockchain with name {name} initialized and stored in the blockchain folder.")

    def __post_init__(self):
        # Initialize positions as a dictionary of Position objects
        if self.positions is None:
            self.positions = {}
        # Initialize the transaction log as an empty DataFrame if none is provided
        if self.transaction_log is None:
            self.transaction_log = pd.DataFrame(columns=['Date', 'Action', 'Ticker', 'Quantity', 'Price', 'Cash','Position Type'])#,'Shares or Options'])
    
        # Initialize the entry prices as a dictionary
        if self.entry_prices is None:
            self.entry_prices = {}

    def buy(self, ticker: str, quantity: int, price: float, date: datetime, position_type: str,strategy_type:str):
        """Executes a buy order for the specified ticker (Shares or Options).
        Args:
            ticker (str): The ticker symbol of the asset being bought (e.g., "AAPL" for Apple shares or "^GSPC" for SPX index).
            quantity (int): The number of units being bought (e.g., shares or options).
            price (float): The price at which the asset is bought.
            date (datetime): The date of the transaction.
            position_type (str): The type of position, such as "Shares" or "Options".
            strategy_type (str): The strategy being applied for the trade, such as "volatility", "momentum", or "mean_reversion".
        
        Returns:
            None: The function does not explicitly return a value but performs the following:
                - Logs the transaction details (e.g., date, type, ticker, quantity, price, position_type).
                - Updates or creates a new position in the portfolio based on the ticker and position type.
                - For "cash" strategy, positions are keyed by the ticker symbol.
                - For "vol" strategy, positions are keyed uniquely by (ticker, position_type).
                - Logs warnings if there is insufficient cash or if position types conflict.
                - Updates the `entry_prices` dictionary with the new entry price for the position.

        Notes:
                - This method handles two distinct strategies ("cash" and "vol"), each using different logic for managing positions.
                - Ensures that mixed position types (e.g., "Shares" and "Options") are handled appropriately by creating new positions.

        """
        if strategy_type=="cash" :#same logic but actually the key needs to be different !
            total_cost = price * quantity
            if self.cash >= total_cost:
                self.cash -= total_cost
                logging.info(f"Buying {quantity} of {ticker} at {price} for {position_type}.")
                if ticker in self.positions:
                    position = self.positions[ticker]
                    
                    if position.position_type == position_type:
                        # Update existing position
                        new_quantity = position.quantity + quantity
                        new_entry_price = ((position.entry_price * position.quantity) + (price * quantity)) / new_quantity
                        position.quantity = new_quantity
                        position.entry_price = new_entry_price
                    else:
                        logging.warning(f"Cannot mix positions of different types for {ticker} ({position.position_type} vs {position_type}).Creating a new position.")
                        self.positions[ticker] = Position(ticker, quantity, price, position_type)
                else:
                    # Create a new position
                    self.positions[ticker] = Position(ticker, quantity, price, position_type)
                self.log_transaction(date, 'BUY', ticker, quantity, price, position_type)
                self.entry_prices[ticker] = price
            else:
                if self.verbose:
                    logging.warning(f"Not enough cash to buy {quantity} {position_type} of {ticker} at {price}. Available cash: {self.cash}")
        elif strategy_type=="vol": #we need an unique key
            total_cost = price * quantity
            if self.cash >= total_cost:
                self.cash -= total_cost
                logging.info(f"Buying {quantity} of {ticker} at {price} for {position_type}.")
                #unique key:
                position_key = (ticker, position_type)

                if position_key in self.positions:
                    position = self.positions[position_key]
                    
                    #if position.position_type == position_type:
                        # Update existing position
                    new_quantity = position.quantity + quantity
                    if new_quantity !=0:
                        new_entry_price = ((position.entry_price * position.quantity) + (price * quantity)) / new_quantity
                    else:
                        new_entry_price =price #the moment we cut the previous level
                    position.quantity = new_quantity
                    position.entry_price = new_entry_price
                    #else:
                    #    logging.warning(f"Cannot mix positions of different types for {ticker} ({position.position_type} vs {position_type}).Creating a new position.")
                    #    self.positions[ticker] = Position(ticker, quantity, price, position_type)
                else:
                    # Create a new position
                    self.positions[position_key] = Position(ticker, quantity, price, position_type)
                self.log_transaction(date, 'BUY', ticker, quantity, price, position_type)
                self.entry_prices[position_key] = price
            else:
                if self.verbose:
                    logging.warning(f"Not enough cash to buy {quantity} {position_type} of {ticker} at {price}. Available cash: {self.cash}")

    def sell(self, ticker: str, quantity: int, price: float, date: datetime, position_type: str,strategy_type: str):
        """Executes a sell order for the specified ticker (Shares or Options).
        Args:
        
            ticker (str): The ticker symbol of the asset being sold (e.g., "AAPL" for Apple shares or "^GSPC" for SPX index).
            quantity (int): The number of units being sold. We only pass positive values here.
            price (float): The price at which the asset is being sold.
            date (datetime): The date of the transaction.
            position_type (str): The type of position being sold, such as "Shares" or "Options".
            strategy_type (str): The strategy being applied for the trade, such as "cash" or "vol".
                - "cash": Traditional strategy without short selling. Positions must exist to be sold.
                - "vol": Volatility strategy that allows short selling if sufficient cash is available.
        Returns:
        None: The function modifies the portfolio state as follows:
            - For "cash" strategy:
                - Decreases the position's quantity by the amount sold.
                - Adds the sale proceeds to the cash balance.
                - Deletes the position if the quantity reaches zero.
                - Logs warnings if there is an attempt to sell more than the existing quantity or sell a nonexistent position.
            - For "vol" strategy:
                - Handles short selling if the cash position can cover it.
                - Updates position quantity and entry price based on short selling logic.
                - Executes partial sales or partial short sales if there is insufficient cash to cover it.
                - Creates new short positions if no position exists for the ticker and if the cash amount allow it (We want to be able as in a bank to rebuy the position in case of a brutal movement).
                - Logs warnings for cases where short selling is constrained by insufficient cash or other restrictions: not executed in that case, or partially executed.
        Notes:
            - For "cash" strategy, short selling is not allowed.
            - For "vol" strategy, positions are uniquely identified by `(ticker, position_type)` to insure that positions lock shares AND options separately.
            - Ensure that all transactions respect the cash constraints and position limits.
            - Partial execution of short selling is supported in scenarios where full execution is not feasible (not enough cash to justify such position: avoiding getting short 22 billions if the initial cash is 1m).
        """
        if strategy_type=="cash":
            if ticker in self.positions:
                position = self.positions[ticker]
                
                if position.position_type == position_type and position.quantity >= quantity:
                    # Update position
                    position.quantity -= quantity 
                    self.cash += price * quantity
                    if position.quantity == 0:
                        del self.positions[ticker]
                        del self.entry_prices[ticker]
                    self.log_transaction(date, 'SELL', ticker, quantity, price, position_type)
                else:
                    if self.verbose:
                        logging.warning(
                            f"Not enough {position_type} to sell {quantity} of {ticker}. "
                            f"Position size: {position.quantity if position.position_type == position_type else 0}."
                        )
            else:
                if self.verbose:
                    logging.warning(f"No position found for {ticker} ({position_type}).")
        #####No short selling on cash strategies 
        
        else: ##We added the short selling on vol strategies and unique key !
            position_key = (ticker, position_type)
            if position_key in self.positions: 
                
                position = self.positions[position_key]
                
                #if position.position_type == position_type: #and position.quantity >= quantity
                if position.quantity >= quantity and position.quantity>0:
                    #We have enough long position, to just reduce it 
                    # Update position
                    position.quantity -= quantity #the quantity we order is always positive
                    self.cash += price * quantity
                    if position.quantity == 0:
                        del self.positions[position_key]
                        del self.entry_prices[position_key]
                    self.log_transaction(date, 'SELL', ticker, -quantity, price, position_key[1])#as we are changing the function get_portfolio
                elif ((position.quantity>0 and position.quantity <= quantity and  self.cash>= price*(quantity-position.quantity)) or (position.quantity<0 and self.cash>= price*(quantity-position.quantity))):  
                    #enough cash to justify/cover the short selling
                    #OR
                    #enough cash to justify/cover the total short position we want to increase
                    new_quantity=position.quantity -quantity #the quantity we order is always positive
                    #we are already short or will be so we need to recompute the entry price
                    new_price = (position.entry_price*abs(position.quantity)+price*quantity)/new_quantity
                    position.quantity = new_quantity
                    position.entry_prices=new_price
                    self.cash += price * quantity
                    self.log_transaction(date, 'SELL', ticker, -quantity, price, position_key[1])
                
                elif position.quantity>0 and self.cash <= price*(quantity-position.quantity):
                #not enough to cover the new short position: partially executing it
                #if our position's quantity was enough, it would have fall in the first condition: Therefore, quantity>position.quantity
                    partial_short_position = int(self.cash/price +position.quantity)
                    #Here: we are only shorting the part we are already long and only exeeding to the limit of what is covered currently by the cash position 
                    position.quantity -= partial_short_position
                    self.cash += price*partial_short_position #not changing 
                    position.entry_prices = price 
                    self.log_transaction(date, 'SELL', ticker, -partial_short_position, price, position_key[1])
                elif position.quantity<0 and self.cash>-position.quantity*price+1000:
                #not enough to cover the full extend of the short position 
                # but the current short is still well covered with a floor of 1000
                    partial_short_position = int(self.cash/price +position.quantity) # should remain something positive 1000/price
                    if partial_short_position>0:
                        position.quantity -= partial_short_position
                        self.cash += price*partial_short_position 
                        position.entry_prices = price 
                        self.log_transaction(date, 'SELL', ticker, -partial_short_position, price, position_key[1])
                    else: #the partial short position is not enough important to short it 
                        if self.verbose:
                            logging.warning(f"The cash is not enough to short fully. The partial short position of {position_type} to sell {partial_short_position} of {ticker} is not enough. Not executed ")
                else: #we execute one part of it as not enough to cover the shorting  
                    if self.verbose:
                        logging.warning(
                            f"Not enough {position_type} to sell {quantity} of {ticker}: the cash position doesn't cover it ")                        
    
                
            else: #create a short position
                # Create a new position
                if self.cash >= price*quantity: #enough to cover the new short

                    if self.verbose:
                        logging.warning(f"No position to sell. As we are in vol strategy, we are going to short sell. Enough cash to justify it.")
                    self.positions[position_key] = Position(ticker, -quantity, price, position_key[1])
                    self.cash += price * quantity
                    self.log_transaction(date, 'SELL', ticker, -quantity, price, position_key[1])
                    self.entry_prices[position_key] = price
                else: #not enough to cover the new short:
                    partial_short_position = int(self.cash/price) 
                    if self.verbose:
                        logging.warning(f"No position to sell. As we are in vol strategy, we are going to short sell. Not enough cash to justify it, so we are executing only a portion equal to {partial_short_position} instead of {quantity}.")
                    self.positions[position_key] = Position(ticker, -partial_short_position, price, position_key[1])
                    self.cash += price * partial_short_position
                    self.log_transaction(date, 'SELL', ticker, -partial_short_position, price, position_key[1])
                    self.entry_prices[position_key] = price
    
    def log_transaction(self, date, action, ticker, quantity, price,position_type):
        """Logs the transaction.
        Args:
            date (datetime): When the transaction occurred.
            action (str): The type of transaction performed, such as "BUY" or "SELL".
            ticker (str): The ticker symbol of the asset involved in the transaction 
                        (e.g., "AAPL" for Apple shares or "^GSPC" for SPX index).
            quantity (int): The number of units involved in the transaction.
                            - Positive for buying or increasing a position.
                            - Negative for selling or reducing a position.
            price (float): The price per unit at which the transaction was executed.
            position_type (str): The type of position, such as "Shares" or "Options". 
                                       

        Returns:
            None: This method updates the portfolio's transaction history by appending the 
                transaction details to our block of transactions (saved in the blockchain)
              """
        transaction = pd.DataFrame([{
            'Date': date,
            'Action': action,
            'Ticker': ticker,
            'Quantity': quantity,
            'Price': price,
            'Cash': self.cash,
            'Position Type': position_type  # Shares or Options
            

        }])
        #if not transaction.empty and transaction.notna().any().any():
        self.transaction_log = pd.concat([self.transaction_log, transaction], ignore_index=True)

    def get_cash_balance(self):
        return self.cash

    def get_transaction_log(self):
        return self.transaction_log

    def get_portfolio_value(self, market_prices: dict,strategy_type:str):
        """Calculates the total portfolio value based on the current market prices.
        
        Args:
        market_prices (dict): A dictionary containing the current market prices of assets.
        strategy_type (str): The strategy being used, either "cash" or "vol".
                             - "cash": positions are keyed by ticker.
                             - "vol": positions are keyed by (ticker, position_type).

        Returns:
            float: The total portfolio value, which takes into account:
                - Cash balance.
                - Market value of all positions (shares or options).
                - For short positions, the valuation accounts for current market prices (as the Cash already take into account the "entry price").

        Notes:
            - For the "cash" strategy:
                - Iterates over positions keyed by ticker.
                - Values are directly multiplied by the current market price.
            - For the "vol" strategy:
                - Calculates the value of shares and options separately: different information needed.

        """
        portfolio_value = self.cash
        if strategy_type == "cash":
            for ticker, position in self.positions.items():
                portfolio_value += position.quantity * market_prices[ticker]
        else:
            #position_key = (ticker, position_type)
            for position_key, position in self.positions.items():
                
                if position_key[0] in list(market_prices["ticker"].values()):
                    ticker=position_key[0]
                    
                    if position_key[1] =="Options":
                        idx = list(market_prices["ticker"].keys())[list(market_prices['ticker'].values()).index(ticker)]        
                        price_option = market_prices["Price Option"][idx]
                        if position.quantity>=0:
                            portfolio_value +=position.quantity*(price_option)#price_option  -position.entry_price

                        else: #if short
                            portfolio_value +=position.quantity*(price_option) #-position.entry_price   if the current price is lower than the entry, it's positive
  
                    else:
                        idx = list(market_prices["ticker"].keys())[list(market_prices['ticker'].values()).index(ticker)]
                        spot = market_prices['Adj Close'][idx]
                        if position.quantity>=0:
                            portfolio_value += position.quantity* (spot)#spot -position.entry_price

                        else: #if short
                            portfolio_value += position.quantity* (spot) #-position.entry_price

        return portfolio_value
    

    def execute_portfolio(self, portfolio: dict, prices: dict, date: datetime, strategy_type: str):
        """Executes the trades for the portfolio based on the generated weights.
        
        Args:
        portfolio (dict): A dictionary representing the target portfolio allocation. The structure depends on the strategy:
                          - For "cash" strategy:
                            - Keys are ticker symbols (e.g., "AAPL", "^GSPC").
                            - Values are target weights.
                          - For "vol" strategy:
                            - Keys are (ticker, position_type) tuples.
                            - Values are target weights.
        prices (dict): A dictionary containing the current market prices of assets.
        date (datetime): The date of the trade execution.
        strategy_type (str): "cash" or "vol"
    Returns:
        None: This method modifies the portfolio and cash balance in place. Delegates the execution to private methods (`_execute_cash_strategy` or `_execute_vol_strategy`) based on `strategy_type`.

        """
        if strategy_type == "cash":
            self._execute_cash_strategy(portfolio, prices, date)
        elif strategy_type == "vol":
            self._execute_vol_strategy(portfolio, prices, date)
    
    def _execute_cash_strategy(self, portfolio: dict, prices: dict, date: datetime):
        """Handle cash strategy trading.
        This method ensures that the portfolio is rebalanced to align with the target weights specified
        in the `portfolio` dictionary. It first processes all sell orders to free up cash and then handles
        buy orders to achieve the target allocations. This way it maximizes the cash available.

        Args:
            portfolio (dict): A dictionary specifying the target portfolio allocation. Keys are ticker symbols (e.g., "AAPL"),
                            and values are target weights as fractions (e.g., 0.2 for 20% of portfolio value).
            prices (dict): A dictionary containing the current market prices for assets.
            date (datetime): The date of trade execution.

        Returns:
            None: The method modifies the internal state of the portfolio and cash balance directly.

        """
        # First, handle all the sell orders to free up cash
        for ticker, weight in portfolio.items():
            price = prices.get(ticker)
            if price is None:
                if self.verbose:
                    logging.warning(f"Price for {ticker} not available on {date}")
                continue
            
            total_value = self.get_portfolio_value(prices,"cash")
            target_value = total_value * weight
            current_value = self.positions.get(ticker, Position(ticker, 0, 0,"Shares")).quantity * price
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)
            
            if quantity_to_trade < 0:
                self.sell(ticker, abs(quantity_to_trade), price, date,"Shares","cash")
        
        # Then, handle all the buy orders, checking if there's enough cash
        for ticker, weight in portfolio.items():
            price = prices.get(ticker)
            if price is None:
                if self.verbose:
                    logging.warning(f"Price for {ticker} not available on {date}")
                continue
            
            total_value = self.get_portfolio_value(prices,"cash")
            target_value = total_value * weight
            current_value = self.positions.get(ticker, Position(ticker, 0, 0,"Shares")).quantity * price
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)
            
            if quantity_to_trade > 0:
                available_cash = self.get_cash_balance()
                cost = quantity_to_trade * price
                
                if cost <= available_cash:
                    self.buy(ticker, quantity_to_trade, price, date,"Shares","cash")
                else:
                    if self.verbose:
                        logging.warning(f"Not enough cash to buy {quantity_to_trade} of {ticker} on {date}. Needed: {cost}, Available: {available_cash}")
                        logging.info(f"Buying as many shares of {ticker} as possible with available cash.")
                    quantity_to_trade = int(available_cash / price)
                    self.buy(ticker, quantity_to_trade, price, date,"Shares","cash")

    def _execute_vol_strategy(self, portfolio: dict, prices: dict, date: datetime):
        """
        Execute volatility strategy delta hedging for SPX and SX5E.

        Args:
        portfolio (dict): A dictionary specifying the target portfolio allocation. Keys are ticker symbols (e.g., "^GSPC", "SX5E"),
        and values are target weights as fractions (e.g., 0.5 for 50% of portfolio value).
    
        prices (dict): A dictionary containing:
                - "ticker": Mapping of indices to their identifiers (e.g., {0: "^GSPC", 1: "SX5E"}).
                - "Adj Close": Adjusted close prices of the underlying assets.
                - "Price Option": Prices of the options contracts.
                - "Cost Hedging": Cost of hedging per option.

        date (datetime): The date of execution for trades.

        Returns:
            None: Updates the portfolio by executing trades for both options and underlying shares.

        Process:
            - Step 1: Validate Portfolio:
            Skips execution if the portfolio is empty or invalid.
            - Step 2: Adjust Options Exposure:
            Calculates the difference between the target value and the current value of options positions.
            Buys or sells options contracts to align with the target exposure.
            - Step 3: Perform Delta Hedging:
            Calculates the required hedging position based on the options delta and adjusts the underlying asset exposure.
            Buys or sells shares of the underlying asset to neutralize the delta.

        """
        if not portfolio:
            logging.warning(f"Empty or invalid portfolio passed for execution on {date}. Skipping.")
            return 
        for ticker, weight in portfolio.items():
            position_key_option = (ticker, "Options")
            position_key_share = (ticker, "Shares")
            if ticker in list(prices['ticker'].values()):
                idx = list(prices["ticker"].keys())[list(prices['ticker'].values()).index(ticker)]
                spot = prices["Adj Close"][idx]
                price_option = prices["Price Option"][idx]
                Cost_heging=prices["Cost Hedging"][idx]
                
                total_value = self.get_portfolio_value(prices,"vol")
                target_value = total_value * weight
                current_value = self.positions.get(position_key_option, Position(position_key_option, 0, 0,"Options")).quantity * price_option
                #in all cases its the same diff value (if current value negative or the target value negative)
                #we buy back the current value or we short the current value but always - and the target is always the one we take 
                diff_value = target_value - current_value
                                    
                quantity_to_trade = int(diff_value / price_option)
                delta = Cost_heging/spot
                target_value_hedging = quantity_to_trade*Cost_heging*weight #weight here is either -1 or 1 and as if we are short, the cost is negative, we need to adjust with the weight for the sign 
                
                current_value_hedging = self.positions.get(position_key_share, Position(ticker, 0, 0,"Shares")).quantity * spot
                diff_value_hedging = target_value_hedging-current_value_hedging

                quantity_to_trade_for_hedging = int(diff_value_hedging/spot)
                logging.info(f"The quantity to trade for delta hedging is {quantity_to_trade_for_hedging}")
                if quantity_to_trade < 0:
                    self.sell(ticker, abs(quantity_to_trade), price_option, date,"Options","vol")
                if quantity_to_trade_for_hedging<0:
                    self.sell(ticker, abs(quantity_to_trade_for_hedging), spot, date,"Shares","vol")
                    logging.info(f"Delta hedge executed for {ticker} on {date}.")
        
        for ticker, weight in portfolio.items():
            position_key_option = (ticker, "Options")
            position_key_share = (ticker, "Shares")
            if ticker in list(prices['ticker'].values()):
                idx = list(prices["ticker"].keys())[list(prices['ticker'].values()).index(ticker)]
                spot = prices["Adj Close"][idx]
                price_option = prices["Price Option"][idx]
                Cost_heging=prices["Cost Hedging"][idx]
                
                total_value = self.get_portfolio_value(prices,"vol")
                target_value = total_value * weight
                current_value = self.positions.get(position_key_option, Position(ticker, 0, 0,"Options")).quantity * price_option
                diff_value = target_value - current_value
                quantity_to_trade = int(diff_value / price_option)
                delta = Cost_heging/spot
                
                target_value_hedging = quantity_to_trade*Cost_heging*weight
                
                current_value_hedging = self.positions.get(position_key_share, Position(ticker, 0, 0,"Shares")).quantity * spot
                diff_value_hedging = target_value_hedging-current_value_hedging
                quantity_to_trade_for_hedging = int(diff_value_hedging/spot)
                logging.info(f"The quantity to trade for delta hedging is {quantity_to_trade_for_hedging}")
                if quantity_to_trade > 0:
                    available_cash = self.get_cash_balance()
                    cost = quantity_to_trade * price_option
                    if cost <= available_cash:
                        self.buy(ticker, abs(quantity_to_trade), price_option, date,"Options","vol")
                    else:
                        if self.verbose:
                            logging.warning(f"Not enough cash to buy {quantity_to_trade} options of {ticker} on {date}. Needed: {cost}, Available: {available_cash}")
                            logging.info(f"Buying as many options of {ticker} as possible with available cash.")
                        quantity_to_trade = int(available_cash / price_option)
                        self.buy(ticker, abs(quantity_to_trade), price_option, date,"Options","vol")

                if quantity_to_trade_for_hedging>0:
                    available_cash = self.get_cash_balance()
                    if Cost_heging<=available_cash:
                        self.buy(ticker, abs(quantity_to_trade_for_hedging), spot, date,"Shares","vol")
                    else:
                        if self.verbose:
                            logging.warning(f"Not enough cash to buy {quantity_to_trade_for_hedging} shares of {ticker} for delta hedging on {date}. Needed: {Cost_heging}, Available: {available_cash}")
                            logging.info(f"Buying as many shares of {ticker} as possible with available cash.")
                        quantity_to_trade_for_hedging=int(available_cash / spot)
                        logging.info(f"The quantity to trade for delta hedging is {quantity_to_trade_for_hedging}")
                        self.buy(ticker, quantity_to_trade_for_hedging, spot, date,"Shares","vol")
                        logging.info(f"Delta hedge executed for {ticker} on {date}.")
                    
            ##end of modified 
    
    
    ####################
    def get_transaction_log(self):
        """Returns the transaction log."""
        return self.transaction_log

@dataclass
class RebalanceFlag:
    def time_to_rebalance(self, t: datetime):
        pass 

# Implementation of e.g. rebalancing at the end of each month
@dataclass
class EndOfMonth(RebalanceFlag):
    def time_to_rebalance(self, t: datetime):
        # Convert to pandas Timestamp for convenience
        pd_date = pd.Timestamp(t)
        # Get the last business day of the month
        last_business_day = pd_date + pd.offsets.BMonthEnd(0)
        # Check if the given date matches the last business day
        return pd_date == last_business_day

@dataclass
class RiskModel:
    def trigger_stop_loss(self, t: datetime, portfolio: dict, prices: dict,broker: Broker, strategy_type: str):
        pass

@dataclass
class StopLoss(RiskModel):
    """
    Implements a stop-loss risk model to trigger portfolio rebalancing or liquidation when losses exceed a specified threshold.

    Attributes:
        threshold (float): 
            The loss threshold (expressed as a decimal fraction) at which stop-loss orders are triggered. 
            Defaults to 0.1 (10%).

    Methods:
        trigger_stop_loss(t, portfolio, prices, position_type, broker, strategy_type):
            Evaluates the portfolio and triggers stop-loss actions for positions exceeding the loss threshold.
    """
    threshold: float = 0.1
    def trigger_stop_loss(self, t: datetime, portfolio: dict, prices: dict, broker: Broker, strategy_type: str):
        """        
        Evaluates the portfolio for stop-loss conditions and executes necessary trades.

        Args:
            t (datetime): The current timestamp when the stop-loss evaluation is performed.
            portfolio (dict): A dictionary of portfolio weights.
            prices (dict): A dictionary containing the current market prices for tickers:
                    - For "cash" strategy: {ticker: current_price}.
                    - For "vol" strategy:
                        - "ticker": Mapping of indices to their identifiers (e.g., "^GSPC").
                        - "Adj Close": Adjusted close prices for underlying assets.
                        - "Price Option": Current prices for options.
            position_type (str): The type of position to evaluate, e.g., "Shares" or "Options".
            broker (Broker): The broker instance managing portfolio positions and executing trades.
            strategy_type (str): The strategy type, either "cash" or "vol," determining the logic for stop-loss execution.

        Process:
            - For "cash" strategy:
                1. Checks each ticker in the broker's positions.
                2. Compares the current market price to the entry price.
                3. If the loss percentage exceeds the threshold, sells all units of the position.
            - For "vol" strategy:
                1. Evaluates both "Shares" and "Options" positions.
                2. Adjusts logic based on long or short positions:
                    - For long positions: Loss is calculated as (current_price - entry_price) / entry_price.
                    - For short positions: Loss is calculated as (entry_price - current_price) / entry_price.
                3. Executes necessary buy or sell orders based on loss conditions.
        """
        if strategy_type=="cash":
            
            for ticker, position in list(broker.positions.items()): 
                entry_price = broker.entry_prices[ticker]
                current_price = prices.get(ticker)
                
                if current_price is None:
                    logging.warning(f"Price for {ticker} not available on {t}")
                    continue
                # Calculate the loss percentage
                loss = (current_price - entry_price) / entry_price
                if loss < -self.threshold:
                    logging.info(f"Stop loss triggered for {ticker} at {t}. Selling all Shares in our cash position.")
                    broker.sell(ticker, position.quantity, current_price, t,"Shares","cash")
            
        else:
            for position_key, position in list(broker.positions.items()):
                ticker = position_key[0]
                position_type=position_key[1]
                if position_type=="Shares":
                    
                #if position.position_type =="Shares":
                    entry_price = broker.entry_prices[position_key]
                    idx = list(prices["ticker"].keys())[list(prices['ticker'].values()).index(ticker)]
                    current_price = prices["Adj Close"][idx]
                    if current_price is None:
                        logging.warning(f"Price for {ticker} not available on {t}")
                        continue
                    # Calculate the loss percentage
                    if position.quantity >0:
                        loss = (current_price - entry_price) / entry_price
                        if loss < -self.threshold: #Only if the current price is lower than entry price significantly 
                            logging.info(f"Stop loss triggered for {ticker} at {t}. Selling all {position_type}.")
                            broker.sell(ticker, abs(position.quantity), current_price, t,position_type,"vol")
                    else: #if the position is a short: we look at the "negative" difference: how much did it decrease
                        loss = (entry_price-current_price) / entry_price
                        if loss < -self.threshold:
                            logging.info(f"Stop loss triggered for {ticker} at {t}. Buying back all {position_type}.")
                            broker.buy(ticker, abs(position.quantity), current_price, t,position_type,"vol")

                else:
                    entry_price = broker.entry_prices[position_key]
                    idx = list(prices["ticker"].keys())[list(prices['ticker'].values()).index(ticker)]
                    current_price = prices["Price Option"][idx]
                    if current_price is None:
                        logging.warning(f"Price for {ticker} not available on {t}")
                        continue
                    if position.quantity>0:
                        # Calculate the loss percentage
                        loss = (current_price - entry_price) / entry_price
                        if loss < -self.threshold:
                            logging.info(f"Stop loss triggered for {ticker} at {t}. Selling all {position_type}.")
                            broker.sell(ticker, abs(position.quantity), current_price, t,position_type,"vol")
                    else:
                        # Calculate the loss percentage
                        loss = (entry_price-current_price) / entry_price
                        if loss < -self.threshold:
                            logging.info(f"Stop loss triggered for {ticker} at {t}. Buying back all {position_type}.")
                            broker.buy(ticker, abs(position.quantity), current_price, t,position_type,"vol")

@dataclass
class Backtest:
    initial_date: datetime
    final_date: datetime
    strategy_type: str #= "cash"  or "vol"
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'CSCO', 'NFLX']#,'^GSPC', '^STOXX50E'
    index_universe = ['^GSPC', '^STOXX50E']
    information_class : type  = Information
    s: timedelta = timedelta(days=360)
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column : str ='Adj Close'
    rebalance_flag : type = EndOfMonth
    risk_model : type = StopLoss
    initial_cash: int = 1000000  # Initial cash in the portfolio
    name_blockchain: str = 'backtest'
    verbose: bool = True
    #broker: Broker = Broker(cash=initial_cash, verbose=verbose)
    broker: Broker = field(default_factory=lambda: Broker(cash=1000000, verbose=False))
    PnL: float =0.0



    def __post_init__(self):
        #added
        # Validate strategy type
        if self.strategy_type not in ["cash", "vol"]:
            raise ValueError(f"Invalid strategy_type '{self.strategy_type}'. Must be 'cash' or 'vol'.")
        if self.strategy_type == "vol":
            self.universe = self.index_universe

        logging.info(f"Backtest initialized with strategy type: {self.strategy_type}")
                    #flask_app_path = '/Users/dusicabajalica/Desktop/M2/Courses/Python/pybacktestchain/pybacktestchain/flask_app/app.py'
        self.flask_process = start_flask_app()  # Start Flask app
        self.ngrok_url = start_ngrok()  # Start ngrok and get the URL
        if self.flask_process:
            base_url = start_ngrok()
            if base_url:
                logging.info(f"Flask app is running at {self.ngrok_url }")
            else:
                logging.error("Failed to retrieve ngrok URL.")
        else:
            logging.error("Failed to start Flask app.")
        #self.ngrok_url = start_ngrok()  # Start Cloudflared (disguised as ngrok)
        
        #if self.ngrok_url:
        #    logging.info(f"Flask app running at {self.ngrok_url}")
        #else:
        #    logging.error("Failed to start Cloudflared. Exiting initialization.")
        #    raise RuntimeError("Failed to start Cloudflared tunnel.")

        # end of added 
        
        self.backtest_name = generate_random_name()
        self.broker.initialize_blockchain(self.name_blockchain)

    ###third version: 
    def run_backtest(self):
        """
        Executes the backtest for the defined strategy over the specified date range.

        The method orchestrates the entire backtesting process, including data retrieval, 
        portfolio rebalancing, applying risk models, and recording the results.

        All this is done by:
        1. Retrieving market data based on the strategy type ("cash" or "volatility"). 
        2. Initializing data and information processing modules.
        3. Iteratively process each date in the backtest range:
            - Apply risk management models (e.g., stop-loss).
            - Rebalance the portfolio based on the defined rebalance frequency and strategy logic.
        4. Calculating and log the final portfolio value and P&L.
        5. Save the transaction log and store the backtest results in a blockchain for record-keeping.

        Args:
            None. All necessary parameters are retrieved from the instance variables.

        Attributes (from instance):
            self.initial_date (datetime): 
                Start date of the backtest.
            self.final_date (datetime): 
                End date of the backtest.
            self.strategy_type (str): 
                The trading strategy type ("cash" or "vol").
            self.universe (list): 
                List of tickers or indices to be traded in the backtest.
            self.index_universe (list): 
                List of indices for volatility strategies (used only for "vol").
            self.ngrok_url (str): 
                URL for accessing external surface vol data (used for dynamic data retrieval).
            self.broker (Broker): 
                Handles trades, maintains positions, and tracks cash and portfolio value.
            self.information_class (class): 
                Dynamically initialized class for processing data and computing portfolios.
            self.time_column (str): 
                Column name for time data in the dataset.
            self.company_column (str): 
                Column name for the company or ticker information in the dataset.
            self.adj_close_column (str): 
                Column name for adjusted close prices.
            self.rebalance_flag (function): 
                Determines when to rebalance the portfolio.
            self.initial_cash (float): 
                Starting cash for the backtest.
            self.backtest_name (str): 
                Name of the backtest for logging and saving results.

        Raises:
            ValueError: If the provided strategy type is unsupported.

        Key Steps:
            1. Data Retrieval:
                - For "cash" strategies, retrieves price data using `get_stocks_data`.
                - For "vol" strategies, retrieves implied volatility and option data using `get_index_data_vols`.
            2. Data and Information Initialization:
                - Initializes the `DataModule` with retrieved data.
                - Dynamically initializes the `information_class` with strategy-specific parameters.
            3. Backtest Iteration:
                - Iterates through each date in the date range, applying the following:
                    - Risk Model: Executes stop-loss logic based on portfolio and prices.
                    - Rebalancing: Computes portfolio weights and executes trades when rebalancing is triggered.
            4. Final Portfolio Evaluation:
                - Calculates and logs the final portfolio value and P&L.
                - Saves transaction logs and stores the backtest results in the blockchain.

        Output:
            - Saves transaction logs as a CSV file named after the backtest.
            - Stores the backtest results in the blockchain for immutable record-keeping.
        
        Example: 
                initial_date = datetime(2024, 10, 1)
                final_date = datetime(2025, 1, 10)
                strategy_type = "vol"
                indices = ["^STOXX50E","^GSPC"]  
                risk_model_class = StopLoss
                name_blockchain = 'shortskew_sx5e'##
                verbose = True

                # Initialize the Backtest object with the Momentul information class
                backtest = Backtest(
                    initial_date=initial_date,
                    final_date=final_date,
                    strategy_type=strategy_type,
                    information_class=lambda **kwargs: ShortSkew(
                        **{
                            "indices": indices,           
                            "strategy_type": strategy_type,
                            **kwargs                      
                        }
                    ),
                    risk_model=risk_model_class,
                    name_blockchain=name_blockchain,
                    verbose=verbose
                )

                # Run the backtest
                backtest.run_backtest()

        """
        logging.info(f"Running backtest from {self.initial_date} to {self.final_date}.")
        
        # Format initial and final dates
        init_ = self.initial_date.strftime('%Y-%m-%d')
        final_ = self.final_date.strftime('%Y-%m-%d')
        self.risk_model = self.risk_model(threshold=0.1)
        # Retrieve data specific to the strategy type
        if self.strategy_type == "vol":
            logging.info("Retrieving implied volatility and option data for the universe.")
            df = get_index_data_vols(
                        self.universe,
                        init_,
                        final_,
                        percentage_spot=1.0,  # Example parameter for vol data
                        base_url=self.ngrok_url
                    )
            df.reset_index(drop=True, inplace=True) #as there was an index problem I think
            
        elif self.strategy_type == "cash":
            logging.info("Retrieving price data for the universe.")
            df = get_stocks_data(self.universe, init_, final_)
            
            
        else:
            raise ValueError(f"Unsupported strategy type: {self.strategy_type}")

        # Initialize the DataModule
        
        data_module = DataModule(df)
        
        # Dynamically initialize the Information class
        info_kwargs = {
            's': self.s,
            'data_module': data_module,
            'time_column': self.time_column,
            'company_column': self.company_column,
            'adj_close_column': self.adj_close_column,
        }
        if self.strategy_type == "vol":
            info_kwargs.update({'indices': self.index_universe, 'strategy_type': self.strategy_type})
            
        # Initialize the information class dynamically
        info = self.information_class(**info_kwargs)
        
        # Run the backtest logic
        
        for t in pd.date_range(start=self.initial_date, end=self.final_date, freq='D'):  
            
            logging.info(f"Processing date: {t}")
            
            if self.risk_model is not None:
                #logging.info("Applying risk model.")
                portfolio = info.compute_portfolio(t, info.compute_information(t,base_url=self.ngrok_url))
                
                logging.debug(f"Portfolio at {t}: {portfolio}")
                prices = info.get_prices(t, self.strategy_type,str(type(info).__name__))
                
                logging.debug(f"Prices at {t}: {prices}")
                logging.debug(f"Broker state at {t}: {self.broker}")
                self.risk_model.trigger_stop_loss(t, portfolio, prices, self.broker,self.strategy_type) 

            if self.rebalance_flag().time_to_rebalance(t):
                logging.info("-----------------------------------")
                logging.info(f"Rebalancing portfolio at {t}")
                
                information_set = info.compute_information(t,base_url=self.ngrok_url)
                portfolio = info.compute_portfolio(t, information_set)
                prices = info.get_prices(t, self.strategy_type,str(type(info).__name__))
                self.broker.execute_portfolio(portfolio, prices, t, self.strategy_type)

            
        # Final portfolio value
        Portfolio_value= self.broker.get_portfolio_value(info.get_prices(self.final_date, self.strategy_type,str(type(info).__name__)),self.strategy_type)
        logging.info(f"Backtest completed. Final portfolio value: {Portfolio_value}")
        df = self.broker.get_transaction_log()
        self.PnL=Portfolio_value- self.initial_cash
        logging.info(f"Final P&L: {self.PnL}")
        
        # Save the transaction log
        df.to_csv(f"backtests/{self.backtest_name}.csv")

        # Store the backtest in the blockchain
        self.broker.blockchain.add_block(self.backtest_name, df.to_string())
        logging.info(f"The backtest's name is {self.backtest_name}")
        logging.info("Backtest results stored in blockchain.")