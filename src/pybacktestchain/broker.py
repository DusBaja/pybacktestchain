import pandas as pd
import logging
from dataclasses import dataclass
from datetime import datetime
import os 
import pickle
from pybacktestchain.data_module import UNIVERSE_SEC, FirstTwoMoments, get_stocks_data, DataModule, Information,Momentum,ShortSkew, get_index_data_vol,get_index_data_vols
from pybacktestchain.utils import generate_random_name
from pybacktestchain.blockchain import Block, Blockchain
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
        """Executes a buy order for the specified ticker (Shares or Options)."""
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
        """Executes a sell order for the specified ticker (Shares or Options)."""
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
                print("position_key in the sell !!!",position_key)
                print("Currennt positionssss in the sell ",self.positions)
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
                #else: 
                #    if self.verbose:
                #        logging.warning(f"Cannot mix positions of different types for {ticker} ({position.position_type} vs {position_type}).Creating a short selling position.")
                #    self.positions[ticker] = Position(ticker, -quantity, price, position_type)
                #    self.log_transaction(date, 'SELL', ticker, -quantity, price, position_type)
                
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
    def log_transaction(self, date, action, ticker, quantity, price,position_type="shares"):
        """Logs the transaction."""
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
        """Calculates the total portfolio value based on the current market prices."""
        portfolio_value = self.cash
        if strategy_type == "cash":
            for ticker, position in self.positions.items():
                portfolio_value += position.quantity * market_prices[ticker]
        else:
            #position_key = (ticker, position_type)
            for position_key, position in self.positions.items():
                print("position_key in the get_portfolio_value",position_key)
                if position_key[0] in list(market_prices["ticker"].values()):
                    ticker=position_key[0]
                    
                    if position_key[1] =="Options":
                        idx = list(market_prices["ticker"].keys())[list(market_prices['ticker'].values()).index(ticker)]        
                        price_option = market_prices["Price Option"][idx]
                        if position.quantity>=0:
                            portfolio_value +=position.quantity*(price_option)#price_option  -position.entry_price
                            print("For ",position_key, "normally options")
                            print("With a position: ",position)
                            print("position.quantity",position.quantity)
                            print("Current price",price_option, " and the previous entry was ", position.entry_price)
                            print("position.quantity*(price_option-position.entry_price)",position.quantity*(price_option-position.entry_price)) 
                        else: #if short
                            portfolio_value +=position.quantity*(price_option) #-position.entry_price   if the current price is lower than the entry, it's positive
                            print("For ",position_key, "normally options")
                            print("With a position: ",position)
                            print("position.quantity",position.quantity, "negative normally")
                            print("Current price",price_option, " and the previous entry was ", position.entry_price)
                            print("position.quantity*(price_option-position.entry_price)",position.quantity*(price_option-position.entry_price))    
                    else:
                        idx = list(market_prices["ticker"].keys())[list(market_prices['ticker'].values()).index(ticker)]
                        spot = market_prices['Adj Close'][idx]
                        if position.quantity>=0:
                            portfolio_value += position.quantity* (spot)#spot -position.entry_price
                            print("For ",position_key, "normally shares")
                            print("With a position: ",position)
                            print("position.quantity",position.quantity, "normally positive")
                            print("Current spot",spot, " and the previous entry was ", position.entry_price)
                            print("position.quantity",position.quantity* (spot-position.entry_price))
                        else: #if short
                            portfolio_value += position.quantity* (spot) #-position.entry_price
                            print("For ",position_key, "shares normally")
                            print("With a position: ",position)
                            print("position.quantity",position.quantity, "noramlly negative")
                            print("Current spot",spot, " and the previous entry was ", position.entry_price)
                            print("position.quantity",position.quantity* (spot-position.entry_price))
        return portfolio_value
    
    #modifid
    #def execute_portfolio(self, portfolio: dict, prices: dict, date: datetime):
    #    """Executes the trades for the portfolio based on the generated weights."""
        
        # First, handle all the sell orders to free up cash
    #    for ticker, weight in portfolio.items():
    #        price = prices.get(ticker)
    #        if price is None:
    #            if self.verbose:
    #                logging.warning(f"Price for {ticker} not available on {date}")
    #            continue
    #        
    #        total_value = self.get_portfolio_value(prices)
    #        target_value = total_value * weight
    #        current_value = self.positions.get(ticker, Position(ticker, 0, 0)).quantity * price
    #        diff_value = target_value - current_value
    #        quantity_to_trade = int(diff_value / price)
    #        
    #        if quantity_to_trade < 0:
    #            self.sell(ticker, abs(quantity_to_trade), price, date)
        
        # Then, handle all the buy orders, checking if there's enough cash
    #    for ticker, weight in portfolio.items():
    #        price = prices.get(ticker)
    #        if price is None:
    #           if self.verbose:
    #                logging.warning(f"Price for {ticker} not available on {date}")
    #            continue
            
    #        total_value = self.get_portfolio_value(prices)
    #        target_value = total_value * weight
    #        current_value = self.positions.get(ticker, Position(ticker, 0, 0)).quantity * price
    #        diff_value = target_value - current_value
    #        quantity_to_trade = int(diff_value / price)
            
    #        if quantity_to_trade > 0:
    #            available_cash = self.get_cash_balance()
    #            cost = quantity_to_trade * price
    #            
    #            if cost <= available_cash:
    #                self.buy(ticker, quantity_to_trade, price, date,"Shares or Options")
    #            else:
    #                if self.verbose:
    #                    logging.warning(f"Not enough cash to buy {quantity_to_trade} of {ticker} on {date}. Needed: {cost}, Available: {available_cash}")
    #                    logging.info(f"Buying as many shares of {ticker} as possible with available cash.")
    #                quantity_to_trade = int(available_cash / price)
    #                self.buy(ticker, quantity_to_trade, price, date,"Shares or Options")
    ####################
    def execute_portfolio(self, portfolio: dict, prices: dict, date: datetime, strategy_type: str):
        """Executes the trades for the portfolio based on the generated weights."""
        if strategy_type == "cash":
            self._execute_cash_strategy(portfolio, prices, date)
        elif strategy_type == "vol":
            self._execute_vol_strategy(portfolio, prices, date)
    
    def _execute_cash_strategy(self, portfolio: dict, prices: dict, date: datetime):
        """Handle cash strategy trading."""
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

        Parameters:
            portfolio (dict): Portfolio weights.
            prices (dict): Dictionary containing prices (option and shares) and option data for indices.
            date (datetime): Current date of execution.
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
                print("quantity_to_trade_for_hedging",quantity_to_trade_for_hedging)
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
                print("delta",delta)
                target_value_hedging = quantity_to_trade*Cost_heging*weight
                
                current_value_hedging = self.positions.get(position_key_share, Position(ticker, 0, 0,"Shares")).quantity * spot
                diff_value_hedging = target_value_hedging-current_value_hedging
                quantity_to_trade_for_hedging = int(diff_value_hedging/spot)
                print("quantity_to_trade_for_hedging",quantity_to_trade_for_hedging)
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
                        print("quantity_to_trade_for_hedging",quantity_to_trade_for_hedging)
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
    def trigger_stop_loss(self, t: datetime, portfolio: dict, prices: dict,position_type:str,broker: Broker, strategy_type: str):
        pass

@dataclass
class StopLoss(RiskModel):
    threshold: float = 0.1
    def trigger_stop_loss(self, t: datetime, portfolio: dict, prices: dict,position_type:str, broker: Broker, strategy_type: str):
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
                    logging.info(f"Stop loss triggered for {ticker} at {t}. Selling all {position_type}.")
                    broker.sell(ticker, position.quantity, current_price, t,position_type,"cash")
            
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
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'CSCO', 'NFLX','^GSPC', '^STOXX50E']
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
    broker: Broker = Broker(cash=initial_cash, verbose=verbose)



    def __post_init__(self):
        #added
        # Validate strategy type
        if self.strategy_type not in ["cash", "vol"]:
            raise ValueError(f"Invalid strategy_type '{self.strategy_type}'. Must be 'cash' or 'vol'.")
        if self.strategy_type == "vol":
            self.universe = self.index_universe
        logging.info(f"Backtest initialized with strategy type: {self.strategy_type}")
        #flask_app_path = '/Users/dusicabajalica/Desktop/M2/Courses/Python/pybacktestchain/pybacktestchain/flask_app/app.py'
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        flask_app_path = os.path.join(package_root, "flask_app", "app.py")
        self.flask_process = start_flask_app(flask_app_path)  # Start Flask app
        self.ngrok_url = start_ngrok()  # Start ngrok and get the URL
        #self.ngrok_url = start_ngrok()  # Start Cloudflared (disguised as ngrok)
        logging.info(f"Flask app running at {self.ngrok_url}")
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
        #    if self.rebalance_flag().time_to_rebalance(t):
        #        logging.info(f"Rebalancing portfolio at {t}.")
        #        information_set = info.compute_information(t, base_url=self.ngrok_url)
        #        portfolio = info.compute_portfolio(t, information_set)
        #        prices = info.get_prices(t, self.strategy_type)
        #        self.broker.execute_portfolio(portfolio, prices, t, self.strategy_type)
        ###################Debeuging from there: all above seems to work fine for both vol and cash    
            
            logging.info(f"Processing date: {t}")
            
            if self.risk_model is not None:
                #logging.info("Applying risk model.")
                portfolio = info.compute_portfolio(t, info.compute_information(t,base_url=self.ngrok_url))
                logging.debug(f"Portfolio at {t}: {portfolio}")
                prices = info.get_prices(t, self.strategy_type,str(type(info).__name__))
                
                logging.debug(f"Prices at {t}: {prices}")
                logging.debug(f"Broker state at {t}: {self.broker}")
                self.risk_model.trigger_stop_loss(t, portfolio, prices,'Shares', self.broker,self.strategy_type) #test with only shares

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
        logging.info(f"Final P&L: {Portfolio_value- self.initial_cash}")
        print("self cash initial", self.initial_cash)
        # Save the transaction log
        df.to_csv(f"backtests/{self.backtest_name}.csv")

        # Store the backtest in the blockchain
        self.broker.blockchain.add_block(self.backtest_name, df.to_string())
        print("self.backtest_name",self.backtest_name)
        logging.info("Backtest results stored in blockchain.")