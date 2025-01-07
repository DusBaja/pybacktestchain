import pandas as pd
import logging
from dataclasses import dataclass
from datetime import datetime
import os 
import pickle
from pybacktestchain.data_module import UNIVERSE_SEC, FirstTwoMoments, get_stocks_data, DataModule, Information,Momentum,ShortSkew, get_index_data_vol
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

    def buy(self, ticker: str, quantity: int, price: float, date: datetime, position_type="Shares"):
        """Executes a buy order for the specified ticker (Shares or Options)."""
        total_cost = price * quantity
        if self.cash >= total_cost:
            self.cash -= total_cost
            if ticker in self.positions:
                position = self.positions[ticker]
                if position.position_type == position_type:
                    # Update existing position
                    new_quantity = position.quantity + quantity
                    new_entry_price = ((position.entry_price * position.quantity) + (price * quantity)) / new_quantity
                    position.quantity = new_quantity
                    position.entry_price = new_entry_price
                else:
                    logging.warning(f"Cannot mix positions of different types for {ticker} ({position.position_type} vs {position_type}).")
            else:
                # Create a new position
                self.positions[ticker] = Position(ticker, quantity, price, position_type)
            self.log_transaction(date, 'BUY', ticker, quantity, price, position_type)
            self.entry_prices[ticker] = price
        else:
            if self.verbose:
                logging.warning(f"Not enough cash to buy {quantity} {position_type} of {ticker} at {price}. Available cash: {self.cash}")

    def sell(self, ticker: str, quantity: int, price: float, date: datetime, position_type="Shares"):
        """Executes a sell order for the specified ticker (Shares or Options)."""
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

    def get_portfolio_value(self, market_prices: dict):
        """Calculates the total portfolio value based on the current market prices."""
        portfolio_value = self.cash
        for ticker, position in self.positions.items():
            portfolio_value += position.quantity * market_prices[ticker]
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
            
            total_value = self.get_portfolio_value(prices)
            target_value = total_value * weight
            current_value = self.positions.get(ticker, Position(ticker, 0, 0,"Shares")).quantity * price
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)
            
            if quantity_to_trade < 0:
                self.sell(ticker, abs(quantity_to_trade), price, date)
        
        # Then, handle all the buy orders, checking if there's enough cash
        for ticker, weight in portfolio.items():
            price = prices.get(ticker)
            if price is None:
                if self.verbose:
                    logging.warning(f"Price for {ticker} not available on {date}")
                continue
            
            total_value = self.get_portfolio_value(prices)
            target_value = total_value * weight
            current_value = self.positions.get(ticker, Position(ticker, 0, 0,"Shares")).quantity * price
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)
            
            if quantity_to_trade > 0:
                available_cash = self.get_cash_balance()
                cost = quantity_to_trade * price
                
                if cost <= available_cash:
                    self.buy(ticker, quantity_to_trade, price, date),"Shares"
                else:
                    if self.verbose:
                        logging.warning(f"Not enough cash to buy {quantity_to_trade} of {ticker} on {date}. Needed: {cost}, Available: {available_cash}")
                        logging.info(f"Buying as many shares of {ticker} as possible with available cash.")
                    quantity_to_trade = int(available_cash / price)
                    self.buy(ticker, quantity_to_trade, price, date,"Shares")

    def _execute_vol_strategy(self, portfolio: dict, prices: dict, date: datetime):
        """
        Execute volatility strategy delta hedging for SPX and SX5E.

        Parameters:
            portfolio (dict): Portfolio weights.
            prices (dict): Dictionary containing prices and option data for indices.
            date (datetime): Current date of execution.
        """
        for index in ["^GSPC", "^STOXX50E"]:
            if index not in portfolio:
                continue

            option_price = prices.get(index)
            if option_price is None:
                logging.warning(f"Option price for {index} not available on {date}")
                continue

            # Retrieve data for delta computation
            spot_price = portfolio.get(f"{index}_spot_price")
            implied_vol = portfolio.get(f"{index}_implied_vol")
            strike_price = portfolio.get(f"{index}_strike_price", spot_price * self.percentage_spot)
            time_to_maturity = portfolio.get(f"{index}_time_to_maturity", 21 / 365)  # Default 1-month expiry
            risk_free_rate = portfolio.get(f"{index}_risk_free_rate", 0.0315)  # Default 3.15%

            if not all([spot_price, implied_vol, option_price]):
                logging.warning(f"Missing data for delta hedging {index} on {date}")
                continue

            
            delta = Information.compute_delta(spot_price, strike_price, time_to_maturity, risk_free_rate, implied_vol, self.option_type)

            
            total_value = self.get_portfolio_value(prices)
            target_value = total_value * portfolio[index]
            option_quantity = int(target_value / option_price)
            underlying_hedge_quantity = -delta * option_quantity

            # Adjust option position
            current_option_position = self.positions.get(index, Position(index, 0, option_price,"Options"))
            option_diff = option_quantity - current_option_position.quantity

            if option_diff > 0:
                self.buy(index, option_diff, option_price, date,"Options")
            elif option_diff < 0:
                self.sell(index, abs(option_diff), option_price, date,"Options")

            # Adjust hedge position
            hedge_ticker = f"{index}_hedge"
            current_hedge_position = self.positions.get(hedge_ticker, Position(hedge_ticker, 0, spot_price,"Shares"))
            hedge_diff = underlying_hedge_quantity - current_hedge_position.quantity

            if hedge_diff > 0:
                self.buy(hedge_ticker, hedge_diff, spot_price, date,"Shares")
            elif hedge_diff < 0:
                self.sell(hedge_ticker, abs(hedge_diff), spot_price, date,"Shares")

            logging.info(f"Delta hedge executed for {index} on {date}. Option quantity: {option_quantity}, Hedge quantity: {underlying_hedge_quantity}")    ##end of modified 
    
    
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
    def trigger_stop_loss(self, t: datetime, portfolio: dict, prices: dict):
        pass

@dataclass
class StopLoss(RiskModel):
    threshold: float = 0.1
    def trigger_stop_loss(self, t: datetime, portfolio: dict, prices: dict, broker: Broker):
        
        for ticker, position in list(broker.positions.items()):
            entry_price = broker.entry_prices[ticker]
            current_price = prices.get(ticker)
            if current_price is None:
                logging.warning(f"Price for {ticker} not available on {t}")
                continue
            # Calculate the loss percentage
            loss = (current_price - entry_price) / entry_price
            if loss < -self.threshold:
                logging.info(f"Stop loss triggered for {ticker} at {t}. Selling all shares.")
                broker.sell(ticker, position.quantity, current_price, t)
@dataclass
class Backtest:
    initial_date: datetime
    final_date: datetime
    strategy_type: str = "cash" 
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'CSCO', 'NFLX']
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
    broker = Broker(cash=initial_cash, verbose=verbose)



    def __post_init__(self):
        #added
        # Validate strategy type
        if self.strategy_type not in ["cash", "vol"]:
            raise ValueError(f"Invalid strategy_type '{self.strategy_type}'. Must be 'cash' or 'vol'.")
        if self.strategy_type == "vol":
            self.universe = self.index_universe
        logging.info(f"Backtest initialized with strategy type: {self.strategy_type}")
        flask_app_path = '/Users/dusicabajalica/Desktop/M2/Courses/Python/pybacktestchain/pybacktestchain/flask_app/app.py'
        self.flask_process = start_flask_app(flask_app_path)  # Start Flask app
        self.ngrok_url = start_ngrok()  # Start ngrok and get the URL
        logging.info(f"Flask app running at {self.ngrok_url}")

        # end of added 
        
        self.backtest_name = generate_random_name()
        self.broker.initialize_blockchain(self.name_blockchain)


    #def run_backtest(self):
        #logging.info(f"Running backtest from {self.initial_date} to {self.final_date}.")
        ## added 
        #if self.strategy_type == "cash":
        ##stop added 
        #    logging.info(f"Retrieving price data for universe")
        ##added
        #elif self.strategy_type == "vol":
        #    logging.info(f"Retrieving implied volatility and option data for universe: {self.universe}")
        #stop added 

        #self.risk_model = self.risk_model(threshold=0.1)
        # self.initial_date to yyyy-mm-dd format
        #init_ = self.initial_date.strftime('%Y-%m-%d')
        # self.final_date to yyyy-mm-dd format
        #final_ = self.final_date.strftime('%Y-%m-%d')
        #df = get_stocks_data(self.universe, init_, final_)

        # Initialize the DataModule
        #data_module = DataModule(df)

        # Create the Information object
        #info = self.information_class(s = self.s, 
        #                            data_module = data_module,
        #                            time_column=self.time_column,
        #                            company_column=self.company_column,
        #                            adj_close_column=self.adj_close_column)
        
        
        #for t in pd.date_range(start=self.initial_date, end=self.final_date, freq='D'):
            #added
        #    if self.strategy_type == "cash":
            # stop added 
        #        if self.risk_model is not None:
        #            portfolio = info.compute_portfolio(t, info.compute_information(t))
        #            prices = info.get_prices(t,self.strategy_type)
        #            self.risk_model.trigger_stop_loss(t, portfolio, prices, self.broker)
            
        #        if self.rebalance_flag().time_to_rebalance(t):
        #            logging.info("-----------------------------------")
        #            logging.info(f"Rebalancing portfolio at {t}")
        #            information_set = info.compute_information(t)
        #            portfolio = info.compute_portfolio(t, information_set)
        #            prices = info.get_prices(t,self.strategy_type)
        #            self.broker.execute_portfolio(portfolio, prices, t,self.strategy_type)
            # added 
       #     elif self.strategy_type == "vol":
                # Placeholder for volatility strategy logic (to be implemented later)
        #        logging.info(f"Volatility strategy at {t} - Pending implementation")
            # stop added 

        #logging.info(f"Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(info.get_prices(self.final_date,self.strategy_type))}")
        #df = self.broker.get_transaction_log()
        # save to csv, use the backtest name 
        #df.to_csv(f"backtests/{self.backtest_name}.csv")

        # store the backtest in the blockchain
        #self.broker.blockchain.add_block(self.backtest_name, df.to_string())
    ##updated version
    #def run_backtest(self):
        #logging.info(f"Running backtest from {self.initial_date} to {self.final_date}.")

        #if self.strategy_type == "vol":
        #    logging.info(f"Retrieving implied volatility and option data for universe: {self.universe}")
         #   init_ = self.initial_date.strftime('%Y-%m-%d')
        #    final_ = self.final_date.strftime('%Y-%m-%d')
        #    df = pd.concat(
        #        [
         #           get_index_data_vol(
        #                index,
        #               init_,
         #               final_,
        #                1,
        #                base_url=self.ngrok_url  # Pass the dynamic ngrok URL
        #            ) for index in self.universe
        #        ],
        #        ignore_index=True
        #    )
        #else:
        #    logging.info(f"Retrieving price data for universe")
         #   init_ = self.initial_date.strftime('%Y-%m-%d')
        #    final_ = self.final_date.strftime('%Y-%m-%d')
        #    df = get_stocks_data(self.universe, init_, final_)

        #data_module = DataModule(df)

        #info = self.information_class(
        #    s=self.s,
        #    data_module=data_module,
        #    time_column=self.time_column,
        #    company_column=self.company_column,
        #    adj_close_column=self.adj_close_column,
        #    indices=self.index_universe,
         #   strategy_type=self.strategy_type,
        #    percentage_spot=1.0  # Adjust as necessary
        #)

        #for t in pd.date_range(start=self.initial_date, end=self.final_date, freq='D'):
        #    if self.strategy_type == "cash":
        #        if self.risk_model is not None:
        #            portfolio = info.compute_portfolio(t, info.compute_information(t))
        #            prices = info.get_prices(t, self.strategy_type)
        #            self.risk_model.trigger_stop_loss(t, portfolio, prices, self.broker)

        #        if self.rebalance_flag().time_to_rebalance(t):
        #            logging.info("-----------------------------------")
        #            logging.info(f"Rebalancing portfolio at {t}")
        #            information_set = info.compute_information(t)
        #            portfolio = info.compute_portfolio(t, information_set)
        #            prices = info.get_prices(t, self.strategy_type)
        #            self.broker.execute_portfolio(portfolio, prices, t, self.strategy_type)

        #    elif self.strategy_type == "vol":
        #        if self.rebalance_flag().time_to_rebalance(t):
        #            logging.info("-----------------------------------")
        #            logging.info(f"Rebalancing volatility strategy at {t}")
        #            information_set = info.compute_information(t, base_url=self.ngrok_url)
        #            portfolio = info.compute_portfolio(t, information_set)
        #            prices = info.get_prices(t, self.strategy_type)
        #            self.broker.execute_portfolio(portfolio, prices, t, self.strategy_type)

        #logging.info(f"Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(info.get_prices(self.final_date, self.strategy_type))}")
        #df = self.broker.get_transaction_log()
        #df.to_csv(f"backtests/{self.backtest_name}.csv")
        #self.broker.blockchain.add_block(self.backtest_name, df.to_string())
    ###third version: 
    def run_backtest(self):
        logging.info(f"Running backtest from {self.initial_date} to {self.final_date}.")
        
        # Format initial and final dates
        init_ = self.initial_date.strftime('%Y-%m-%d')
        final_ = self.final_date.strftime('%Y-%m-%d')
        
        # Retrieve data specific to the strategy type
        if self.strategy_type == "vol":
            df = pd.concat(
                [
                    get_index_data_vol(
                        index,
                        init_,
                        final_,
                        percentage_spot=1.0,  # Example parameter for vol data
                        base_url=self.ngrok_url
                    ) for index in self.universe
                ],
                ignore_index=True
            )
        elif self.strategy_type == "cash":
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
        
        # Add specific arguments for volatility strategies
        if self.strategy_type == "vol":
            info_kwargs.update({'indices': self.index_universe,'strategy_type': self.strategy_type})

        # Initialize the information class dynamically
        info = self.information_class(**info_kwargs)

        # Run the backtest logic
        for t in pd.date_range(start=self.initial_date, end=self.final_date, freq='D'):
            if self.rebalance_flag().time_to_rebalance(t):
                logging.info(f"Rebalancing portfolio at {t}.")
                information_set = info.compute_information(t, base_url=self.ngrok_url)
                portfolio = info.compute_portfolio(t, information_set)
                prices = info.get_prices(t, self.strategy_type)
                self.broker.execute_portfolio(portfolio, prices, t, self.strategy_type)

        # Final portfolio value
        logging.info(f"Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(info.get_prices(self.final_date, self.strategy_type))}")
        df = self.broker.get_transaction_log()

        # Save the transaction log
        df.to_csv(f"backtests/{self.backtest_name}.csv")

        # Store the backtest in the blockchain
        self.broker.blockchain.add_block(self.backtest_name, df.to_string())
