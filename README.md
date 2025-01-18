# pybacktestchain-vol-cash

`pybacktestchain-vol-cash` is a Python package designed for financial backtesting with blockchain tracking, cash and implied volatility data, and a user-friendly interface. Store your backtests in a Blockchain, test your volatility strategies and cash strategies. There is a Dash for userfriendly interface to backtest your best strategies ! 

## Installation

```bash
$ pip install pybacktestchain-vol-cash

```
## Package Structure


### Main Directory: src/pybacktestchain_vol_cash

 - `data_module.py`: Contains functions for handling data, cash and volatility. This includes data loading, preprocessing. This part defines our main strategies defined: FirstTwoMoments, Momentum (long 1m call 105% on either SPX or SX5E, delta hedging in limits of the current cash level) and ShortSkew (shorting a put 1m 95% strike on either SPX or SX5E, delta hedging in limits of the current cash level). The capacity of going short is set up to -100% of the current cash position.

- `broker.py`: Implements execution logic for backtesting. Extends the core functionality to simulate trading: sell, buy functions, but also rebalancing, defining the portfolio valuation and defining the short selling limits. 

- `blockchain.py`: Handles the blockchain logic to track and store backtesting results, ensuring transparency and immutability.


### Dash Application: Backtest_app

 - `app.py`: The entry point for launching the Dash application. Provides an interface for users to interact with the backtesting framework, visualize results, and customize settings. It'll be launched on user's local port.


### Flask Application: Flask_app

- `utils.py`: Launches through nohup.out the Flask server locally on port 5000. This allow the code to work.
Additionnally, sets up an ngrok tunnel for external access to the implied volatility surface made outside of the package.

- `nohup.out`: Captures logs and ensures that the Flask application can run persistently in the background.


## Core Logic and Backtesting

The src/pybacktestchain_vol_cash directory contains the core modules. 
Example usage - This could be easily reproduced by running app.py from Backtest_app.

```
from src.pybacktestchain_vol_cash.data_module import FirstTwoMoments,ShortSkew,Momentum
from src.pybacktestchain_vol_cash.broker import Backtest, StopLoss
from src.pybacktestchain_vol_cash.blockchain import load_blockchain
from datetime import datetime


# Set verbosity for logging
verbose = False  # Set to True to enable logging, or False to suppress it

backtest = Backtest(
    initial_date=datetime(2024, 6, 10),
    final_date=datetime(2024, 12, 20),
    strategy_type= "cash",
    information_class=FirstTwoMoments,#
    risk_model=StopLoss,
    name_blockchain='backtest',
    verbose=verbose
)
backtest.run_backtest()

block_chain = load_blockchain('backtest')
print(str(block_chain))
# check if the blockchain is valid
print(block_chain.is_valid())


# For the vol strategy Momentum: 

initial_date = datetime(2024, 10, 1)
final_date = datetime(2025, 1, 10)
strategy_type = "vol"
indices = ["^STOXX50E","^GSPC"]  # Focus only on SX5E
risk_model_class = StopLoss
name_blockchain = 'momentum'##

verbose = True

# Initialize the Backtest object with the Momentul information class
backtest = Backtest(
    initial_date=initial_date,
    final_date=final_date,
    strategy_type=strategy_type,
    information_class=lambda **kwargs: Momentum(
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

# Load and validate the blockchain
block_chain = load_blockchain(name_blockchain)
print(str(block_chain))

# Check if the blockchain is valid
print("Is blockchain valid?", block_chain.is_valid())
```

## Usage - Key Features

**Blockchain Integration**: Immutable tracking of backtest results.

**Volatility strategies are now accessible**: Implied volatility strategy such as Short Skew on Indices through the Dash interface or directly via code. Flask + ngrok: Allows external API access for implied volatility data - constatly updated.

**User-Friendly Interfaces**: The Dash interface is accessible in the Backtest_app directory. Simplifies interaction with backtesting tools. 

**Modular Design**: Extensible modules for data handling, trading logic, and result tracking.


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`pybacktestchain` was created by Juan F. Imbet. It is licensed under the terms of the MIT license. The current code is created by Dusica Bajalica, from the 'pybacktestchain'. 

## Credits

`pybacktestchain-vol-cash` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
