#%%
from src.pybacktestchain.data_module import FirstTwoMoments
from src.pybacktestchain.broker import Backtest, StopLoss
from src.pybacktestchain.blockchain import load_blockchain
from datetime import datetime

# Set verbosity for logging
verbose = False  # Set to True to enable logging, or False to suppress it

backtest = Backtest(
    initial_date=datetime(2019, 1, 1),
    final_date=datetime(2020, 1, 1),
    information_class=FirstTwoMoments,
    risk_model=StopLoss,
    name_blockchain='backtest',
    verbose=verbose
)
