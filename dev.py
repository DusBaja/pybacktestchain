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
    strategy_type= "cash",
    information_class=FirstTwoMoments,
    risk_model=StopLoss,
    name_blockchain='backtest',
    verbose=verbose
)
backtest.run_backtest()

block_chain = load_blockchain('backtest')
print(str(block_chain))
# check if the blockchain is valid
print(block_chain.is_valid())