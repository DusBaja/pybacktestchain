#%%

from src.pybacktestchain.data_module import FirstTwoMoments,ShortSkew,Momentum
from src.pybacktestchain.broker import Backtest, StopLoss
from src.pybacktestchain.blockchain import load_blockchain
from datetime import datetime


# Set verbosity for logging
verbose = False  # Set to True to enable logging, or False to suppress it
''''''''''
backtest = Backtest(
    initial_date=datetime(2024, 10, 1),
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


#%%
#%%
#%%
''''''''' 
# For the vol strategy Momentum: working for both
initial_date = datetime(2024, 10, 1)
final_date = datetime(2024, 12, 20)
strategy_type = "vol"
indices = ["^STOXX50E","^GSPC"]  # Focus only on SX5E
risk_model_class = StopLoss
name_blockchain = 'shortskew_sx5e'##
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
#'''''''''