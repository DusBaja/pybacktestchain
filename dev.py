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


#%%
#%%
#%%
'''''''''' For the vol strategy ShortSkew: working
backtest = Backtest(
    initial_date=datetime(2024, 10, 1),  
    final_date=datetime(2024, 10, 20),  
    strategy_type="vol",                
    information_class=lambda **kwargs: ShortSkew(
        indices=["^STOXX50E"],          # Focus only on SX5E 
        strategy_type="vol",
        **kwargs                        
    ),
    risk_model=StopLoss,                
    name_blockchain='shortskew_sx5e',   
    verbose=verbose                     
)

# Run the backtest
backtest.run_backtest()

# Load and validate the blockchain
block_chain = load_blockchain('shortskew_sx5e')
print(str(block_chain))

# Check if the blockchain is valid
print("Is blockchain valid?", block_chain.is_valid())
'''''''''