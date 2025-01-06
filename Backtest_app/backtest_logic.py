from datetime import datetime
from src.pybacktestchain.data_module import FirstTwoMoments,ShortSkew
from src.pybacktestchain.broker import Backtest, StopLoss
from src.pybacktestchain.blockchain import load_blockchain

def run_backtest(start_date, end_date, strategy_type):
    verbose = False  # Optionally, you can set this dynamically later

    
    if strategy_type == "cash":
        information_class = FirstTwoMoments
    else:
       
        information_class = ShortSkew

    backtest = Backtest(
        initial_date=start_date,
        final_date=end_date,
        strategy_type=strategy_type,
        information_class=information_class,
        risk_model=StopLoss,
        name_blockchain='backtest',
        verbose=verbose
    )
    
    backtest.run_backtest()

    block_chain = load_blockchain('backtest')
    return str(block_chain), block_chain.is_valid()
