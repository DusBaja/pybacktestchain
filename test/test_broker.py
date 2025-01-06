import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import unittest
from datetime import datetime, timedelta
from pybacktestchain.broker import Broker, Backtest, Position
from unittest.mock import patch

class TestBroker(unittest.TestCase):
    
    def setUp(self):
        self.broker = Broker(cash=1000000, verbose=False)
        
    def test_buy_sell(self):
        # Test buying and selling stock
        self.broker.buy("AAPL", 100, 150.0, datetime(2025, 1, 1))
        # Check if position exists after buying
        self.assertEqual(self.broker.positions["AAPL"].quantity, 100)
        self.assertEqual(self.broker.get_cash_balance(), 1000000 - 150 * 100)

        self.broker.sell("AAPL", 50, 160.0, datetime(2025, 1, 2))
        # Check position after selling
        self.assertEqual(self.broker.positions["AAPL"].quantity, 50)
        self.assertEqual(self.broker.get_cash_balance(), 1000000 - 150 * 100 + 50 * 160)

    def test_transaction_log(self):
        # Check if the transaction log gets updated
        self.broker.buy("AAPL", 100, 150.0, datetime(2025, 1, 1))
        self.broker.sell("AAPL", 50, 160.0, datetime(2025, 1, 2))

        log = self.broker.get_transaction_log()
        self.assertEqual(len(log), 2)
        self.assertEqual(log.iloc[0]["Action"], "BUY")
        self.assertEqual(log.iloc[1]["Action"], "SELL")
    
    def test_initialize_blockchain(self):
        # Test if blockchain is initialized correctly
        with patch('pybacktestchain.broker.Blockchain') as MockBlockchain:
            mock_blockchain = MockBlockchain.return_value
            self.broker.initialize_blockchain("test_chain")
            mock_blockchain.store.assert_called_once()
    # test the cash part 
    def test_execute_portfolio_cash(self):
        
        portfolio = {
            "AAPL": 0.5,  # 50% in AAPL
            "GOOGL": 0.5   # 50% in GOOGL
        }
        prices = {
            "AAPL": 150.0,
            "GOOGL": 2500.0
        }
        
        self.broker.execute_portfolio(portfolio, prices, datetime(2025, 1, 1), "cash")
        
        self.assertGreater(len(self.broker.get_transaction_log()), 0)
     # test portfolio execution with 'vol' strategy - to be completed 

    def test_execute_portfolio_vol(self):
       
        portfolio = {
            "AAPL": 0.5,  
            "GOOGL": 0.5   
        }
        prices = {
            "AAPL": 150.0,
            "GOOGL": 2500.0
        }
        
        with patch('pybacktestchain.broker.Broker._execute_vol_strategy') as mock_vol_strategy:
            self.broker.execute_portfolio(portfolio, prices, datetime(2025, 1, 1), "vol")
            mock_vol_strategy.assert_called_once()

    def test_invalid_strategy(self):

        with self.assertRaises(ValueError):
            Backtest(
                initial_date=datetime(2025, 1, 1),
                final_date=datetime(2026, 1, 10),
                strategy_type="invalid_strategy",
                initial_cash=1000000
            )
    
    def test_execute_portfolio_vol(self):
        """Test portfolio execution with 'vol' strategy, including delta hedging."""
        portfolio = {
            "^GSPC": 0.6,  # 60% allocation to S&P 500 options
            "^STOXX50E": 0.4  # 40% allocation to Euro Stoxx 50 options
        }
        prices = {
            "^GSPC": 10.0,  # Option price for S&P 500
            "^STOXX50E": 8.0,  # Option price for Euro Stoxx 50
            "^GSPC_hedge": 4500.0,  # Underlying price for S&P 500
            "^STOXX50E_hedge": 4200.0  # Underlying price for Euro Stoxx 50
        }

        # Mock vol strategy data
        mock_delta_data = {
            "^GSPC": {"delta": 0.6},
            "^STOXX50E": {"delta": -0.4}
        }

    def mock_vol_strategy_side_effect(portfolio, prices, date):
        for index, weight in portfolio.items():
            option_price = prices[index]
            underlying_price = prices.get(f"{index}_hedge", None)
            delta = mock_delta_data[index]["delta"]

            # Option quantity based on allocation
            total_value = self.broker.get_portfolio_value(prices)
            target_value = total_value * weight
            option_quantity = int(target_value / option_price)

            # Hedge quantity
            hedge_quantity = -delta * option_quantity

            # Mock execution: Add positions and log
            self.broker.buy(index, option_quantity, option_price, date)
            self.broker.buy(f"{index}_hedge", hedge_quantity, underlying_price, date)

        with patch('pybacktestchain.broker.Broker._execute_vol_strategy', side_effect=mock_vol_strategy_side_effect):
            self.broker.execute_portfolio(portfolio, prices, datetime(2025, 1, 1), "vol")

                # Assertions

        transaction_log = self.broker.get_transaction_log()
        self.assertGreater(len(transaction_log), 0, "Transaction log should have entries for vol strategy.")
        self.assertIn("^GSPC", self.broker.positions, "Position for ^GSPC should be created.")
        self.assertIn("^GSPC_hedge", self.broker.positions, "Hedge position for ^GSPC should be created.")
        self.assertIn("^STOXX50E", self.broker.positions, "Position for ^STOXX50E should be created.")
        self.assertIn("^STOXX50E_hedge", self.broker.positions, "Hedge position for ^STOXX50E should be created.")

        # Check cash updates (ensure it decreased after buys)
        self.assertLess(self.broker.get_cash_balance(), 1000000, "Cash should decrease after executing vol strategy.")

if __name__ == '__main__':
    unittest.main()
