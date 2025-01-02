import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import unittest
from datetime import datetime, timedelta
from pybacktestchain.broker import Broker, Backtest, Position
from unittest.mock import patch

class TestBroker(unittest.TestCase):
    
    def setUp(self):
        # Initialize broker with 1,000,000 cash
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
    
    def test_execute_portfolio_cash(self):
        # Test portfolio execution with 'cash' strategy
        portfolio = {
            "AAPL": 0.5,  # 50% in AAPL
            "GOOGL": 0.5   # 50% in GOOGL
        }
        prices = {
            "AAPL": 150.0,
            "GOOGL": 2500.0
        }
        # This will trigger the _execute_cash_strategy method
        self.broker.execute_portfolio(portfolio, prices, datetime(2025, 1, 1), "cash")
        # Check that some transactions were made
        self.assertGreater(len(self.broker.get_transaction_log()), 0)
    
    def test_execute_portfolio_vol(self):
        # Test portfolio execution with 'vol' strategy
        portfolio = {
            "AAPL": 0.5,  # 50% in AAPL
            "GOOGL": 0.5   # 50% in GOOGL
        }
        prices = {
            "AAPL": 150.0,
            "GOOGL": 2500.0
        }
        # This will trigger the _execute_vol_strategy method, but it's a placeholder
        with patch('pybacktestchain.broker.Broker._execute_vol_strategy') as mock_vol_strategy:
            self.broker.execute_portfolio(portfolio, prices, datetime(2025, 1, 1), "vol")
            mock_vol_strategy.assert_called_once()

    def test_invalid_strategy(self):
        # Test invalid strategy type
        with self.assertRaises(ValueError):
            Backtest(
                initial_date=datetime(2025, 1, 1),
                final_date=datetime(2026, 1, 10),
                strategy_type="invalid_strategy",
                initial_cash=1000000
            )


if __name__ == '__main__':
    unittest.main()
