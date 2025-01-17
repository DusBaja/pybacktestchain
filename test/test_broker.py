import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import unittest
import pandas as pd
from datetime import datetime, timedelta
from pybacktestchain.broker import Broker, Backtest, Position, StopLoss
from unittest.mock import patch, MagicMock
import warnings

class TestBroker(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.broker = Broker(cash=1000000, verbose=False)
        self.date = datetime(2025, 1, 1)

    def test_buy_sell_cash(self):
        """Test buying and selling in the cash strategy."""
        self.broker.buy("AAPL", 100, 150.0, self.date, "Shares", "cash")
        self.assertIn("AAPL", self.broker.positions)
        self.assertEqual(self.broker.positions["AAPL"].quantity, 100)
        self.assertEqual(self.broker.get_cash_balance(), 1000000 - 150 * 100)

        self.broker.sell("AAPL", 50, 160.0, self.date, "Shares", "cash")
        self.assertEqual(self.broker.positions["AAPL"].quantity, 50)
        self.assertEqual(self.broker.get_cash_balance(), 1000000 - 150 * 100 + 50 * 160)

    def test_buy_sell_vol(self):
        """Test buying and selling in the vol strategy."""
        self.broker.buy("^GSPC", 10, 100.0, self.date, "Options", "vol")
        self.assertIn(("^GSPC", "Options"), self.broker.positions)

    def test_stop_loss_cash(self):
        """Test stop-loss trigger in cash strategy."""
        self.broker.buy("AAPL", 100, 150.0, self.date, "Shares", "cash")
        prices = {"AAPL": 120.0}  # Price drop
        stop_loss = StopLoss(threshold=0.1)
        stop_loss.trigger_stop_loss(self.date, {}, prices, "Shares", self.broker, "cash")
        self.assertNotIn("AAPL", self.broker.positions)



if __name__ == "__main__":
    unittest.main()
