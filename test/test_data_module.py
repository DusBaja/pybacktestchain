import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import unittest
import pandas as pd
from datetime import datetime, timedelta
from src.pybacktestchain.data_module import *
from flask_app.utils import start_flask_app, start_ngrok

flask_app_path = '/Users/dusicabajalica/Desktop/M2/Courses/Python/pybacktestchain/pybacktestchain/flask_app/app.py'
flask_process = start_flask_app(flask_app_path)
ngrok_url = start_ngrok()
print("the URL:", ngrok_url)

class TestDataModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize reusable test resources."""
        cls.ngrok_url = ngrok_url  
        cls.ticker = "^GSPC"
        cls.start_date = "2024-10-01"
        cls.end_date = "2024-10-20"
        cls.data = get_index_data_vol(cls.ticker, cls.start_date, cls.end_date, 1, cls.ngrok_url)

    def setUp(self):
        """Run before every test case."""
        self.data_module = DataModule(data=self.data)

    def test_data_retrieval(self):
        """Test if data retrieval works as expected."""
        result = get_data_api(self.start_date, self.ticker, self.ngrok_url)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty, "Data retrieval returned an empty DataFrame.")
        
    def test_volatility_computation(self):
        """Test volatility computation logic."""
        vol_data = get_volatility_from_api(self.start_date, self.ticker, self.ngrok_url)
        self.assertIsInstance(vol_data, pd.DataFrame)
        self.assertFalse(vol_data.empty, "Volatility computation returned an empty DataFrame.")
        
    def test_information_integration(self):
        """Test integration of Information class with DataModule."""
        information = Information(
            s=timedelta(days=10),
            data_module=self.data_module,
            indices=["^GSPC", "^STOXX50E"],
            strategy_type="cash"
        )
        test_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        prices = information.get_prices(test_date, "cash")
        self.assertIsInstance(prices, dict, "Prices should be a dictionary.")
        self.assertIn("^GSPC", prices, "Expected ^GSPC in the computed prices.")
        self.assertGreater(prices["^GSPC"], 0, "Price for ^GSPC should be greater than 0.")

    def test_black_scholes_function(self):
        """Test the Black-Scholes option pricing function."""
        spot_price = 4500  
        strike_price = 4400  
        T = 21 / 365 
        r = 0.0315  
        sigma = 0.2  
        option_type = 'call'
        option_price = Information.black_scholes(spot_price, strike_price, T, r, sigma, option_type)
        self.assertGreater(option_price, 0, "Option price should be greater than 0.")

        option_type = 'put'
        option_price_put = Information.black_scholes(spot_price, strike_price, T, r, sigma, option_type)
        self.assertGreater(option_price_put, 0, "Put option price should be greater than 0.")

    def test_edge_case_empty_data(self):
        """Test behavior with an empty dataset."""
        empty_data = pd.DataFrame(columns=["Date", "Close", "ImpliedVol", "ticker"])
        empty_module = DataModule(data=empty_data)
        information = Information(data_module=empty_module, indices=["^GSPC"])
        test_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        prices = information.get_prices(test_date, "cash")
        self.assertEqual(prices, {}, "Expected an empty dictionary when data is empty.")

    def test_multiple_tickers(self):
        """Test retrieval of multiple tickers."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        stock_data = get_stocks_data(tickers, self.start_date, self.end_date)
        self.assertIsInstance(stock_data, pd.DataFrame)
        self.assertEqual(
            len(stock_data["ticker"].unique()), len(tickers),
            "Expected data for all requested tickers."
        )

    def test_volatility_strategy_momentum(self):
        """Test the volatility strategy in the Momentum class."""
        # Fetch volatility data
        momentum_data = get_index_data_vol("^GSPC", self.start_date, self.end_date, 1, self.ngrok_url)
        self.assertIsInstance(momentum_data, pd.DataFrame)
        self.assertFalse(momentum_data.empty, "Volatility data should not be empty.")

        # Initialize Momentum class with vol strategy
        data_module = DataModule(data=momentum_data)
        momentum = Momentum(
            s=timedelta(days=10),
            data_module=data_module,
            indices=["^GSPC", "^STOXX50E"],
            strategy_type="vol",
            percentage_spot=1.05
        )

        # Generate information set
        test_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        information_set = momentum.compute_information(test_date, base_url=self.ngrok_url)

        # Verify information set structure
        self.assertIn("expected_return", information_set, "Information set should contain expected_return.")
        self.assertIn("implied_vols", information_set, "Information set should contain implied_vols.")
        self.assertIn("spot_prices", information_set, "Information set should contain spot_prices.")

        # Test portfolio computation
        portfolio = momentum.compute_portfolio(test_date, information_set)
        self.assertIsInstance(portfolio, dict, "Portfolio should be a dictionary.")
        self.assertIn("^GSPC", portfolio, "Expected ^GSPC in the portfolio.")
        self.assertGreater(portfolio["^GSPC"], 0, "Portfolio allocation to ^GSPC should be greater than 0.")

if __name__ == "__main__":
    unittest.main()
