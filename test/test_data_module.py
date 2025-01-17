import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import unittest
from unittest.mock import patch
import pandas as pd
from datetime import datetime, timedelta
from src.pybacktestchain.data_module import *
from flask_app.utils import start_flask_app, start_ngrok

# Start Flask app and ngrok for obtaining the URL
flask_app_path = '/Users/dusicabajalica/Desktop/M2/Courses/Python/pybacktestchain/pybacktestchain/flask_app/app.py'
flask_process = start_flask_app(flask_app_path)
ngrok_url = start_ngrok()
print("the URL:", ngrok_url)


class TestDataModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize reusable test resources."""
        cls.ticker = "^GSPC"
        cls.start_date = "2024-10-01"
        cls.end_date = "2024-10-20"
        cls.base_url = ngrok_url

    def test_get_data_api(self):
        """Test get_data_api with mock response."""
        mock_response = pd.DataFrame({"Date": ["2024-10-01", "2024-10-02"], "Close": [4500, 4520]})
        with patch("src.pybacktestchain.data_module.get_data_api", return_value=mock_response) as mock_get_data_api:
            result = get_data_api(self.start_date, self.ticker, self.base_url)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.empty)

    def test_get_volatility_from_api(self):
        """Test get_volatility_from_api with mock response."""
        mock_response = pd.DataFrame({"Strike": [4000, 4500, 5000], "Volatility": [0.15, 0.2, 0.25]})
        with patch("src.pybacktestchain.data_module.get_volatility_from_api", return_value=mock_response) as mock_get_volatility:
            result = get_volatility_from_api(self.start_date, self.ticker, self.base_url)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.empty)

    def test_get_index_data_vol(self):
        """Test get_index_data_vol with mock responses."""
        mock_data = pd.DataFrame({
            "Date": ["2024-10-01", "2024-10-02"],
            "Close": [4500, 4520],
            "Volatility": [0.15, 0.2],
            "ticker": [self.ticker, self.ticker],
        })
        with patch("src.pybacktestchain.data_module.get_index_data_vol", return_value=mock_data) as mock_get_index_data_vol:
            result = get_index_data_vol(self.ticker, self.start_date, self.end_date, 1.0, self.base_url)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.empty)

    def test_black_scholes(self):
        """Test the Black-Scholes option pricing function."""
        spot_price = 4500  
        strike_price = 4400  
        T = 21 / 365  
        r = 0.03  
        sigma = 0.2  
        call_price = Information.black_scholes(spot_price, strike_price, T, r, sigma, "call")
        put_price = Information.black_scholes(spot_price, strike_price, T, r, sigma, "put")
        self.assertGreater(call_price, 0, "Call price should be greater than 0.")
        self.assertGreater(put_price, 0, "Put price should be greater than 0.")

    def test_compute_delta(self):
        """Test the delta computation function."""
        spot_price = 4500  
        strike_price = 4400  
        T = 21 / 365  
        r = 0.03  
        sigma = 0.2  
        call_delta = Information.compute_delta(spot_price, strike_price, T, r, sigma, "call")
        put_delta = Information.compute_delta(spot_price, strike_price, T, r, sigma, "put")
        self.assertGreaterEqual(call_delta, 0)
        self.assertLessEqual(call_delta, 1)
        self.assertLessEqual(put_delta, 0)
        self.assertGreaterEqual(put_delta, -1)

    def test_get_stocks_data(self):
        """Test get_stocks_data with mock response."""
        tickers = ["AAPL", "MSFT"]
        mock_data = pd.DataFrame({
            "Date": ["2024-10-01", "2024-10-02"],
            "Close": [150, 152],
            "ticker": ["AAPL", "AAPL"],
        })
        with patch("src.pybacktestchain.data_module.get_stocks_data", return_value=mock_data) as mock_get_stocks_data:
            result = get_stocks_data(tickers, self.start_date, self.end_date)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertFalse(result.empty)
            self.assertIn("ticker", result.columns)

if __name__ == "__main__":
    unittest.main()
