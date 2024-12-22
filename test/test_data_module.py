import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
import pandas as pd
from datetime import datetime, timedelta
from pybacktestchain.data_module import *
from pybacktestchain.data_module import get_index_data_vol
flask_process = start_flask_app()
ngrok_url = start_ngrok()


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
        result = get_data_api(self.start_date,  self.ticker,self.ngrok_url)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty, "Data retrieval returned an empty DataFrame.")
        

    def test_volatility_computation(self):
        """Test volatility computation logic."""
        vol_data = get_volatility_from_api(self.start_date, self.ticker,  self.ngrok_url)
        self.assertIsInstance(vol_data, pd.DataFrame)
        self.assertFalse(vol_data.empty, "Volatility computation returned an empty DataFrame.")
        
    def test_information_integration(self):
        """Test integration of Information class with DataModule."""
        information = Information(
            s=timedelta(days=10),
            data_module=self.data_module,
            indices=["^GSPC", "^STOXX50E"]
        )
        test_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        prices = information.get_prices(test_date)
        self.assertIsInstance(prices, dict, "Prices should be a dictionary.")
        self.assertIn("^GSPC", prices, "Expected ^GSPC in the computed prices.")
        self.assertGreater(prices["^GSPC"], 0, "Price for ^GSPC should be greater than 0.")

    def test_edge_case_empty_data(self):
        """Test behavior with an empty dataset."""
        empty_data = pd.DataFrame(columns=["Date", "Close", "ImpliedVol", "ticker"])
        empty_module = DataModule(data=empty_data)
        information = Information(data_module=empty_module, indices=["^GSPC"])
        test_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        prices = information.get_prices(test_date)
        self.assertEqual(prices, {}, "Expected an empty dictionary when data is empty.")

    def test_multiple_tickers(self):
        """Test retrieval of multiple tickers."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        stock_data = get_stocks_data(tickers,self.start_date, self.end_date)
        self.assertIsInstance(stock_data, pd.DataFrame)
        self.assertEqual(
            len(stock_data["ticker"].unique()), len(tickers),
            "Expected data for all requested tickers."
        )


if __name__ == "__main__":
    unittest.main()
