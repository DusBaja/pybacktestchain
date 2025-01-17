import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../flask_app/')))


import unittest
import requests
import subprocess
import time
import os
from flask_app.utils import start_flask_app, start_ngrok


class FlaskAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the Flask app and ngrok tunnel before running tests."""
        flask_app_path = '/Users/dusicabajalica/Desktop/M2/Courses/Python/pybacktestchain/pybacktestchain/flask_app/app.py'
        cls.flask_process = start_flask_app(flask_app_path)  # Start Flask app
        cls.ngrok_url = start_ngrok()  # Start ngrok

        if not cls.ngrok_url:
            raise RuntimeError("Could not start ngrok or fetch the URL.")
        print(f"ngrok is running at: {cls.ngrok_url}")
        
        # Give the Flask app some time to start
        time.sleep(3)

    @classmethod
    def tearDownClass(cls):
        """Clean up by terminating Flask app and ngrok processes."""
        if cls.flask_process:
            cls.flask_process.terminate()
            cls.flask_process.wait()
        print("Flask app process terminated.")

    def get_data_api(self, date, name):
        """Fetch data from the Flask API based on date and index name."""
        try:
            response = requests.get(f"{self.ngrok_url}/api/data", params={"date": date, "index": name})
            self.assertEqual(response.status_code, 200, f"Unexpected status code: {response.status_code}")
            
            data = response.json()
            self.assertIsInstance(data, list, "API response should return a list.")
            print(f"\nData for {name} on {date}: {data}")
        except Exception as e:
            self.fail(f"An error occurred while fetching data: {e}")

    def test_get_data_valid_input(self):
        """Test the API with valid date and index."""
        selected_date = "2024-10-26"
        selected_name = "Euro Stoxx 50"
        self.get_data_api(selected_date, selected_name)

    def test_get_data_invalid_input(self):
        """Test the API with an invalid date or index."""
        invalid_date = "invalid-date"
        invalid_name = "Invalid Index"
        try:
            response = requests.get(f"{self.ngrok_url}/api/data", params={"date": invalid_date, "index": invalid_name})
            self.assertNotEqual(response.status_code, 200, "API should not return 200 for invalid input.")
            print(f"Invalid input test returned status code: {response.status_code}")
        except Exception as e:
            self.fail(f"An error occurred while testing invalid input: {e}")


if __name__ == "__main__":
    unittest.main()
