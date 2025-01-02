import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import requests
import subprocess
import time
import pandas as pd
from flask_app.utils import start_flask_app,start_ngrok


def get_data_api(date, name, base_url):
    """Fetch data from the Flask API based on date and index name and display as a DataFrame."""
    try:
        response = requests.get(f"{base_url}/api/data", params={"date": date, "index": name})
        
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                print(f"\nData for {name} on {date}:")
                print(df)
            else:
                print(f"No data available for {name} on {date}.")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")

if __name__ == "__main__":
    flask_app_path = '/Users/dusicabajalica/Desktop/M2/Courses/Python/pybacktestchain/pybacktestchain/flask_app/app.py'
    flask_process = start_flask_app(flask_app_path)
    ngrok_url = start_ngrok()
    if ngrok_url:
        
        print(f"ngrok is running at: {ngrok_url}")
        
        
        selected_date = "2024-10-26"
        selected_name = "Euro Stoxx 50"
        
        get_data_api(selected_date, selected_name, ngrok_url)
    else:
        print("Could not start ngrok or fetch the URL.")

