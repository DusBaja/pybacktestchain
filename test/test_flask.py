import requests
import subprocess
import time
import json
import pandas as pd
from flask.app import * 
def start_ngrok():
    """Start ngrok and retrieve the public URL dynamically."""
    ngrok_process = subprocess.Popen(['ngrok', 'http', '5000'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(5)
    
    # Fetch the public URL from the ngrok API (this works because ngrok exposes a local API on http://localhost:4040)
    try:
        url_response = requests.get('http://localhost:4040/api/tunnels')
        url_data = url_response.json()
        public_url = url_data['tunnels'][0]['public_url']
        return public_url
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ngrok URL: {e}")
        return None

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
    
    ngrok_url = start_ngrok()
    if ngrok_url:
        
        print(f"ngrok is running at: {ngrok_url}")
        
        
        selected_date = "2024-10-26"
        selected_name = "Euro Stoxx 50"
        
        get_data_api(selected_date, selected_name, ngrok_url)
    else:
        print("Could not start ngrok or fetch the URL.")

