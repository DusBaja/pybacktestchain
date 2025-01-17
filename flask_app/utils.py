import subprocess
import time
import logging
import requests

def start_flask_app(flask_app_path):
    """
    Start the Flask application by running app.py.

    :param flask_app_path: Path to the Flask app to be executed (e.g., 'app.py')
    :return: Process object for the Flask app.
    """
    try:
        
        flask_process = subprocess.Popen(
            ['python', flask_app_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(10)  # Give it time to start the app
        return flask_process
    except Exception as e:
        logging.error(f"Error starting Flask app: {e}")
        return None
#To access our dynamic link for our vol surface (changing everytime because the free version):

#quit us one day before the end of the project !! Needed a new account 
def start_ngrok():
    """Start ngrok and retrieve the public URL dynamically from the code we run on our server."""
    # Start ngrok process in the background
    ngrok_process = subprocess.Popen(['ngrok', 'http', '5000'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)
    
    try:
        url_response = requests.get('http://localhost:4040/api/tunnels')
        url_data = url_response.json()
        public_url = url_data['tunnels'][0]['public_url']
        return public_url
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ngrok URL: {e}")
        return None
