import subprocess
import time
import logging
import requests
import os 
def start_flask_app():
    """
    Dynamically checks if Flask is running on port 5000; if not, starts the app with nohup.

    Start the Flask application by running app.py.

    :return: Process object for the Flask app.
    """
    try:
        
        # First, check if Flask is already running on port 5000
        check_process = subprocess.run(['lsof', '-i', ':5000'], capture_output=True, text=True)
        if check_process.stdout:
            print("Flask is already running on port 5000.")
            return

        # Start Flask with nohup
        current_dir = os.path.dirname(os.path.abspath(__file__))
        flask_app_path = os.path.join(current_dir, 'app.py')

        utils_dir = os.path.abspath(os.path.dirname(__file__))  # Absolute path of the current utils.py file
        nohup_out_path = os.path.join(utils_dir, 'nohup.out')
        os.makedirs(utils_dir, exist_ok=True)
        print("Flask not running. Starting with nohup...")
        print(f"Flask app path being used: {flask_app_path}")
        print(f"Nohup out path: {nohup_out_path}")
        with open(nohup_out_path, 'a') as log_file:
            subprocess.Popen(
                ['nohup', 'poetry', 'run', 'python', flask_app_path],
                stdout=log_file,  # Write stdout to nohup.out
                stderr=log_file,  # Write stderr to nohup.out
                preexec_fn=os.setpgrp  # Prevent process termination on parent exit
            )

        print("Waiting for Flask to initialize...")
        time.sleep(15)  # time for Flask to initialize

        # Secondly, verify flask is running
        check_process = subprocess.run(['lsof', '-i', ':5000'], capture_output=True, text=True)
        if not check_process.stdout:
            raise RuntimeError("Flask failed to start on port 5000.")

        print("Flask app started successfully.")
  
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
