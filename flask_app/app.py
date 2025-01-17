from flask import Flask, jsonify, render_template, request
import pandas as pd
import os

app = Flask(__name__)

flask_app_root = os.path.abspath(os.path.dirname(__file__))
#data_folder = "/Users/dusicabajalica/Desktop/M2/Courses/Python/pybacktestchain/pybacktestchain/flask_app/Data_treated"
data_folder = os.path.join(flask_app_root, "Data_treated")

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    """API endpoint to return data from an Excel file based on date and index."""
    date_param = request.args.get('date')
    index_param = request.args.get('index')
    print(f"Received date: {date_param}, index: {index_param}")

    if not date_param or not index_param:
        return jsonify({"error": "Both 'date' and 'index' parameters are required"}), 400

    
    matching_files = []
    for file in os.listdir(data_folder):
        if file.endswith('.xlsx') and date_param in file and index_param in file:
            matching_files.append(file)

    if not matching_files:
        return jsonify({"error": f"No matching files found for {index_param} on {date_param}"}), 404

    
    selected_file = matching_files[0]
    file_path = os.path.join(data_folder, selected_file)

    try:
        
        df = pd.read_excel(file_path)
        df.columns = df.columns.astype(str)
        if "Unnamed: 0" in df.columns:
            df.rename(columns={"Unnamed: 0": "Strike/Maturity"}, inplace=True)
            df.set_index("Strike/Maturity", inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')  # Coerce errors to NaN
        df = df.fillna("N/A").replace([float('inf'), float('-inf')], "N/A")
        df = df.astype(str)
        df_reset = df.reset_index() 

        df_dict = df_reset.to_dict(orient='records')
        return jsonify(df_dict)

    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/api/files', methods=['GET'])
def list_files():
    """API endpoint to list all available Excel files."""
    try:
        
        files = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')]
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": f"Error listing files: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
