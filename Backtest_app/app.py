import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime
from Backtest_app.backtest_logic import run_backtest  # Import the backtest function
from datetime import datetime
import dash_bootstrap_components as dbc

# Initialize Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Layout for the Dash app
app.layout = html.Div([
    # Main Header
    html.H1('Backtest Dashboard', style={'textAlign': 'center', 'color': '#ffffff'}),
    
    # Date Picker and Strategy Dropdown inside a styled container
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=datetime(2019, 1, 1),
                    end_date=datetime(2020, 1, 1),
                    display_format='YYYY-MM-DD',
                    style={'width': '100%', 'marginBottom': '20px'}
                ),
            ], width=6),
            dbc.Col([
                dcc.Dropdown(
                    id='strategy-dropdown',
                    options=[{'label': 'Cash', 'value': 'cash'}, {'label': 'Vol', 'value': 'vol'}],
                    value='cash',
                    style={'width': '100%', 'marginBottom': '20px'}
                ),
            ], width=6)
        ]),
        
        # Button to trigger the backtest
        dbc.Row([
            dbc.Col([
                html.Button('Run Backtest', id='run-btn', n_clicks=0, style={
                    'width': '100%', 'padding': '10px', 'background-color': '#007BFF', 'border': 'none', 'color': 'white', 'border-radius': '5px'
                }),
            ], width=12)
        ], style={'marginBottom': '20px'}),
        
        # Output Section
        dbc.Row([
            dbc.Col([
                html.Div(id='backtest-output', style={'color': 'white', 'padding': '20px', 'background': '#343a40', 'border-radius': '5px'})
            ])
        ]),
        
    ], fluid=True),
])

# Callback function to display the results
@app.callback(
    Output('backtest-output', 'children'),
    [Input('run-btn', 'n_clicks')],
    [Input('date-picker', 'start_date'), Input('date-picker', 'end_date'), Input('strategy-dropdown', 'value')]
)
def display_results(n_clicks, start_date, end_date, strategy_type):
    if n_clicks > 0:
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")  # If time is present
        except ValueError:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")  # If only date is present
        
        try:
            end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")  # If time is present
        except ValueError:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")  # If only date is present
        
        # Running the backtest
        block_chain_str, is_valid = run_backtest(start_date, end_date, strategy_type)
        
        # Displaying results in a more styled way
        return html.Div([
            html.P(f"Blockchain: {block_chain_str}", style={'fontSize': '16px'}),
            html.P(f"Is blockchain valid? {is_valid}", style={'fontSize': '16px'}),
        ])
    return "Press 'Run Backtest' to see results"

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
