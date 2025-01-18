import sys
import os
import time
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime
from src.pybacktestchain_vol_cash.data_module import FirstTwoMoments, ShortSkew, Momentum
from src.pybacktestchain_vol_cash.broker import Backtest, StopLoss
from src.pybacktestchain_vol_cash.blockchain import load_blockchain
import dash_bootstrap_components as dbc

# Initialize Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Layout for the Dash app
app.layout = html.Div([
    # Main Header
    html.H1('Backtest Dashboard', style={'textAlign': 'center', 'color': '#ffffff'}),

    # Date Picker, Strategy Dropdown, and Verbose Toggle inside a styled container
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=datetime(2024, 10, 1),
                    end_date=datetime(2025, 1, 10),
                    display_format='YYYY-MM-DD',
                    style={'width': '100%', 'marginBottom': '20px'}
                ),
            ], width=4),
            dbc.Col([
                dcc.Dropdown(
                    id='strategy-type-dropdown',
                    options=[
                        {'label': 'Cash', 'value': 'cash'},
                        {'label': 'Volatility', 'value': 'vol'}
                    ],
                    value='cash',
                    style={'width': '100%', 'marginBottom': '20px'}
                ),
            ], width=4),
            dbc.Col([
                dcc.Dropdown(
                    id='class-dropdown',
                    options=[
                        {'label': 'FirstTwoMoments', 'value': 'FirstTwoMoments'},
                        {'label': 'Momentum', 'value': 'Momentum'},
                        {'label': 'ShortSkew', 'value': 'ShortSkew'}
                    ],
                    value='FirstTwoMoments',
                    style={'width': '100%', 'marginBottom': '20px'}
                ),
            ], width=4)
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Checklist(
                    id='verbose-toggle',
                    options=[{'label': 'Verbose', 'value': True}],
                    value=[],
                    style={'marginBottom': '20px', 'color': 'white'}
                ),
            ], width=12)
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
    [Input('date-picker', 'start_date'), Input('date-picker', 'end_date'), Input('strategy-type-dropdown', 'value'), Input('class-dropdown', 'value'), Input('verbose-toggle', 'value')]
)
def display_results(n_clicks, start_date, end_date, strategy_type, class_name, verbose_list):
    if n_clicks > 0:
        # Parse the dates
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Determine verbosity
        verbose = True if True in verbose_list else False
        
        indices = ["^STOXX50E", "^GSPC"] if strategy_type == 'vol' else []

        # Determine information class
        if class_name == 'FirstTwoMoments':
            information_class = FirstTwoMoments
        elif class_name == 'Momentum':
            information_class = lambda **kwargs: Momentum(
                **{
                    "indices": indices,
                    "strategy_type": strategy_type,
                    **kwargs
                }
            )
        elif class_name == 'ShortSkew':
            information_class = lambda **kwargs: ShortSkew(
                **{
                    "indices": indices,
                    "strategy_type": strategy_type,
                    **kwargs
                }
            )
        else:
            return "Invalid class selected."

        # Run backtest
        name_blockchain = 'backtest'
        backtest = Backtest(
            initial_date=start_date,
            final_date=end_date,
            strategy_type=strategy_type,
            information_class=information_class,
            risk_model=StopLoss,
            name_blockchain=name_blockchain,
            verbose=verbose
        )

        start_time = time.time()
        backtest.run_backtest()

        # Load transaction log and calculate P&L
        try:
            backtest_name = backtest.backtest_name
            transaction_log_path = f"backtests/{backtest_name}.csv"
            df = pd.read_csv(transaction_log_path)
            PnL = backtest.PnL
            

            # Display results
            return html.Div([
                html.H4("Transaction Log:"),
                html.Pre(df.to_string(index=False)),
                html.H4(f"Final P&L: {PnL}"),
                html.P(f"Execution Time: {time.time() - start_time:.2f} seconds", style={'fontSize': '16px'})
            ])
        except Exception as e:
            return f"Error loading transaction log: {e}"

    return "Press 'Run Backtest' to see results."

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
