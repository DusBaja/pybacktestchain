import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime
from Backtest_app.backtest_logic import run_backtest  # Import the backtest function
from datetime import datetime

app = dash.Dash(__name__)


app.layout = html.Div([
    html.H1('Backtest Dashboard'),
    html.Div([
        dcc.DatePickerRange(
            id='date-picker',
            start_date=datetime(2019, 1, 1),
            end_date=datetime(2020, 1, 1)
        ),
        dcc.Dropdown(
            id='strategy-dropdown',
            options=[{'label': 'Cash', 'value': 'cash'}, {'label': 'Vol', 'value': 'vol'}],
            value='cash'
        ),
        html.Button('Run Backtest', id='run-btn', n_clicks=0)
    ]),
    html.Div(id='backtest-output'),
])

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
        
        
        block_chain_str, is_valid = run_backtest(start_date, end_date, strategy_type)
        
        return html.Div([
            html.P(f"Blockchain: {block_chain_str}"),
            html.P(f"Is blockchain valid? {is_valid}")
        ])
    return "Press 'Run Backtest' to see results"
    


if __name__ == '__main__':
    app.run_server(debug=True)
