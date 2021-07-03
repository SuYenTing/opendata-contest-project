import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0'}]
                )
# app = dash.Dash(external_stylesheets=[dbc.themes.COSMO])
