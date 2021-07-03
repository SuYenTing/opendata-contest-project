# 專題網站程式碼
# navbar程式碼參考
# https://github.com/facultyai/dash-bootstrap-components/blob/main/examples/advanced-component-usage/Navbars.py

# 載入套件
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# 載入頁面
from app import app
from page_index import page_index
# from page_map_analysis import page_map_analysis
from page_model_analysis import page_model_analysis
from page_map_analysis_bi import page_map_analysis
from page_opinion_analysis import page_opinion_analysis
# from page_under_construction import page_under_construction
import callbacks


# this example that adds a logo to the navbar brand
app.layout = html.Div([

    # navbar
    dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src="assets/img/baby.png", height="50px")),
                            dbc.Col(dbc.NavbarBrand("Circle of Life: 少子女化保衛戰", className="ml-2")),
                        ],
                        align="center",
                        no_gutters=True,
                    ),
                    href="/",
                ),
                dbc.NavbarToggler(id="navbar-toggler-menu"),
                dbc.Collapse(
                    dbc.Nav([dbc.NavItem(dbc.NavLink("首頁", href="/", active="exact")),
                             dbc.NavItem(dbc.NavLink("指標視覺化分析", href="/map_analysis", active="exact")),
                             dbc.NavItem(dbc.NavLink("模型預測與分析", href="/model_analysis", active="exact")),
                             dbc.NavItem(dbc.NavLink("輿情系統", href="/public_opinion", active="exact")),
                             ],
                            className="ml-auto", navbar=True),
                    id="navbar-collapse-menu",
                    navbar=True,
                ),
            ]
        ),
        color="dark",
        dark=True,
        className="mb-5",
    ),

    # page content
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


# toggle the collapse on small screens
@app.callback(
    Output(f"navbar-collapse-menu", "is_open"),
    [Input(f"navbar-toggler-menu", "n_clicks")],
    [State(f"navbar-collapse-menu", "is_open")])
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# navbar link
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):

    if pathname == "/":
        return page_index
    elif pathname == "/map_analysis":
        # return page_map_analysis
        return page_map_analysis
    elif pathname == "/model_analysis":
        return page_model_analysis
    elif pathname == "/public_opinion":
        return page_opinion_analysis
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
    # app.run_server(host='0.0.0.0', debug=False, port=8050)
