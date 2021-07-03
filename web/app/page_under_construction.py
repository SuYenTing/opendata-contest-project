# 目前網站正在製作頁面
import dash_bootstrap_components as dbc
import dash_html_components as html

# 主頁面內容
page_under_construction = html.Div([
    dbc.Row([
        dbc.Col([
            html.Center(html.H1('此頁面正在製作中，敬請期待!')),
            html.Br(),
            html.Center(html.Img(src="assets/img/construction.jpg", height="300px"))
        ])
    ])
])