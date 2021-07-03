# 指標視覺化分析頁面
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

# 主頁面內容
page_map_analysis = html.Div([

    # 嵌入PowerBI
    html.Iframe(src="https://app.powerbi.com/view?r=eyJrIjoiNmEwNDJlOTktMWQwZS00MGVhLTllZjYtNTVhOGU5MjRlMTNjIiwidCI6ImU3NTkxNTI3LWQ2MzctNDZmMC1hYzNjLTRhODE3OWVlNTA1OSIsImMiOjEwfQ%3D%3D",
                style={"height": "720px", "width": "100%"})
])