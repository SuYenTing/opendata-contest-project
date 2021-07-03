# 首頁
import dash_bootstrap_components as dbc
import dash_html_components as html

# 卡片內容
card_movie = [
    dbc.CardHeader("網站影片介紹"),
    dbc.CardBody([
            html.Iframe(
                src="https://www.youtube.com/embed/9o5-INL75sw",
                style={"height": "315px", "width": "560px"})
        ])
]

card_introduction = [
    dbc.CardHeader("服務簡介"),
    dbc.CardBody([
            html.P(
                "我國近年來少子女化情況日益嚴重，依據內政部2020年的人口數統計，台灣人口數首度出現負成長。"
                "行政院在我國少子女化對策計畫(107年-113年)內容中指出，少子女化會有四個影響：第一個是加速人口結構失衡、第二個是在學人數下降，"
                "衝擊教育體系、第三個是勞動人口減少，影響經濟發展、第四個是總扶養比增加，青壯年人口的撫養負擔加重。由上述可知，"
                "少子女化是一個嚴重的國安危機，為協助政府解決少子女化問題，本網站提供三項工具：指標視覺化、模型預測與分析及輿情系統。",
                className="card-text"),
        ])
]

card_map_analysis = [
    dbc.CardHeader("指標視覺化分析"),
    dbc.CardBody([
            html.P('我們將少子女化指標區分為五大類，分別為人口結構、婚育概況、托育教保、就業經濟及房市狀況。並以視覺化方式呈現各面向相關指標資訊，協助政府能掌握各地狀況。',
                   className="card-text"),
        ])
]

card_model_analysis = [
    dbc.CardHeader("模型預測與分析"),
    dbc.CardBody([
            html.P('使用內政部大數據模擬資料，透過機器學習模型預測已婚適育婦女是否有小孩，並以機器學習可解釋模型來分析影響生育率之因素，輔以視覺化呈現，'
                   '讓使用者能夠快速讀懂模型分析結果，掌握解決少子女化問題關鍵點。',
                   className="card-text"),
        ])
]

card_opinion_analysis = [
    dbc.CardHeader("輿情系統"),
    dbc.CardBody([
            html.P('透過爬蟲蒐集各大論壇(例如PTT、Dcard及BabyHome等)與少子女化議題相關之文章，'
                   '進行聲量分析與文字探勘。提供政府單位在政策制定或推行時，有民意可供參考。',
                   className="card-text"),
        ])
]

# 主頁面內容
page_index = html.Div([

    dbc.Row([
        dbc.Col([
            html.Center(html.Img(src="assets/img/index_fig.jpg", height='550px')),
            html.Center(html.Span(['(Photo by ',
                                   html.A('Jacky Zhao', href='https://unsplash.com/@jackyzha0?utm_source=unsplash&'
                                                             'utm_medium=referral&utm_content=creditCopyText'),
                                   ' On ',
                                   html.A('Unsplash', href='https://unsplash.com/s/photos/kid-play?utm_source='
                                                           'unsplash&utm_medium=referral&utm_content=creditCopyText'),
                                   ')'
                                   ], style={'color': 'gray', 'font-size': '8px'}))
        ], md=4),

        dbc.Col([

            # # 首頁標題
            # html.H2('Circle of Life:少子女化保衛戰', style={'color': '#3271e7'}),
            # html.Br(),

            # 影片介紹
            dbc.Row([dbc.Col(dbc.Card(card_movie))]),
            html.Br(),

            # 專題簡介
            dbc.Row([dbc.Col(dbc.Card(card_introduction))]),
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(card_map_analysis), md=4),
                dbc.Col(dbc.Card(card_model_analysis), md=4),
                dbc.Col(dbc.Card(card_opinion_analysis), md=4),
            ]),

        ], md=8),
    ], no_gutters=True),
])

