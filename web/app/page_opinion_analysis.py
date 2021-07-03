# 指標視覺化分析頁面
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

from custom_functions import CreateDBEngine

# 建立議題選項
query = 'select distinct issue from web.popular_article;'
topicList = pd.read_sql(query, con=CreateDBEngine())
topicList = [{'label': topicList['issue'][i], 'value': topicList['issue'][i]} for i in range(len(topicList))]

# 建立觀察期間選項
query = 'select distinct sday from web.popular_article;'
periodList = pd.read_sql(query, con=CreateDBEngine())
periodList = [{'label': periodList['sday'][i], 'value': periodList['sday'][i]} for i in range(len(periodList))]

# 主頁面內容
page_opinion_analysis = html.Div([

    # 分析議題
    dbc.Row([
        dbc.Card([

            dbc.CardHeader([
                html.H4("輿情分析", className="card-title"),
                html.H6("最近大家都在討論什麼少子女化議題?", className="card-subtitle"),
            ]),

            dbc.CardBody([

                dbc.Row([
                    dbc.Col([
                        html.Span('請選取議題：'),
                        dcc.Dropdown(id='pubicOpinionTopicDropdown',
                                     options=topicList,
                                     value='少子女化',
                                     clearable=False),
                    ], md=4),

                    dbc.Col([
                        html.Span('請選取觀察期間(天)：'),
                        dcc.Dropdown(id='pubicOpinionPeriodDropdown',
                                     options=periodList,
                                     value='30',
                                     clearable=False)
                    ], md=4)
                ])
            ])

        ], style={"width": "100%"})
    ]),

    html.Br(),

    # 聲量分析
    dbc.Row([

        dbc.Card([

            dbc.CardHeader([
                html.H4("議題聲量分析", className="card-title"),
                html.H6("最近這個議題常被討論度嗎?", className="card-subtitle"),
            ]),

            dbc.CardBody([
                dbc.Spinner(dcc.Graph(id="topicVolumeFig"), color="primary")
            ])
        ], style={"width": "100%"})
    ]),

    html.Br(),

    # 熱門文章
    dbc.Row([

        dbc.Card([

            dbc.CardHeader([
                html.H4("輿情分析", className="card-title"),
                html.H6("最近大家都在討論什麼內容?", className="card-subtitle"),
            ]),

            dbc.CardBody([

                # 各版熱門文章
                dbc.Tabs([

                    dbc.Tab(dbc.CardBody([

                        html.Div(id="pttGossipingHotArticle")

                    ]), label="PTT-八卦板"),

                    dbc.Tab(dbc.CardBody([

                        html.Div(id="pttBabyMotherHotArticle")

                    ]), label="PTT-親子板"),

                    dbc.Tab(dbc.CardBody([

                        html.Div(id="dcardTrendingHotArticle")

                    ]), label="Dcard-時事板"),

                    dbc.Tab(dbc.CardBody([

                        html.Div(id="dcardParentChildHotArticle")

                    ]), label="Dcard-親子板"),

                    dbc.Tab(dbc.CardBody([

                        html.Div(id="babyHomeHotArticle")

                    ]), label="BabyHome寶貝家庭親子網")
                ])
            ])
        ], style={"width": "100%"})
    ]),

    html.Br(),

    # 文字雲與關鍵字向量圖
    dbc.Row([

        # 文字雲
        dbc.Col([

            dbc.Card([
                dbc.CardHeader([
                    html.H4("文字雲", className="card-title"),
                    html.H6("從議題中找出關鍵字", className="card-subtitle"),
                ]),

                dbc.CardBody([
                    html.Img(id='wordCloudFig')
                ])
            ], style={"height": "650px", "width": "100%"})
        ]),

        # 關鍵字向量圖
        dbc.Col([

            dbc.Card([
                dbc.CardHeader([
                    html.H4("關鍵字向量圖", className="card-title"),
                    html.H6("觀察關鍵字的相近程度(採用word2Vec及TSNE模型)", className="card-subtitle"),
                ]),

                dbc.CardBody([

                    dcc.RadioItems(
                        id="wordVectorRadio",
                        options=[
                            {'label': '2D圖', 'value': '2D'},
                            {'label': '3D圖', 'value': '3D'}
                        ],
                        value="2D",
                        labelStyle={'display': 'block'}
                    ),

                    dbc.Spinner(dcc.Graph(id="wordVectorFig"), color="primary")
                ])
            ], style={"height": "700px", "width": "100%"})
        ])
    ]),
])
