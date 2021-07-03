# 模型分析與預測頁面
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

import pickle
import plotly.graph_objects as go

# # 讀取模型說明markdown
# with open('assets/model_data/model_instruction.md', encoding='utf-8') as f:
#     modelInstructionMarkdown = f.read()

# 讀取數據
with open('assets/model_data/web_ml_model_data.pkl', 'rb') as f:
    y_train, y_test, xgboost_train_pred, xgboost_test_pred, shapImportanceDf, villageFeatureInfo, laborForceData, featureChName = pickle.load(f)

# 繪製特徵重要程度圖形
feature_names = shapImportanceDf.index.tolist()
importance_values = shapImportanceDf.tolist()
data = [go.Bar(y=feature_names,
               x=importance_values,
               orientation='h')]
layout = go.Layout(
    title='特徵重要程度(Mean absolute SHAP value)',
    plot_bgcolor='#fff',
    showlegend=False
)
featureImportanceFig = go.Figure(data=data, layout=layout)
featureImportanceFig.update_yaxes(automargin=True)
featureImportanceFig.update_xaxes(automargin=True)
featureImportanceFig.update_layout(height=200+len(feature_names)*20,
                                   margin=go.layout.Margin(l=100, r=50, b=50, t=50, pad=4),
                                   title_x=0.5,
                                   font_family='Noto Sans TC')

# 分析說明頁籤內容
description_content = dbc.CardBody([

    dbc.Row([

        dbc.Col([
            # dcc.Markdown(modelInstructionMarkdown)
            html.Img(src="assets/model_data/model_intro_fig/model_intro_fig_1.PNG", height='500px'),
            html.Img(src="assets/model_data/model_intro_fig/model_intro_fig_2.PNG", height='500px'),
            html.Img(src="assets/model_data/model_intro_fig/model_intro_fig_3.PNG", height='500px'),
            html.Img(src="assets/model_data/model_intro_fig/model_intro_fig_4.PNG", height='500px'),
            html.Img(src="assets/model_data/model_intro_fig/model_intro_fig_5.PNG", height='500px'),
            html.Img(src="assets/model_data/model_intro_fig/model_intro_fig_6.PNG", height='500px'),
            html.Img(src="assets/model_data/model_intro_fig/model_intro_fig_7.PNG", height='500px'),
            html.Img(src="assets/model_data/model_intro_fig/model_intro_fig_8.PNG", height='500px'),
            html.Img(src="assets/model_data/model_intro_fig/model_intro_fig_9.PNG", height='500px'),
            html.Img(src="assets/model_data/model_intro_fig/model_intro_fig_10.PNG", height='500px'),
            html.Img(src="assets/model_data/model_intro_fig/model_intro_fig_11.PNG", height='500px')
        ], md=10)
    ]),
])

# 預測績效頁籤內容
performance_content = dbc.CardBody([

    # 預測分類門檻值調整
    dbc.Row([
        dbc.Card([

            dbc.CardHeader([
                html.H4("預測門檻機率值調整", className="card-title"),
                html.H6("超過此門檻值即預測為「有小孩」分類", className="card-subtitle"),
            ]),

            dbc.CardBody(
                [
                    dcc.Slider(
                        id="predict_threshold",
                        min=0,
                        max=1,
                        step=0.1,
                        marks={i/10: '{}'.format(i/10) for i in range(10)},
                        value=0.5
                    )
                ]
            )
        ], style={"width": "100%"})
    ]),

    html.Br(),

    dbc.Row([

        # 模型預測績效指標
        dbc.Col([
            dbc.Card([

                dbc.CardHeader([
                    html.H4("模型預測績效指標", className="card-title"),
                ]),

                dbc.CardBody([
                    html.Div(id="model_performance_metrics")
                ])
            ], style={"height": "650px"})
        ], md=6),

        # 混淆矩陣
        dbc.Col([
            dbc.Card([

                dbc.CardHeader([
                    html.H4("混淆矩陣", className="card-title"),
                ]),

                dbc.CardBody([
                    dcc.RadioItems(
                        id="model_confusion_dataset",
                        options=[
                            {'label': '訓練集', 'value': 'train'},
                            {'label': '測試集', 'value': 'test'}
                        ],
                        value="train",
                        labelStyle={'display': 'block'}
                    ),
                    dcc.Graph(id="model_confusion_matrix")
                ])
            ], style={"height": "650px"})
        ], md=6)
    ]),

    html.Br(),

    dbc.Row([

        # ROC曲線
        dbc.Col([
            dbc.Card([

                dbc.CardHeader([
                    html.H4("ROC曲線", className="card-title"),
                ]),

                dbc.CardBody([
                    dcc.RadioItems(
                        id="model_roc_curve_dataset",
                        options=[
                            {'label': '訓練集', 'value': 'train'},
                            {'label': '測試集', 'value': 'test'}
                        ],
                        value="train",
                        labelStyle={'display': 'block'}
                    ),
                    dcc.Graph(id="model_roc_curve")
                ])
            ], style={"height": "650px"})
        ], md=6),

        dbc.Col([

        ], md=6)
    ]),
])

# 重要特徵頁籤內容
feature_importance_content = dbc.CardBody([

    dbc.Row([

        dbc.Col([
            dbc.Card([

                dbc.CardHeader([
                    html.H4("特徵重要程度", className="card-title"),
                    html.H6("哪個特徵最具有影響力?", className="card-subtitle"),
                ]),

                dbc.CardBody([
                    dcc.Graph(id="featureImportanceFig", figure=featureImportanceFig)
                ])
            ], style={"height": "600px"})
        ]),

        dbc.Col([
            dbc.Card([

                dbc.CardHeader([
                    html.H4("SHAP特徵影響力", className="card-title"),
                    html.H6("特徵如何影響預測結果?", className="card-subtitle"),
                ]),

                dbc.CardBody([
                    html.Img(src="assets/model_data/xgboost_shap.png")
                ])
            ], style={"height": "600px"})
        ])
    ])
])

# 縣市選單
countyDropdownOptions = ['基隆市', '臺北市', '新北市', '桃園市', '新竹市', '新竹縣', '苗栗縣', '臺中市', '彰化縣',
                         '南投縣', '雲林縣', '嘉義市', '嘉義縣', '臺南市', '高雄市', '屏東縣', '臺東縣', '花蓮縣',
                         '宜蘭縣', '澎湖縣', '金門縣', '連江縣']

# 樣本分析頁籤內容
sample_analysis_content = dbc.CardBody([

    # 輸入特徵值區域
    dbc.Row([
        dbc.Card([

            dbc.CardHeader([
                html.H4("調整特徵值觀察模型預測過程", className="card-title")
            ]),

            dbc.CardBody([
                dbc.Row([

                    dbc.Col([html.Span('請選取縣市：'),
                             dcc.Dropdown(id='modelCountyDropdown',
                                          options=[{'value': x, 'label': x} for x in countyDropdownOptions],
                                          value='臺北市',
                                          clearable=False)]),

                    dbc.Col([html.Span('請選取鄉鎮市區：'),
                             dcc.Dropdown(id='modelTownDropdown', value='中正區', clearable=False)]),

                    dbc.Col([html.Span('請選取村里：'),
                             dcc.Dropdown(id='modelVillageDropdown', value='建國里', clearable=False)])
                ]),

                html.Br(),

                dbc.Row([

                    dbc.Col([html.Span('請輸入年紀：'),
                             html.Br(),
                             dcc.Input(id="feature_age", type="number", min=18, max=40, value=25)
                             ]),

                    dbc.Col([html.Span('請選擇教育程度：'),
                             dcc.Dropdown(id='feature_education',
                                          options=[{'value': 1, 'label': '國中以下'},
                                                   {'value': 2, 'label': '高中職'},
                                                   {'value': 3, 'label': '專科'},
                                                   {'value': 4, 'label': '大學'},
                                                   {'value': 5, 'label': '碩博士'}],
                                          value='4',
                                          clearable=False)
                             ]),

                    dbc.Col([html.Span('是否有殼：'),
                             dcc.RadioItems(
                                 id="feature_have_house",
                                 options=[
                                     {'label': '有', 'value': '1'},
                                     {'label': '否', 'value': '0'}
                                 ],
                                 value="1",
                                 labelStyle={'display': 'block'})
                             ])
                ]),

                html.Br(),

                dbc.Row([

                    dbc.Col([html.Span('是否為身心障礙者：'),
                             dcc.RadioItems(
                                 id="feature_have_disability",
                                 options=[
                                     {'label': '有', 'value': '1'},
                                     {'label': '否', 'value': '0'}
                                 ],
                                 value="0",
                                 labelStyle={'display': 'block'})
                             ]),

                    dbc.Col([html.Span('是否為中低收入戶：'),
                             dcc.RadioItems(
                                 id="feature_have_low_type",
                                 options=[
                                     {'label': '有', 'value': '1'},
                                     {'label': '否', 'value': '0'}
                                 ],
                                 value="0",
                                 labelStyle={'display': 'block'})
                             ]),

                    dbc.Col([html.Span('是否有原住民身份：'),
                             dcc.RadioItems(
                                 id="feature_living_type",
                                 options=[
                                     {'label': '有', 'value': '1'},
                                     {'label': '否', 'value': '0'}
                                 ],
                                 value="0",
                                 labelStyle={'display': 'block'})
                             ])
                ]),

            ])

        ], style={"width": "100%"})
    ]),

    html.Br(),

    # 特徵貢獻圖
    dbc.Row([

        dbc.Col([

            dbc.Card([

                dbc.CardHeader([
                    html.H4("模型預測結果", className="card-title")
                ]),

                dbc.CardBody([
                    html.Div(id="predictProbTable")
                ])
            ], style={"width": "100%", "height": "650px"})
        ], md=4),

        dbc.Col([
            dbc.Card([

                dbc.CardHeader([
                    html.H4("特徵貢獻圖", className="card-title"),
                    html.H6("每個特徵如何影響預測結果?", className="card-subtitle"),
                ]),

                dbc.CardBody([
                    dcc.Graph(id="featureContributionsFig")
                ])
            ], style={"width": "100%", "height": "650px"})
        ], md=8),
    ])
]),


# 樣本分析頁籤內容
town_analysis_content = dbc.CardBody([

    # 輸入特徵值區域
    dbc.Row([
        dbc.Col([
            dbc.Card([

                dbc.CardHeader([
                    html.H4("以鄉鎮市區角度觀察模型如何預測", className="card-title"),
                ]),

                dbc.CardBody([
                    dbc.Row([

                        dbc.Col([html.Span('請選取縣市：'),
                                 dcc.Dropdown(id='townAnalysisCountyDropdown',
                                              options=[{'value': x, 'label': x} for x in countyDropdownOptions],
                                              value='臺北市',
                                              clearable=False)]),

                        dbc.Col([html.Span('請選取鄉鎮市區：'),
                                 dcc.Dropdown(id='townAnalysisTownDropdown', value='中正區', clearable=False)])
                    ])
                ])
            ], style={"width": "50%"}),

        ])
    ]),

    html.Br(),

    # 輸出SHAP圖形
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("SHAP特徵影響力", className="card-title"),
                    html.H6("特徵如何影響預測結果?", className="card-subtitle"),
                ]),

                dbc.CardBody([
                    html.Img(id='townFeatureAnalysisFig')
                ])
            ], style={"width": "50%"}),
        ]),
    ])
]),

# 主頁面內容
page_model_analysis = html.Div([

    # 子標籤頁面
    dbc.Tabs([
        dbc.Tab(description_content, label="模型說明"),
        dbc.Tab(performance_content, label="預測績效"),
        dbc.Tab(feature_importance_content, label="重要特徵"),
        dbc.Tab(sample_analysis_content, label="樣本分析"),
        dbc.Tab(town_analysis_content, label="鄉鎮市區分析"),
    ])
])
