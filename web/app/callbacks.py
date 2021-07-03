# 後台處理函數
# 匯入套件
# import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

# 模型預測與分析所需之套件
import pickle
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix

# 匯入自定義程式碼
from app import app
from custom_functions import CreateDBEngine

# 讀取前置作業數據
# 讀取數據
with open('assets/model_data/web_ml_model_data.pkl', 'rb') as f:
    y_train, y_test, xgboost_train_pred, xgboost_test_pred, shapImportanceDf, villageFeatureInfo, laborForceData, featureChName = pickle.load(f)

# 讀取模型
xgbModel = xgb.Booster()
xgbModel.load_model("assets/model_data/xgboost_model.txt")

# 進行SHAP分析
explainer = shap.TreeExplainer(xgbModel)

# 圖片共用設定
figConfigLayout = {'title_x': 0.5,
                   'showlegend': True,
                   'hovermode': 'x unified',
                   'font_family': 'Noto Sans TC',
                   # 'legend': dict(title='', orientation='h', yanchor='top', xanchor='center', y=1.1, x=0.5),
                   'xaxis': dict(tickformat=',d', dtick=1),
                   'yaxis': dict(title='')}
figNumsConfigLayout = {'font_family': 'Noto Sans TC',
                       'height': 250,
                       'width': 300,
                       'margin': dict(l=0, r=0, t=50, b=0)}


# 模型預測績效指標
@app.callback(Output("model_performance_metrics", "children"),
              [Input("predict_threshold", "value")])
def train_model_performance_metrics(cutoff):

    # 績效指標函數
    def performance_eval(y_true, y_score, cutoff=0.5):
        y_pred = np.where(y_score > cutoff, 1, 0)
        return pd.DataFrame({'metric': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                             'score': [round(accuracy_score(y_true, y_pred), 4),
                                       round(precision_score(y_true, y_pred), 4),
                                       round(recall_score(y_true, y_pred), 4),
                                       round(f1_score(y_true, y_pred), 4),
                                       round(roc_auc_score(y_true, y_score), 4)]})

    performanceEvalData = pd.merge(performance_eval(y_true=y_train, y_score=xgboost_train_pred, cutoff=cutoff),
                                   performance_eval(y_true=y_test, y_score=xgboost_test_pred, cutoff=cutoff),
                                   on=['metric'])
    performanceEvalData.columns = ['指標', '訓練集分數', '測試集分數']
    return dbc.Table.from_dataframe(performanceEvalData)


# 繪製混淆矩陣
@app.callback(Output("model_confusion_matrix", "figure"),
              Input("model_confusion_dataset", "value"),
              Input("predict_threshold", "value"))
def train_model_performance_metrics(dataset, cutoff):

    def plotly_confusion_matrix(y_true, y_score, cutoff=0.5, labels=None, percentage=True):

        # 計算confusion matrix
        y_pred = np.where(y_score > cutoff, 1, 0)
        cm = confusion_matrix(y_true, y_pred)

        cm_normalized = np.round(100 * cm / cm.sum(), 1)

        if labels is None:
            labels = [str(i) for i in range(cm.shape[0])]

        zmax = cm.sum()

        data = [go.Heatmap(
            z=cm,
            x=[f" {lab}" for lab in labels],
            y=[f" {lab}" for lab in labels],
            hoverinfo="skip",
            zmin=0, zmax=zmax, colorscale='Blues',
            showscale=False,
        )]

        layout = go.Layout(
            title="混淆矩陣(Confusion Matrix)",
            xaxis=dict(title='預測分類',
                       constrain="domain",
                       tickmode='array',
                       tickvals=[f" {lab}" for lab in labels],
                       ticktext=[f" {lab}" for lab in labels]),
            yaxis=dict(title=dict(text='實際分類', standoff=20),
                       autorange="reversed",
                       side='left',
                       scaleanchor='x',
                       scaleratio=1,
                       tickmode='array',
                       tickvals=[f" {lab}" for lab in labels],
                       ticktext=[f" {lab}" for lab in labels]),
            plot_bgcolor='#fff',
        )
        fig = go.Figure(data, layout)
        annotations = []
        for x in range(cm.shape[0]):
            for y in range(cm.shape[1]):
                top_text = f"{cm_normalized[x, y]}%" if percentage else f"{cm[x, y]}"
                bottom_text = f"{cm_normalized[x, y]}%" if not percentage else f"{cm[x, y]}"
                annotations.extend([
                    go.layout.Annotation(
                        x=fig.data[0].x[y],
                        y=fig.data[0].y[x],
                        text=top_text,
                        showarrow=False,
                        font=dict(size=20)
                    ),
                    go.layout.Annotation(
                        x=fig.data[0].x[y],
                        y=fig.data[0].y[x],
                        text=f" <br> <br> <br>({bottom_text})",
                        showarrow=False,
                        font=dict(size=12)
                    )]
                )

        fig.update_layout(annotations=annotations,
                          title_x=0.5,
                          font_family='Noto Sans TC')
        return fig

    if dataset == "train":
        fig = plotly_confusion_matrix(y_true=y_train, y_score=xgboost_train_pred,
                                      cutoff=cutoff, labels=['沒小孩', '有小孩'], percentage=True)
    elif dataset == "test":
        fig = plotly_confusion_matrix(y_true=y_test, y_score=xgboost_test_pred,
                                      cutoff=cutoff, labels=['沒小孩', '有小孩'], percentage=True)

    return fig


# 繪製ROC曲線
@app.callback(Output("model_roc_curve", "figure"),
              Input("model_roc_curve_dataset", "value"),
              Input("predict_threshold", "value"))
def train_model_performance_metrics(dataset, cutoff):

    def plotly_roc_auc_curve(fpr, tpr, thresholds, score, cutoff=None, round=2):
        trace0 = go.Scatter(x=fpr, y=tpr,
                            mode='lines',
                            name='ROC AUC CURVE',
                            text=[f"threshold: {th:.{round}f} <br> FP: {fp:.{round}f} <br> TP: {tp:.{round}f}"
                                  for fp, tp, th in zip(fpr, tpr, thresholds)],
                            hoverinfo="text"
                            )
        data = [trace0]
        layout = go.Layout(title='ROC曲線',
                           #    width=450,
                           #    height=450,
                           xaxis=dict(title='False Positive Rate', range=[0, 1], constrain="domain"),
                           yaxis=dict(title='True Positive Rate', range=[0, 1], constrain="domain",
                                      scaleanchor='x', scaleratio=1),
                           hovermode='closest',
                           plot_bgcolor='#fff', )
        fig = go.Figure(data, layout)
        shapes = [dict(type='line',
                       xref='x',
                       yref='y',
                       x0=0,
                       x1=1,
                       y0=0,
                       y1=1,
                       line=dict(color="darkslategray", width=4, dash="dot"),
                       )]

        if cutoff is not None:
            threshold_idx = np.argmin(np.abs(thresholds - cutoff))
            cutoff_tpr = tpr[threshold_idx]
            cutoff_fpr = fpr[threshold_idx]

            shapes.append(
                dict(type='line', xref='x', yref='y',
                     x0=0, x1=1, y0=cutoff_tpr, y1=cutoff_tpr,
                     line=dict(color="lightslategray", width=1)))
            shapes.append(
                dict(type='line', xref='x', yref='y',
                     x0=cutoff_fpr, x1=cutoff_fpr, y0=0, y1=1,
                     line=dict(color="lightslategray", width=1)))

            annotations = [go.layout.Annotation(x=0.6, y=0.4,
                                                text=f"roc-auc-score: {score:.{round}f}",
                                                showarrow=False, align="right",
                                                xanchor='left', yanchor='top'),
                           go.layout.Annotation(x=0.6, y=0.35,
                                                text=f"cutoff: {cutoff:.{round}f}",
                                                showarrow=False, align="right",
                                                xanchor='left', yanchor='top'),
                           go.layout.Annotation(x=0.6, y=0.3,
                                                text=f"TPR: {cutoff_tpr:.{round}f}",
                                                showarrow=False, align="right",
                                                xanchor='left', yanchor='top'),
                           go.layout.Annotation(x=0.6, y=0.24,
                                                text=f"FPR: {cutoff_fpr:.{round}f}",
                                                showarrow=False, align="right",
                                                xanchor='left', yanchor='top'),
                           ]
            fig.update_layout(annotations=annotations,
                              title_x=0.5,
                              font_family='Noto Sans TC')

        fig.update_layout(shapes=shapes)
        return fig

    if dataset == "train":
        fpr, tpr, thresholds = roc_curve(y_true=y_train, y_score=xgboost_train_pred, pos_label=1)
        score = roc_auc_score(y_true=y_train, y_score=xgboost_train_pred)
        fig = plotly_roc_auc_curve(fpr=fpr, tpr=tpr, thresholds=thresholds, score=score, cutoff=cutoff, round=2)

    elif dataset == "test":
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=xgboost_test_pred, pos_label=1)
        score = roc_auc_score(y_true=y_test, y_score=xgboost_test_pred)
        fig = plotly_roc_auc_curve(fpr=fpr, tpr=tpr, thresholds=thresholds, score=score, cutoff=cutoff, round=2)

    return fig


# 模型分析頁面-依使用者選取的縣市更新鄉鎮市區選項
@app.callback(Output('modelTownDropdown', 'options'),
              Output('modelTownDropdown', 'value'),
              Input('modelCountyDropdown', 'value'))
def townDropdown(countyName):

    townName = villageFeatureInfo[villageFeatureInfo['county'] == countyName]['town'].drop_duplicates().tolist()
    townOptions = [{'label': townName[i], 'value': townName[i]} for i in range(len(townName))]
    townValue = townName[0]
    return townOptions, townValue


# 模型分析頁面-依使用者選取的鄉鎮市區更新村里選項
@app.callback(Output('modelVillageDropdown', 'options'),
              Output('modelVillageDropdown', 'value'),
              Input('modelCountyDropdown', 'value'),
              Input('modelTownDropdown', 'value'))
def villageDropdown(countyName, townName):

    villageName = villageFeatureInfo[(villageFeatureInfo['county'] == countyName) &
                                     (villageFeatureInfo['town'] == townName)]['village'].drop_duplicates().tolist()
    villageOptions = [{'label': villageName[i], 'value': villageName[i]} for i in range(len(villageName))]

    villageValue = villageName[0]
    return villageOptions, villageValue


# 特徵貢獻圖
@app.callback(Output('featureContributionsFig', 'figure'),
              Output('predictProbTable', 'children'),
              Input('modelVillageDropdown', 'value'),
              Input('feature_age', 'value'),
              Input('feature_education', 'value'),
              Input('feature_have_house', 'value'),
              Input('feature_have_disability', 'value'),
              Input('feature_have_low_type', 'value'),
              Input('feature_living_type', 'value'),
              State('modelCountyDropdown', 'value'),
              State('modelTownDropdown', 'value'))
def featureContributionsFig(village, age, education_cd, living_type_cd, new_disability_category,
                            low_type_cd, having_house_type_cd, county, town):

    # 建立特徵資料集
    singleSample = pd.concat([
        pd.DataFrame([[age, int(education_cd), int(living_type_cd), int(new_disability_category),
                       int(low_type_cd), int(having_house_type_cd)]]),
        villageFeatureInfo[(villageFeatureInfo['county'] == county) &
                           (villageFeatureInfo['town'] == town) &
                           (villageFeatureInfo['village'] == village)].drop(columns=['county', 'town', 'village']).reset_index(drop=True),
        laborForceData[(laborForceData['county'] == county) & (laborForceData['age'] == age)]['female_labor_ratio'].reset_index(drop=True)
    ], axis=1)
    singleSample.columns = featureChName
    singleSampleDMatrix = xgb.DMatrix(data=singleSample, feature_names=featureChName)

    # 模型預測
    singleSamplePredict = xgbModel.predict(singleSampleDMatrix)

    # SHAP分析
    single_shap_values = explainer(singleSampleDMatrix)

    # 定義sigmoid函數
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 建立特徵貢獻表
    featureAddProb = sigmoid(single_shap_values.values[0] + single_shap_values.base_values) - sigmoid(
        single_shap_values.base_values)
    contributionDf = pd.DataFrame({'feature': featureChName,
                                   'addProb': featureAddProb,
                                   'measure': 'relative'})
    contributionDf = contributionDf.sort_values('addProb', ascending=False)

    # 加入全樣本平均值資訊
    contributionDf = pd.concat(
        [pd.DataFrame({'feature': '全樣本平均值', 'addProb': sigmoid(single_shap_values.base_values), 'measure': 'absolute'}),
         contributionDf])

    # 加入最後預測資訊資訊
    contributionDf = pd.concat(
        [contributionDf, pd.DataFrame({'feature': '最終預測結果', 'addProb': singleSamplePredict, 'measure': 'total'})])
    contributionDf = contributionDf.reset_index(drop=True)
    contributionDf['addProb'] = round(contributionDf['addProb'] * 100, 2)

    # 繪製特徵貢獻圖
    featureContributionsFig = go.Figure(go.Waterfall(
        x=contributionDf["feature"],
        measure=contributionDf["measure"],
        y=contributionDf["addProb"], base=0,
        decreasing={"marker": {"color": "rgba(50, 200, 50, 1.0)"}},
        increasing={"marker": {"color": "rgba(219, 64, 82, 0.7)"}},
        totals={"marker": {"color": "rgba(230, 230, 30, 1.0)"}}
    ))

    featureContributionsFig.update_layout(title=f"特徵貢獻預測機率瀑布圖 預測有小孩機率={round(singleSamplePredict[0] * 100, 2)}%",
                      xaxis=dict(title="特徵名稱"),
                      yaxis=dict(title="預測機率(%)"),
                      waterfallgap=0.3)

    # 整理預測分類機率表
    predictProbTable = pd.DataFrame(data={'類別': ['有小孩', '沒小孩'],
                                          '預測機率': [f'{round(float(singleSamplePredict)*100,2)}%',
                                                   f'{round((1-float(singleSamplePredict))*100,2)}%']})

    return featureContributionsFig, dbc.Table.from_dataframe(predictProbTable)


# 鄉鎮市區分析頁面-依使用者選取的縣市更新鄉鎮市區選項
@app.callback(Output('townAnalysisTownDropdown', 'options'),
              Output('townAnalysisTownDropdown', 'value'),
              Input('townAnalysisCountyDropdown', 'value'))
def townDropdown(countyName):

    townName = villageFeatureInfo[villageFeatureInfo['county'] == countyName]['town'].drop_duplicates().tolist()
    townOptions = [{'label': townName[i], 'value': townName[i]} for i in range(len(townName))]
    townValue = townName[0]
    return townOptions, townValue


# 鄉鎮市區分析頁面-依使用者選取的鄉鎮市區輸出對應圖片
@app.callback(Output('townFeatureAnalysisFig', 'src'),
              Input('townAnalysisCountyDropdown', 'value'),
              Input('townAnalysisTownDropdown', 'value'))
def townFeatureAnalysisFig(countyName, townName):

    imgFilePath = f'/assets/model_data/town_shap_fig/{countyName}_{townName}.png'

    return imgFilePath


# 輿情分析-議題聲量圖
@app.callback(Output('topicVolumeFig', 'figure'),
              Input('pubicOpinionTopicDropdown', 'value'))
def topicVolumeFig(topic):

    # 至資料庫搜尋議題聲量
    query = f'''
        select time as "日期", quantity as "聲量" from web.sound where issue = "{topic}" 
        and  time > date_sub(curdate(), interval 180 day) order by time;
    '''
    topicVolume = pd.read_sql(query, con=CreateDBEngine())

    # 繪製圖形
    fig = px.line(topicVolume, x="日期", y="聲量")
    fig.update_layout(
        height=300,
        title_text=f'{topic} 議題聲量圖',
        title_x=0.5,
        font_family='Noto Sans TC'
    )

    return fig


# 輿情分析-更新熱門文章
@app.callback(Output('pttGossipingHotArticle', 'children'),
              Output('pttBabyMotherHotArticle', 'children'),
              Output('dcardTrendingHotArticle', 'children'),
              Output('dcardParentChildHotArticle', 'children'),
              Output('babyHomeHotArticle', 'children'),
              Input('pubicOpinionTopicDropdown', 'value'),
              Input('pubicOpinionPeriodDropdown', 'value'))
def updateHotArticle(topic, period):

    # 至資料庫搜尋熱門文章
    query = f'''select time as '日期', board as '板名', title as '文章標題', url as '超連結' from web.popular_article 
    where issue = "{topic}" and sday = "{period}"
    order by time desc;
    '''
    hotArticle = pd.read_sql(query, con=CreateDBEngine())

    # 建立含有超聯結的表
    def generateHotArticleDf(df, boardName):

        df = df[df['板名'] == boardName]
        df_drop_link = df.drop(columns=['超連結', '板名'])
        output = html.Table(
            # Header
            [html.Tr([html.Th(col) for col in df_drop_link.columns])] +

            # Body
            [html.Tr([
                html.Td(df.iloc[i][col]) if col != '文章標題' else html.Td(
                    html.A(href=df.iloc[i]['超連結'], children=df.iloc[i][col], target='_blank')) for col in
                df_drop_link.columns
            ]) for i in range(len(df))]
        )

        return output

    return generateHotArticleDf(hotArticle, "PTT-八卦版"), \
           generateHotArticleDf(hotArticle, "PTT-親子版"), \
           generateHotArticleDf(hotArticle, "Dcard-時事版"), \
           generateHotArticleDf(hotArticle, "Dcard-親子版"), \
           generateHotArticleDf(hotArticle, "BabyHome寶貝家庭親子網")


# 輿情分析-文字雲圖案
@app.callback(Output('wordCloudFig', 'src'),
              Input('pubicOpinionTopicDropdown', 'value'),
              Input('pubicOpinionPeriodDropdown', 'value'))
def wordCloudFig(topic, period):

    imgFilePath = f'/assets/word_cloud/{topic}-{period}.png'

    return imgFilePath


# 輿情分析-關鍵字向量圖
@app.callback(Output('wordVectorFig', 'figure'),
              Input('pubicOpinionTopicDropdown', 'value'),
              Input('pubicOpinionPeriodDropdown', 'value'),
              Input('wordVectorRadio', 'value'))
def wordVectorFig(topic, period, dimension):

    if dimension == '2D':

        # 至資料庫搜尋關鍵字向量資料
        query = f'select label, x, y from web.plot2d where issue = "{topic}" and sday = "{period}"'
        wordVector = pd.read_sql(query, con=CreateDBEngine())

        # 繪製圖形
        fig = px.scatter(wordVector, x="x", y="y", text="label", size_max=60)
        fig.update_traces(textposition='top center')
        fig.update_layout(
            height=550,
            title_text=f'議題: {topic}  觀察期間: {period} 天',
            title_x=0.5,
            font_family='Noto Sans TC'
        )

    elif dimension == '3D':

        # 至資料庫搜尋關鍵字向量資料
        query = f'select label, x, y, z from web.plot3d where issue = "{topic}" and sday = "{period}"'
        wordVector = pd.read_sql(query, con=CreateDBEngine())

        # 繪製圖形
        fig = px.scatter_3d(wordVector, x="x", y="y", z="z", text="label", size_max=60)
        fig.update_traces(textposition='top center')
        fig.update_layout(
            height=550,
            title_text=f'議題: {topic}  觀察期間: {period} 天',
            title_x=0.5,
            font_family='Noto Sans TC'
        )

    return fig
