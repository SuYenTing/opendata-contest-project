# 2021資料創新應用競賽內政黑客松組 - Circle of Life: 少子女化保衛戰

* [2021資料創新應用競賽官方網站](https://opendata-contest.tca.org.tw/)

![](https://github.com/SuYenTing/opendata-contest-project/blob/main/image/poster.png)

## 服務相關連結

* [Circle of Life: 少子女化保衛戰](http://3.13.171.111/)
* [競賽閃電秀影片](https://youtu.be/MyD9GHOsgio)
* [專題PPT](https://github.com/SuYenTing/opendata-contest-project/blob/main/ppt/presentation_ppt.pdf)

## 服務架構

![](https://github.com/SuYenTing/opendata-contest-project/blob/main/image/architecture.png)

## 程式檔案說明

* data: 資料整理程式碼資料夾
    * family_budget.py: 平均每戶家庭收支按區域別分
    * fertility_rate.py: 整理育齡婦女之年齡別生育率及生第一胎平均年齡
    * home_ownership_rate.py: 房屋自有率
    * kindergarten.py: 幼兒(稚)園概況表
    * kindergarten_info_crawler.py: 全國教保資訊網幼兒園基本查詢爬蟲程式
    * labor_force_participation_rate.py: 勞動力參與率
    * postal_code.py: 郵遞區號

* PowerBI: 資料視覺化
    * main.pbix: 少子女化指標數據視覺化
    * data_preparation.ipynb: 視覺化資料之數據檔案

* model: 模型預測與分析
    * experiment.ipynb: 早期版本，預測各鄉鎮市區出生率
    * main.ipynb: 目前版本，以內政部大數據模擬資料為基底做預測

* web: 網站程式及Docker部署設定檔案
    * app: web程式碼(Python Dash)
    * docker-compose.yml
    * dockerfile-dash
