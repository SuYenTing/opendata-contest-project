# 自定義函數
# 匯入套件
import json
from sqlalchemy import create_engine


# 設定MySQL連線函數
def CreateDBEngine():
    secretFile = json.load(open('dbToken.json', 'r'))
    host = secretFile['host']
    username = secretFile['user']
    password = secretFile['password']
    port = secretFile['port']
    database = secretFile['dbName']
    return create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}', echo=False)
