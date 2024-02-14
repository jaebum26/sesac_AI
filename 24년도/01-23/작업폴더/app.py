# app.py
from flask import Flask, render_template
import pandas as pd
import chardet

app = Flask(__name__)

# datas = [
#         {'name': '반원', "level": 60, "point": 360, "exp": 45000},
#         {'name': '반원2', "level": 2, "point": 20, "exp": 200},
#         {'name': '반원3', "level": 3, "point": 30, "exp": 300}
#     ]

rawdata = open('C:/Users/blucom005/Downloads/정리폴더/24년도/01-23/작업폴더/data/data.csv', 'rb').read()
result = chardet.detect(rawdata)
encoding = result['encoding']

datas = pd.read_csv('C:/Users/blucom005/Downloads/정리폴더/24년도/01-23/작업폴더/data/data.csv', encoding=encoding)
datas = datas.to_dict(orient='records')

@app.route('/')
def index():
    return render_template('index.html', datas=datas)

@app.route('/index_table')
def index_table():
    return render_template('index_table.html', datas=datas)

if __name__=="__main__":
  app.run(debug=True)