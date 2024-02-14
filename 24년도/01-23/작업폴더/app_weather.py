from flask import Flask, render_template
import requests
import pandas as pd

app = Flask(__name__)

def get_weather_data():
    api_key = 'cWxAPGn6IGENYq%2B%2B6SXLcFi%2FERX3%2Ff7TfoWBSaOXUP4epY38VtF3V2ZKtSOyCN1DpvDlwlWg2ZSZeTj8zGcEfQ%3D%3D'  # 기상청 API 키를 입력하세요
    
    # 서울 지역의 기상 정보만 가져오도록 수정
    api_url = f'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList?serviceKey={api_key}&pageNo=1&numOfRows=10&dataType=JSON&dataCd=ASOS&dateCd=DAY&startDt=20230101&endDt=20231231&stnIds=108'

    try:
        response = requests.get(api_url)
        response.raise_for_status()  # 오류 발생 시 예외 처리
        data = response.json()
    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
        data = None
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
        data = None
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
        data = None
    except requests.exceptions.RequestException as err:
        print("Oops! Something went wrong:", err)
        data = None

    # 응답이 JSON 형식이 아닐 경우 예외 처리
    if not isinstance(data, dict):
        print("Invalid JSON format in response.")
        return []

    # 오류 처리: 'body' 키 확인
    if 'body' not in data['response']:
        return []

    # 데이터 가공
    items = data['response']['body']['items']['item']
    df = pd.DataFrame(items)

    # 필요한 컬럼만 선택
    df = df[['stnId', 'tm', 'maxTa', 'minTa', 'sumRn']]

    # 컬럼 이름 변경
    df.columns = ['지점', '날짜', '최고기온', '최저기온', '강수량']

    # '지점' 컬럼 값을 '서울'로 변경
    df['지점'] = '서울'

    # 템플릿에 전달할 데이터
    weather_data = df.to_dict(orient='records')

    return weather_data

@app.route('/')
def home():
    # 항상 서울 지역의 기상 정보를 가져옴
    weather_data = get_weather_data()
    return render_template('weather.html', datas=weather_data)

if __name__ == '__main__':
    app.run(debug=True)
