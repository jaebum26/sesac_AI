import requests
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
import time
import os
import pygame

# prevLoc 초기화
prevLoc = []

api_key = "AIzaSyC1Ax9tlpuKI8qC7hh0RVnUNkJYrAOJe10"

def play_mp3(mp3_file):
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def speech_recognator(firstLocation, condition):
    api_key = "AIzaSyC1Ax9tlpuKI8qC7hh0RVnUNkJYrAOJe10"
    # 1) 음성 인식기
    condition = True
    while condition == True:
        try:
            time.sleep(1)
            print("record process is on")
            r = sr.Recognizer() # 음성 인식을 위한 객체 생성
            
            mic = sr.Microphone(device_index = 1)
                # 마이크 객체 선언, 인덱스는 각 노트북의 마이크 번호를 의미합니다. 만약 인식이 안되시면 바꿔보시면서 테스트 해보시면 될 듯 합니다.
            with mic as source:
                audio = r.listen(source,timeout=5, phrase_time_limit = 5) # 마이크에서 5초 동안 음성을 듣고 audio 변수에 저장합니다.

            result = r.recognize_google(audio, language = "ko-KR",show_all=True) # 인식한 음성을 텍스트로 변환

            confidence = float(result['alternative'][0]['confidence'])
            if confidence > 0.85:
                result1 = result['alternative'][0]['transcript']
                print(f"입력 받았습니다. {result1}로 이동합니다.")
                values = geocode_address(result1,api_key)
                if values[0] == True:
                    location2 = values[1:3]
                    print(f"경도와 위도를 반환합니다 : {location2}")
                    firstLocation[:] = location2
                    prevLoc = location2
                    print(values[3])
                    # print(firstLocation)
                    country_code = get_country_code_from_api(values[3])
                    print(country_code)
                    speak_with_country_code(country_code)  # 국가 코드를 전달하여 음성 출력
        except:
            try:
                firstLocation[:] = prevLoc
                
            except:   
                pass
        time.sleep(0.1)

def geocode_address(address, api_key):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"

    # Geocoding API에 요청 보내기
    response = requests.get(
        base_url,
        params={
            "address": address,
            "key": api_key,
        }
    )

    #print(response.json())
    # 응답 확인
    if response.status_code == 200:
        # JSON 응답 파싱
        data = response.json()

        # 결과 확인
        if data["status"] == "OK":
            # 첫 번째 결과의 위도와 경도 추출
            location1 = data["results"][0]["geometry"]["location"]
            latitude = location1["lat"]
            longitude = location1["lng"]

            return [True,latitude, longitude,data]
        else:
            print(f"Geocoding API error: {data['status']} - {data.get('error_message')}")
    else:
        print(f"Geocoding API request failed with status code: {response.status_code}")

    return [False]

def get_country_code_from_api(api_response):
    # 국가 코드를 얻는 실제 API 호출 및 파싱이 이곳에 들어가야 합니다.
    # 여기서는 간단히 첫 번째 결과의 국가 코드를 추출하도록 가정합니다.
    # print(api_response)
    country_code = ""
    for component in api_response['results'][0]['address_components']:
        if 'country' in component['types']:
            country_code = component.get("short_name", "")
            break

    print(country_code)
    # print(country_code)
    country_code = str(country_code)
    # print(country_code)
    return country_code.lower()

# 국가 코드와 대응되는 인사말 딕셔너리
country_greetings = {
    "id": "Halo",  # 인도네시아어
    "jp": "こんにちは",  # 일본어
    "gr": "Γειά σου",  # 그리스어
    "cn": "你好",  # 중국어 (간체)
    "ru": "привет",  # 러시아어
    "th": "สวัสดี",  # 태국어
    "tr": "Merhaba",  # 터키어
    "fr": "Bonjour",  # 프랑스어
    "ph": "Kamusta",  # 필리핀어
    "vn": "Xin chào",  # 베트남어
    "in": "नमस्ते",  # 힌디어 (인도어)
    "it": "Ciao",  # 이탈리아어
    "de": "Hallo" #독일
}

# .mp3파일을 이용한 음성 출력
def speak_with_country_code(country_code):
    if country_code == "kr":
        # print(country_code)
        play_mp3("./안녕하세요.mp3")
        time.sleep(1)
    elif country_code in country_greetings:
        file_path = f"{country_code}.mp3"
        play_mp3(file_path)
        time.sleep(1)
    else:
        print(country_code)
        play_mp3("./hello.mp3")
        time.sleep(1)

# 초기 지역 설정
initial_location = ["Seoul, South Korea"]  # 예시로 초기값을 서울로 설정

# 음성 처리 시작
speech_recognator(initial_location, True)