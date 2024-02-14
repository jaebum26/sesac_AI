import requests
from gtts import gTTS
from playsound import playsound

def get_country_code_from_api(country_name):
    # 실제로는 해당 API에 따라 구현되어야 함
    # 여기에서는 간단히 더미 데이터를 사용
    country_code_data = {
        "인도네시아": "id",
        "일본": "ja",
        "그리스": "el",
        "중국": "zh-CN",
        "러시아": "ru",
        "태국": "th",
        "아일랜드": "ga",
        "터키": "tr",
        "프랑스": "fr",
        "몽골": "mn",
        "필리핀": "fil",
        "하와이": "haw",
        "이스라엘": "he",
        "베트남": "vi",
        "인도": "hi",
        "이탈리아": "it",  # 새로 추가
    }

    return country_code_data.get(country_name)

def get_country_language_code(api_key, location):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"

    params = {
        "latlng": location,
        "key": api_key,
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if data["status"] == "OK":
        results = data["results"]
        if results:
            address_components = results[0]["address_components"]

            # 국가 정보 찾기
            country = next((comp["long_name"] for comp in address_components if "country" in comp["types"]), None)

            if country:
                # 국가 코드를 얻는 API 호출
                country_code = get_country_code_from_api(country)

                if country_code:
                    # 국가 코드에 해당하는 인사말을 언어로 출력
                    speak_with_country_code(country_code)
                else:
                    print(f"Language code for {country} not found.")
            else:
                print("Country information not found in the response.")
        else:
            print("No results found.")
    else:
        print(f"Error: {data['status']}")

# gTTS를 사용하여 음성 출력
def speak_with_country_code(country_code):
    # 국가 코드와 대응되는 인사말 딕셔너리
    country_greetings = {
        "id": "Halo",  # 인도네시아어
        "ja": "こんにちは",  # 일본어
        "el": "Γειά σου",  # 그리스어
        "zh-CN": "你好",  # 중국어 (간체)
        "ru": "привет",  # 러시아어
        "th": "สวัสดี",  # 태국어
        "ga": "Dia dhuit",  # 아일랜드어
        "tr": "Merhaba",  # 터키어
        "fr": "Bonjour",  # 프랑스어
        "mn": "Сайн уу",  # 몽골어
        "fil": "Kamusta",  # 필리핀어
        "haw": "Aloha",  # 하와이어
        "he": "שלום",  # 이스라엘어
        "vi": "Xin chào",  # 베트남어
        "hi": "नमस्ते",  # 힌디어 (인도어)
        "it": "Ciao",  # 이탈리아어
    }

    greeting_text = country_greetings.get(country_code, "Unknown greeting")
    tts = gTTS(text=greeting_text, lang=country_code)
    tts.save("greeting.mp3")
    playsound("greeting.mp3")

# 사용 예시
api_key = "YOUR_GOOGLE_MAPS_API_KEY"
location = "40.714224,-73.961452"

# 국가의 언어 코드 얻고, 해당 언어로 음성 출력
get_country_language_code(api_key, location)
