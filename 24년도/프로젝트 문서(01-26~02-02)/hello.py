from gtts import gTTS
import pygame
import time

def save_greetings_by_country(country_greetings):
    for country_code, greeting_text in country_greetings.items():
        tts = gTTS(text=greeting_text, lang=country_code)
        file_name = f"{country_code}.mp3"
        tts.save(file_name)

# 예시로 주어진 country_greetings 딕셔너리 사용
country_greetings = {
    "id": "Halo",
    "ja": "こんにちは",
    "el": "Γειά σου",
    "zh-CN": "你好",
    "ru": "привет",
    "th": "สวัสดี",
    "tr": "Merhaba",
    "fr": "Bonjour",
    "hi": "Aloha",
    "vi": "Xin chào",
    "it": "Ciao",
    "de": "Hallo"
}

# 각 나라의 인사말을 국가 코드로 저장
save_greetings_by_country(country_greetings)
