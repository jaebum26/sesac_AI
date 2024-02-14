from gtts import gTTS
from playsound import playsound

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

# gTTS를 사용하여 음성 출력
def speak_with_country_code(country_code):
    greeting_text = country_greetings.get(country_code, "Unknown greeting")
    tts = gTTS(text=greeting_text, lang=country_code)
    tts.save("greeting.mp3")
    playsound("greeting.mp3")

# 사용 예시
country_code = input("국가 코드를 입력하세요 (예: 인도네시아어의 경우 'id'): ")

# 국가 코드에 해당하는 인사말을 언어로 출력
speak_with_country_code(country_code)