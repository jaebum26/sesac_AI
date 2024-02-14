import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import cv2
import numpy as np
import webbrowser
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

class CustomVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def process(self, frame):
        self.frame_count += 1
        # 이미지 좌우 반전
        flipped_frame = cv2.flip(frame.data, 1)
        # 밝기 증가
        brightness_factor = 1.2
        brightened_frame = cv2.convertScaleAbs(flipped_frame, alpha=brightness_factor, beta=0)
        return brightened_frame

# Streamlit의 SessionState 모듈을 사용하여 상태 유지
class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

# select_page 함수에서 사용할 상태 변수를 담는 session_state 생성
session_state = SessionState(page_state="메인 페이지")

# 제목을 가운데 정렬하는 함수
def centered_title(title_text):
    return f"<h1 style='text-align:center;'>{title_text}</h1>"

# 메인 페이지
def main_page():
    st.markdown(centered_title('Main Page'), unsafe_allow_html=True)
    st.write('반갑습니다. 저희 사이트를 방문해주셔서 감사합니다.')

# 걸어서 세계속으로 페이지
def walk_the_world():
    st.markdown(centered_title('걸어서 세계속으로'), unsafe_allow_html=True)
    st.write('설명')

    # 설명서 버튼
    if st.button('설명서 보기'):
        # 설명서 HTML 파일 경로
        documentation_path = 'C:/Users/blucom005/Downloads/정리폴더/24년도/프로젝트 문서/manual.html'

        # 새로운 브라우저 창에서 HTML 파일 열기
        webbrowser.open('file://' + documentation_path, new=2)

    # streamlit_webrtc를 사용하여 웹캠 표시
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=CustomVideoProcessor,  # CustomVideoProcessor를 사용하여 좌우 반전 및 밝기 조절
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        ),
    )

# 주제2 페이지
def topic_2():
    st.markdown(centered_title('주제2'), unsafe_allow_html=True)
    st.write('설명')

# 페이지 선택을 캐시하지 않고 직접 실행
with st.sidebar:
    page_state = option_menu("메뉴", ["메인 페이지", "걸어서 세계속으로", "주제2"],
                             icons=['house', 'camera fill', 'kanban'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
                                 "container": {"padding": "5!important", "background-color": "#fafafa"},
                                 "icon": {"color": "orange", "font-size": "25px"},
                                 "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                 "nav-link-selected": {"background-color": "#02ab21"},
                             })

# 페이지에 따라 다른 내용 표시
if page_state == "메인 페이지":
    main_page()
elif page_state == "걸어서 세계속으로":
    walk_the_world()
elif page_state == "주제2":
    topic_2()

# 터미널 명령어 : python -m streamlit run mainPage_v2.py