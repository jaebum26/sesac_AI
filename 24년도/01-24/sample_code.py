# app.py

import streamlit as st

# 애플리케이션 제목
st.title('간단한 Streamlit 예제')

# 텍스트 입력 받기
user_input = st.text_input('당신의 이름을 입력하세요:', '이름')

# 버튼을 클릭하면 메시지 출력
if st.button('인사하기'):
    st.write(f'안녕하세요, {user_input}!')