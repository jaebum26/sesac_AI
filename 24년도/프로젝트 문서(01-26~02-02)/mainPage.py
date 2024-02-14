import streamlit as st

# 상태 변수
page_state = st.sidebar.radio("주제 선택", ["메인 페이지","걸어서 세계속으로", "주제2"])

if page_state == "메인 페이지":
    st.title('Main Page')
    st.write('반갑습니다. 저희 사이트를 방문해주셔서 감사합니다.')

# 페이지 1의 내용
elif page_state == "걸어서 세계속으로":
    st.title('걸어서 세계속으로')
    st.write('설명')

# 페이지 2의 내용
elif page_state == "주제2":
    st.title('주제2')
    st.write('설명')

# 터미널 명령어 : python -m streamlit run mainPage.py